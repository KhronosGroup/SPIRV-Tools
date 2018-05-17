// Copyright (c) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "validate.h"

#include <algorithm>
#include <string>

#include "diagnostic.h"
#include "opcode.h"
#include "spirv_target_env.h"
#include "val/validation_state.h"

using libspirv::Decoration;
using libspirv::DiagnosticStream;
using libspirv::Instruction;
using libspirv::ValidationState_t;

namespace {

// Returns whether the given variable has a BuiltIn decoration.
bool isBuiltInVar(uint32_t var_id, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(var_id);
  return std::any_of(
      decorations.begin(), decorations.end(),
      [](const Decoration& d) { return SpvDecorationBuiltIn == d.dec_type(); });
}

// Returns whether the given structure type has any members with BuiltIn
// decoration.
bool isBuiltInStruct(uint32_t struct_id, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(struct_id);
  return std::any_of(
      decorations.begin(), decorations.end(), [](const Decoration& d) {
        return SpvDecorationBuiltIn == d.dec_type() &&
               Decoration::kInvalidMember != d.struct_member_index();
      });
}

// Returns true if the given ID has the Import LinkageAttributes decoration.
bool hasImportLinkageAttribute(uint32_t id, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(id);
  return std::any_of(decorations.begin(), decorations.end(),
                     [](const Decoration& d) {
                       return SpvDecorationLinkageAttributes == d.dec_type() &&
                              d.params().size() >= 2u &&
                              d.params().back() == SpvLinkageTypeImport;
                     });
}

// Returns a vector of all members of a structure.
std::vector<uint32_t> getStructMembers(uint32_t struct_id,
                                       ValidationState_t& vstate) {
  const auto inst = vstate.FindDef(struct_id);
  return std::vector<uint32_t>(inst->words().begin() + 2, inst->words().end());
}

// Returns a vector of all members of a structure that have specific type.
std::vector<uint32_t> getStructMembers(uint32_t struct_id, SpvOp type,
                                       ValidationState_t& vstate) {
  std::vector<uint32_t> members;
  for (auto id : getStructMembers(struct_id, vstate)) {
    if (type == vstate.FindDef(id)->opcode()) {
      members.push_back(id);
    }
  }
  return members;
}

// Returns whether the given structure is missing Offset decoration for any
// member. Handles also nested structures.
bool isMissingOffsetInStruct(uint32_t struct_id, ValidationState_t& vstate) {
  std::vector<bool> hasOffset(getStructMembers(struct_id, vstate).size(),
                              false);
  // Check offsets of member decorations
  for (auto& decoration : vstate.id_decorations(struct_id)) {
    if (SpvDecorationOffset == decoration.dec_type() &&
        Decoration::kInvalidMember != decoration.struct_member_index()) {
      hasOffset[decoration.struct_member_index()] = true;
    }
  }
  // Check also nested structures
  bool nestedStructsMissingOffset = false;
  for (auto id : getStructMembers(struct_id, SpvOpTypeStruct, vstate)) {
    if (isMissingOffsetInStruct(id, vstate)) {
      nestedStructsMissingOffset = true;
      break;
    }
  }
  return nestedStructsMissingOffset ||
         !std::all_of(hasOffset.begin(), hasOffset.end(),
                      [](const bool b) { return b; });
}

// Rounds x up to the next alignment. Assumes alignment is a power of two.
uint32_t align(uint32_t x, uint32_t alignment) {
  return (x + alignment - 1) & ~(alignment - 1);
}

// Returns std140/430 base alignment of struct member.
uint32_t getBaseAlignment(uint32_t member_id, bool std140,
                          ValidationState_t& vstate) {
  const auto inst = vstate.FindDef(member_id);
  const auto words = inst->words();
  uint32_t baseAlignment = 0;
  switch (inst->opcode()) {
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
      baseAlignment = words[2] / 8;
      break;
    case SpvOpTypeVector: {
      const auto componentId = words[2];
      const auto numComponents = words[3];
      const auto componentSize = vstate.FindDef(componentId)->words()[2] / 8;
      baseAlignment = componentSize * (numComponents == 3 ? 4 : numComponents);
    } break;
    case SpvOpTypeMatrix:
    case SpvOpTypeArray:
      baseAlignment = getBaseAlignment(words[2], std140, vstate);
      if (std140) baseAlignment = align(baseAlignment, 16u);
      break;
    case SpvOpTypeStruct:
      for (auto id : getStructMembers(member_id, vstate)) {
        baseAlignment =
            std::max(baseAlignment, getBaseAlignment(id, std140, vstate));
      }
      if (std140) baseAlignment = align(baseAlignment, 16u);
      break;
    default:
      assert(0);
      break;
  }

  return baseAlignment;
}

// Returns std140/430 struct member size.
uint32_t getMemberSize(uint32_t member_id, bool std140,
                       ValidationState_t& vstate, bool insideArray = false) {
  const auto inst = vstate.FindDef(member_id);
  const auto words = inst->words();
  auto baseAlignment = getBaseAlignment(member_id, std140, vstate);
  if (insideArray && std140) baseAlignment = align(baseAlignment, 16u);
  uint32_t size = 0;
  switch (inst->opcode()) {
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
      return baseAlignment;
    case SpvOpTypeVector: {
      const auto componentId = words[2];
      const auto numComponents = words[3];
      const auto componentSize = vstate.FindDef(componentId)->words()[2] / 8;
      size = componentSize * numComponents;
      if (insideArray && std140) size = align(size, 16u);
      return size;
    }
    case SpvOpTypeArray:
      return vstate.FindDef(words[3])->words()[3] *
             getMemberSize(vstate.FindDef(member_id)->words()[2], std140,
                           vstate, true);
    case SpvOpTypeMatrix:
      return words[3] * baseAlignment;
    case SpvOpTypeStruct:
      for (auto id : getStructMembers(member_id, vstate)) {
        size = align(size, getBaseAlignment(id, std140, vstate));
        size += getMemberSize(id, std140, vstate);
      }
      size = align(size, baseAlignment);
      return size;
    default:
      assert(0);
      return 0;
  }
}

// Checks if struct offsets and strides are std140 or std430 compliant.
bool checkStd140Or430(uint32_t struct_id, bool std140,
                      ValidationState_t& vstate) {
  uint32_t offset = 0;
  const auto members = getStructMembers(struct_id, vstate);
  for (size_t memberIdx = 0; memberIdx < members.size(); memberIdx++) {
    auto id = members[memberIdx];
    uint32_t decOffset = 0xffffffff;
    for (auto& decoration : vstate.id_decorations(struct_id)) {
      if (SpvDecorationOffset == decoration.dec_type() &&
          decoration.struct_member_index() == (int)memberIdx) {
        decOffset = decoration.params()[0];
      }
    }
    if (decOffset == 0xffffffff) return false;
    const auto inst = vstate.FindDef(id);
    if (SpvOpTypeRuntimeArray == inst->opcode()) {
      // Size of runtime array is unknown, thus we cannot continue validation.
      return true;
    }
    if (SpvOpTypeStruct == inst->opcode() &&
        !checkStd140Or430(id, std140, vstate)) {
      return false;
    }
    if (SpvOpTypeArray == inst->opcode()) {
      const auto typeId = inst->words()[2];
      const auto arrayInst = vstate.FindDef(typeId);
      if (SpvOpTypeStruct == arrayInst->opcode() &&
          !checkStd140Or430(typeId, std140, vstate)) {
        return false;
      }
      // Check array stride
      for (auto& decoration : vstate.id_decorations(id)) {
        if (SpvDecorationArrayStride == decoration.dec_type() &&
            decoration.params()[0] !=
                getMemberSize(vstate.FindDef(id)->words()[2], std140, vstate,
                              true)) {
          return false;
        }
      }
    }
    offset = align(offset, getBaseAlignment(id, std140, vstate));
    if (offset != decOffset) {
      return false;
    }
    offset += getMemberSize(id, std140, vstate);
  }
  return true;
}

// Returns true if structure id has given decoration. Handles also nested
// structures.
bool hasDecoration(uint32_t struct_id, SpvDecoration decoration,
                   ValidationState_t& vstate) {
  for (auto& dec : vstate.id_decorations(struct_id)) {
    if (decoration == dec.dec_type()) return true;
  }
  for (auto id : getStructMembers(struct_id, SpvOpTypeStruct, vstate)) {
    if (hasDecoration(id, decoration, vstate)) {
      return true;
    }
  }
  return false;
}

// Returns true if all ids of given type have a specified decoration.
bool checkForRequiredDecoration(uint32_t struct_id, SpvDecoration decoration,
                                SpvOp type, ValidationState_t& vstate) {
  const auto members = getStructMembers(struct_id, vstate);
  for (size_t memberIdx = 0; memberIdx < members.size(); memberIdx++) {
    const auto id = members[memberIdx];
    if (type != vstate.FindDef(id)->opcode()) continue;
    bool found = false;
    for (auto& dec : vstate.id_decorations(id)) {
      if (decoration == dec.dec_type()) found = true;
    }
    for (auto& dec : vstate.id_decorations(struct_id)) {
      if (decoration == dec.dec_type() &&
          (int)memberIdx == dec.struct_member_index()) {
        found = true;
      }
    }
    if (!found) {
      return false;
    }
  }
  for (auto id : getStructMembers(struct_id, SpvOpTypeStruct, vstate)) {
    if (!checkForRequiredDecoration(id, decoration, type, vstate)) {
      return false;
    }
  }
  return true;
}

spv_result_t CheckLinkageAttrOfFunctions(ValidationState_t& vstate) {
  for (const auto& function : vstate.functions()) {
    if (function.block_count() == 0u) {
      // A function declaration (an OpFunction with no basic blocks), must have
      // a Linkage Attributes Decoration with the Import Linkage Type.
      if (!hasImportLinkageAttribute(function.id(), vstate)) {
        return vstate.diag(SPV_ERROR_INVALID_BINARY)
               << "Function declaration (id " << function.id()
               << ") must have a LinkageAttributes decoration with the Import "
                  "Linkage type.";
      }
    } else {
      if (hasImportLinkageAttribute(function.id(), vstate)) {
        return vstate.diag(SPV_ERROR_INVALID_BINARY)
               << "Function definition (id " << function.id()
               << ") may not be decorated with Import Linkage type.";
      }
    }
  }
  return SPV_SUCCESS;
}

// Checks whether an imported variable is initialized by this module.
spv_result_t CheckImportedVariableInitialization(ValidationState_t& vstate) {
  // According the SPIR-V Spec 2.16.1, it is illegal to initialize an imported
  // variable. This means that a module-scope OpVariable with initialization
  // value cannot be marked with the Import Linkage Type (import type id = 1).
  for (auto global_var_id : vstate.global_vars()) {
    // Initializer <id> is an optional argument for OpVariable. If initializer
    // <id> is present, the instruction will have 5 words.
    auto variable_instr = vstate.FindDef(global_var_id);
    if (variable_instr->words().size() == 5u &&
        hasImportLinkageAttribute(global_var_id, vstate)) {
      return vstate.diag(SPV_ERROR_INVALID_ID)
             << "A module-scope OpVariable with initialization value "
                "cannot be marked with the Import Linkage Type.";
    }
  }
  return SPV_SUCCESS;
}

// Checks whether a builtin variable is valid.
spv_result_t CheckBuiltInVariable(uint32_t var_id, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(var_id);
  for (const auto& d : decorations) {
    if (spvIsVulkanEnv(vstate.context()->target_env)) {
      if (d.dec_type() == SpvDecorationLocation ||
          d.dec_type() == SpvDecorationComponent) {
        return vstate.diag(SPV_ERROR_INVALID_ID)
               << "A BuiltIn variable (id " << var_id
               << ") cannot have any Location or Component decorations";
      }
    }
  }
  return SPV_SUCCESS;
}

// Checks whether proper decorations have been appied to the entry points.
spv_result_t CheckDecorationsOfEntryPoints(ValidationState_t& vstate) {
  for (uint32_t entry_point : vstate.entry_points()) {
    const auto& descs = vstate.entry_point_descriptions(entry_point);
    int num_builtin_inputs = 0;
    int num_builtin_outputs = 0;
    for (const auto& desc : descs) {
      for (auto interface : desc.interfaces) {
        Instruction* var_instr = vstate.FindDef(interface);
        if (SpvOpVariable != var_instr->opcode()) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "Interfaces passed to OpEntryPoint must be of type "
                    "OpTypeVariable. Found Op"
                 << spvOpcodeString(static_cast<SpvOp>(var_instr->opcode()))
                 << ".";
        }
        const uint32_t ptr_id = var_instr->word(1);
        Instruction* ptr_instr = vstate.FindDef(ptr_id);
        // It is guaranteed (by validator ID checks) that ptr_instr is
        // OpTypePointer. Word 3 of this instruction is the type being pointed
        // to.
        const uint32_t type_id = ptr_instr->word(3);
        Instruction* type_instr = vstate.FindDef(type_id);
        const auto storage_class =
            static_cast<SpvStorageClass>(var_instr->word(3));
        if (storage_class != SpvStorageClassInput &&
            storage_class != SpvStorageClassOutput) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "OpEntryPoint interfaces must be OpVariables with "
                    "Storage Class of Input(1) or Output(3). Found Storage "
                    "Class "
                 << storage_class << " for Entry Point id " << entry_point
                 << ".";
        }
        if (type_instr && SpvOpTypeStruct == type_instr->opcode() &&
            isBuiltInStruct(type_id, vstate)) {
          if (storage_class == SpvStorageClassInput) ++num_builtin_inputs;
          if (storage_class == SpvStorageClassOutput) ++num_builtin_outputs;
          if (num_builtin_inputs > 1 || num_builtin_outputs > 1) break;
          if (auto error = CheckBuiltInVariable(interface, vstate))
            return error;
        } else if (isBuiltInVar(interface, vstate)) {
          if (auto error = CheckBuiltInVariable(interface, vstate))
            return error;
        }
      }
      if (num_builtin_inputs > 1 || num_builtin_outputs > 1) {
        return vstate.diag(SPV_ERROR_INVALID_BINARY)
               << "There must be at most one object per Storage Class that can "
                  "contain a structure type containing members decorated with "
                  "BuiltIn, consumed per entry-point. Entry Point id "
               << entry_point << " does not meet this requirement.";
      }
      // The LinkageAttributes Decoration cannot be applied to functions
      // targeted by an OpEntryPoint instruction
      for (auto& decoration : vstate.id_decorations(entry_point)) {
        if (SpvDecorationLinkageAttributes == decoration.dec_type()) {
          const char* linkage_name =
              reinterpret_cast<const char*>(&decoration.params()[0]);
          return vstate.diag(SPV_ERROR_INVALID_BINARY)
                 << "The LinkageAttributes Decoration (Linkage name: "
                 << linkage_name << ") cannot be applied to function id "
                 << entry_point
                 << " because it is targeted by an OpEntryPoint instruction.";
        }
      }
    }
  }
  return SPV_SUCCESS;
}

spv_result_t CheckDescriptorSetArrayOfArrays(ValidationState_t& vstate) {
  for (const auto& def : vstate.all_definitions()) {
    const auto inst = def.second;
    if (SpvOpVariable != inst->opcode()) continue;

    // Verify this variable is a DescriptorSet
    bool has_descriptor_set = false;
    for (const auto& decoration : vstate.id_decorations(def.first)) {
      if (SpvDecorationDescriptorSet == decoration.dec_type()) {
        has_descriptor_set = true;
        break;
      }
    }
    if (!has_descriptor_set) continue;

    const auto& words = inst->words();
    const auto* ptrInst = vstate.FindDef(words[1]);
    assert(SpvOpTypePointer == ptrInst->opcode());

    // Check for a first level array
    const auto typePtr = vstate.FindDef(ptrInst->words()[3]);
    if (SpvOpTypeRuntimeArray != typePtr->opcode() &&
        SpvOpTypeArray != typePtr->opcode()) {
      continue;
    }

    // Check for a second level array
    const auto secondaryTypePtr = vstate.FindDef(typePtr->words()[2]);
    if (SpvOpTypeRuntimeArray == secondaryTypePtr->opcode() ||
        SpvOpTypeArray == secondaryTypePtr->opcode()) {
      return vstate.diag(SPV_ERROR_INVALID_ID)
             << "Only a single level of array is allowed for descriptor "
                "set variables";
    }
  }
  return SPV_SUCCESS;
}

spv_result_t CheckDecorationsOfBuffers(ValidationState_t& vstate) {
  for (const auto& id_dec : vstate.id_decorations()) {
    const uint32_t id = id_dec.first;
    const auto& decorations = id_dec.second;
    if (SpvOpTypeStruct != vstate.FindDef(id)->opcode()) continue;
    for (const auto& dec : decorations) {
      const bool isBlock = SpvDecorationBlock == dec.dec_type();
      const bool isBufferBlock = SpvDecorationBufferBlock == dec.dec_type();
      if (isBlock || isBufferBlock) {
        std::string dec_str = isBlock ? "Block" : "BufferBlock";
        if (isMissingOffsetInStruct(id, vstate)) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "Structure id " << id << " decorated as " << dec_str
                 << " must be explicitly laid out with Offset decorations.";
        } else if (isBlock && !checkStd140Or430(id, true, vstate)) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "Structure id " << id << " decorated as Block"
                 << " must follow std140 alignment rules.";
        } else if (isBufferBlock && !checkStd140Or430(id, false, vstate)) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "Structure id " << id << " decorated as BufferBlock"
                 << " must follow std430 alignment rules.";
        } else if (hasDecoration(id, SpvDecorationGLSLShared, vstate)) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "Structure id " << id << " decorated as " << dec_str
                 << " must not use GLSLShared decoration.";
        } else if (hasDecoration(id, SpvDecorationGLSLPacked, vstate)) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "Structure id " << id << " decorated as " << dec_str
                 << " must not use GLSLPacked decoration.";
        } else if (!checkForRequiredDecoration(id, SpvDecorationArrayStride,
                                               SpvOpTypeArray, vstate)) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "Structure id " << id << " decorated as " << dec_str
                 << " must be explicitly laid out with ArrayStride "
                    "decorations.";
        } else if (!checkForRequiredDecoration(id, SpvDecorationMatrixStride,
                                               SpvOpTypeMatrix, vstate)) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "Structure id " << id << " decorated as " << dec_str
                 << " must be explicitly laid out with MatrixStride "
                    "decorations.";
        }
      }
    }
  }
  return SPV_SUCCESS;
}

}  // anonymous namespace

namespace libspirv {

// Validates that decorations have been applied properly.
spv_result_t ValidateDecorations(ValidationState_t& vstate) {
  if (auto error = CheckImportedVariableInitialization(vstate)) return error;
  if (auto error = CheckDecorationsOfEntryPoints(vstate)) return error;
  if (auto error = CheckDecorationsOfBuffers(vstate)) return error;
  if (auto error = CheckLinkageAttrOfFunctions(vstate)) return error;
  if (auto error = CheckDescriptorSetArrayOfArrays(vstate)) return error;
  return SPV_SUCCESS;
}

}  // namespace libspirv
