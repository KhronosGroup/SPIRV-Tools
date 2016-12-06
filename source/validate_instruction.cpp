// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Performs validation on instructions that appear inside of a SPIR-V block.

#include "validate.h"

#include <algorithm>
#include <cassert>

#include <sstream>
#include <string>

#include "diagnostic.h"
#include "enum_set.h"
#include "opcode.h"
#include "operand.h"
#include "spirv_definition.h"
#include "val/function.h"
#include "val/validation_state.h"

using libspirv::AssemblyGrammar;
using libspirv::CapabilitySet;
using libspirv::DiagnosticStream;
using libspirv::ValidationState_t;

namespace {

std::string ToString(const CapabilitySet& capabilities,
                     const AssemblyGrammar& grammar) {
  std::stringstream ss;
  capabilities.ForEach([&grammar, &ss](SpvCapability cap) {
    spv_operand_desc desc;
    if (SPV_SUCCESS ==
        grammar.lookupOperand(SPV_OPERAND_TYPE_CAPABILITY, cap, &desc))
      ss << desc->name << " ";
    else
      ss << cap << " ";
  });
  return ss.str();
}

// Reports a missing-capability error to _'s diagnostic stream and returns
// SPV_ERROR_INVALID_CAPABILITY.
spv_result_t CapabilityError(ValidationState_t& _, int which_operand,
                             SpvOp opcode,
                             const std::string& required_capabilities) {
  return _.diag(SPV_ERROR_INVALID_CAPABILITY)
         << "Operand " << which_operand << " of " << spvOpcodeString(opcode)
         << " requires one of these capabilities: " << required_capabilities;
}

// Returns an operand's required capabilities.
CapabilitySet RequiredCapabilities(const AssemblyGrammar& grammar,
                                   spv_operand_type_t type, uint32_t operand) {
  // Mere mention of PointSize, ClipDistance, or CullDistance in a Builtin
  // decoration does not require the associated capability.  The use of such
  // a variable value should trigger the capability requirement, but that's
  // not implemented yet.  This rule is independent of target environment.
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/365
  if (type == SPV_OPERAND_TYPE_BUILT_IN) {
    switch (operand) {
      case SpvBuiltInPointSize:
      case SpvBuiltInClipDistance:
      case SpvBuiltInCullDistance:
        return CapabilitySet();
      default:
        break;
    }
  }

  spv_operand_desc operand_desc;
  if (SPV_SUCCESS == grammar.lookupOperand(type, operand, &operand_desc)) {
    return operand_desc->capabilities;
  }

  return CapabilitySet();
}

}  // namespace

namespace libspirv {

spv_result_t CapCheck(ValidationState_t& _,
                      const spv_parsed_instruction_t* inst) {
  spv_opcode_desc opcode_desc;
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  if (SPV_SUCCESS == _.grammar().lookupOpcode(opcode, &opcode_desc) &&
      !_.HasAnyOf(opcode_desc->capabilities))
    return _.diag(SPV_ERROR_INVALID_CAPABILITY)
           << "Opcode " << spvOpcodeString(opcode)
           << " requires one of these capabilities: "
           << ToString(opcode_desc->capabilities, _.grammar());
  for (int i = 0; i < inst->num_operands; ++i) {
    const auto& operand = inst->operands[i];
    const auto word = inst->words[operand.offset];
    if (spvOperandIsConcreteMask(operand.type)) {
      // Check for required capabilities for each bit position of the mask.
      for (uint32_t mask_bit = 0x80000000; mask_bit; mask_bit >>= 1) {
        if (word & mask_bit) {
          const auto caps =
              RequiredCapabilities(_.grammar(), operand.type, mask_bit);
          if (!_.HasAnyOf(caps)) {
            return CapabilityError(_, i + 1, opcode,
                                   ToString(caps, _.grammar()));
          }
        }
      }
    } else if (spvIsIdType(operand.type)) {
      // TODO(dneto): Check the value referenced by this Id, if we can compute
      // it.  For now, just punt, to fix issue 248:
      // https://github.com/KhronosGroup/SPIRV-Tools/issues/248
    } else {
      // Check the operand word as a whole.
      const auto caps = RequiredCapabilities(_.grammar(), operand.type, word);
      if (!_.HasAnyOf(caps)) {
        return CapabilityError(_, i + 1, opcode, ToString(caps, _.grammar()));
      }
    }
  }
  return SPV_SUCCESS;
}

// Checks that the Resuld <id> is within the valid bound.
spv_result_t LimitCheckIdBound(ValidationState_t& _,
                               const spv_parsed_instruction_t* inst) {
  if (inst->result_id >= _.getIdBound()) {
    return _.diag(SPV_ERROR_INVALID_BINARY)
           << "Result <id> '" << inst->result_id
           << "' must be less than the ID bound '" << _.getIdBound() << "'.";
  }
  return SPV_SUCCESS;
}

// Checks that the number of OpTypeStruct members is within the limit.
spv_result_t LimitCheckStruct(ValidationState_t& _,
                              const spv_parsed_instruction_t* inst) {
  if (SpvOpTypeStruct != inst->opcode) {
    return SPV_SUCCESS;
  }

  // Number of members is the number of operands of the instruction minus 1.
  // One operand is the result ID.
  const uint16_t limit = 0x3fff;
  if (inst->num_operands - 1 > limit) {
    return _.diag(SPV_ERROR_INVALID_BINARY)
           << "Number of OpTypeStruct members (" << inst->num_operands - 1
           << ") has exceeded the limit (" << limit << ").";
  }

  // Section 2.17 of SPIRV Spec specifies that the "Structure Nesting Depth"
  // must be less than or equal to 255.
  // This is interpreted as structures including other structures as members.
  // The code does not follow pointers or look into arrays to see if we reach a
  // structure downstream.
  // The nesting depth of a struct is 1+(largest depth of any member).
  // Scalars are at depth 0.
  uint32_t max_member_depth = 0;
  // Struct members start at word 2 of OpTypeStruct instruction.
  for (size_t word_i = 2; word_i < inst->num_words; ++word_i) {
    auto member = inst->words[word_i];
    auto memberTypeInstr = _.FindDef(member);
    if (memberTypeInstr && SpvOpTypeStruct == memberTypeInstr->opcode()) {
      max_member_depth = std::max(
          max_member_depth, _.struct_nesting_depth(memberTypeInstr->id()));
    }
  }

  const uint32_t depth_limit = 255;
  const uint32_t cur_depth = 1 + max_member_depth;
  _.set_struct_nesting_depth(inst->result_id, cur_depth);
  if (cur_depth > depth_limit) {
    return _.diag(SPV_ERROR_INVALID_BINARY)
           << "Structure Nesting Depth may not be larger than " << depth_limit
           << ". Found " << cur_depth << ".";
  }
  return SPV_SUCCESS;
}

// Checks that the number of (literal, label) pairs in OpSwitch is within the
// limit.
spv_result_t LimitCheckSwitch(ValidationState_t& _,
                              const spv_parsed_instruction_t* inst) {
  if (SpvOpSwitch == inst->opcode) {
    // The instruction syntax is as follows:
    // OpSwitch <selector ID> <Default ID> literal label literal label ...
    // literal,label pairs come after the first 2 operands.
    // It is guaranteed at this point that num_operands is an even numner.
    unsigned int num_pairs = (inst->num_operands - 2) / 2;
    const unsigned int num_pairs_limit = 16383;
    if (num_pairs > num_pairs_limit) {
      return _.diag(SPV_ERROR_INVALID_BINARY)
             << "Number of (literal, label) pairs in OpSwitch (" << num_pairs
             << ") exceeds the limit (" << num_pairs_limit << ").";
    }
  }
  return SPV_SUCCESS;
}

// Ensure the number of variables of the given class does not exceed the limit.
spv_result_t LimitCheckNumVars(ValidationState_t& _,
                               const SpvStorageClass storage_class) {
  if (SpvStorageClassFunction == storage_class) {
    _.incrementNumLocalVars();
    const uint32_t num_local_vars_limit = 0x7FFFF;
    if (_.num_local_vars() > num_local_vars_limit) {
      return _.diag(SPV_ERROR_INVALID_BINARY)
             << "Number of local variables ('Function' Storage Class) "
                "exceeded the valid limit ("
             << num_local_vars_limit << ").";
    }
  } else {
    _.incrementNumGlobalVars();
    const uint32_t num_global_vars_limit = 0xFFFF;
    if (_.num_global_vars() > num_global_vars_limit) {
      return _.diag(SPV_ERROR_INVALID_BINARY)
             << "Number of Global Variables (Storage Class other than "
                "'Function') exceeded the valid limit ("
             << num_global_vars_limit << ").";
    }
  }
  return SPV_SUCCESS;
}

spv_result_t InstructionPass(ValidationState_t& _,
                             const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  if (opcode == SpvOpCapability)
    _.RegisterCapability(
        static_cast<SpvCapability>(inst->words[inst->operands[0].offset]));
  if (opcode == SpvOpMemoryModel) {
    _.set_addressing_model(
        static_cast<SpvAddressingModel>(inst->words[inst->operands[0].offset]));
    _.set_memory_model(
        static_cast<SpvMemoryModel>(inst->words[inst->operands[1].offset]));
  }
  if (opcode == SpvOpVariable) {
    const auto storage_class =
        static_cast<SpvStorageClass>(inst->words[inst->operands[2].offset]);
    if (auto error = LimitCheckNumVars(_, storage_class)) {
      return error;
    }
    if (storage_class == SpvStorageClassGeneric)
      return _.diag(SPV_ERROR_INVALID_BINARY)
             << "OpVariable storage class cannot be Generic";
    if (_.current_layout_section() == kLayoutFunctionDefinitions) {
      if (storage_class != SpvStorageClassFunction) {
        return _.diag(SPV_ERROR_INVALID_LAYOUT)
               << "Variables must have a function[7] storage class inside"
                  " of a function";
      }
      if (_.current_function().IsFirstBlock(
              _.current_function().current_block()->id()) == false) {
        return _.diag(SPV_ERROR_INVALID_CFG) << "Variables can only be defined "
                                                "in the first block of a "
                                                "function";
      }
    } else {
      if (storage_class == SpvStorageClassFunction) {
        return _.diag(SPV_ERROR_INVALID_LAYOUT)
               << "Variables can not have a function[7] storage class "
                  "outside of a function";
      }
    }
  }

  if (auto error = CapCheck(_, inst)) return error;
  if (auto error = LimitCheckIdBound(_, inst)) return error;
  if (auto error = LimitCheckStruct(_, inst)) return error;
  if (auto error = LimitCheckSwitch(_, inst)) return error;

  // All instruction checks have passed.
  return SPV_SUCCESS;
}
}  // namespace libspirv
