// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#include <assert.h>

#include <iostream>
#include <unordered_map>
#include <vector>

#include "diagnostic.h"
#include "instruction.h"
#include "libspirv/libspirv.h"
#include "opcode.h"
#include "validate.h"

#define spvCheck(condition, action) \
  if (condition) {                  \
    action;                         \
  }

namespace {
class idUsage {
 public:
  idUsage(const spv_opcode_table opcodeTable,
          const spv_operand_table operandTable,
          const spv_ext_inst_table extInstTable, const spv_id_info_t* pIdUses,
          const uint64_t idUsesCount, const spv_id_info_t* pIdDefs,
          const uint64_t idDefsCount, const spv_instruction_t* pInsts,
          const uint64_t instCount, spv_position position,
          spv_diagnostic* pDiagnostic)
      : opcodeTable(opcodeTable),
        operandTable(operandTable),
        extInstTable(extInstTable),
        firstInst(pInsts),
        instCount(instCount),
        position(position),
        pDiagnostic(pDiagnostic) {
    for (uint64_t idUsesIndex = 0; idUsesIndex < idUsesCount; ++idUsesIndex) {
      idUses[pIdUses[idUsesIndex].id].push_back(pIdUses[idUsesIndex]);
    }
    for (uint64_t idDefsIndex = 0; idDefsIndex < idDefsCount; ++idDefsIndex) {
      idDefs[pIdDefs[idDefsIndex].id] = pIdDefs[idDefsIndex];
    }
  }

  bool isValid(const spv_instruction_t* inst);

  template <SpvOp>
  bool isValid(const spv_instruction_t* inst, const spv_opcode_desc);

  std::unordered_map<uint32_t, spv_id_info_t>::iterator find(
      const uint32_t& id) {
    return idDefs.find(id);
  }
  std::unordered_map<uint32_t, spv_id_info_t>::const_iterator find(
      const uint32_t& id) const {
    return idDefs.find(id);
  }

  bool found(std::unordered_map<uint32_t, spv_id_info_t>::iterator item) {
    return idDefs.end() != item;
  }
  bool found(std::unordered_map<uint32_t, spv_id_info_t>::const_iterator item) {
    return idDefs.end() != item;
  }

  std::unordered_map<uint32_t, std::vector<spv_id_info_t>>::iterator findUses(
      const uint32_t& id) {
    return idUses.find(id);
  }
  std::unordered_map<uint32_t, std::vector<spv_id_info_t>>::const_iterator
  findUses(const uint32_t& id) const {
    return idUses.find(id);
  }

  bool foundUses(
      std::unordered_map<uint32_t, std::vector<spv_id_info_t>>::iterator item) {
    return idUses.end() != item;
  }
  bool foundUses(std::unordered_map<
                 uint32_t, std::vector<spv_id_info_t>>::const_iterator item) {
    return idUses.end() != item;
  }

 private:
  const spv_opcode_table opcodeTable;
  const spv_operand_table operandTable;
  const spv_ext_inst_table extInstTable;
  const spv_instruction_t* const firstInst;
  const uint64_t instCount;
  spv_position position;
  spv_diagnostic* pDiagnostic;
  std::unordered_map<uint32_t, std::vector<spv_id_info_t>> idUses;
  std::unordered_map<uint32_t, spv_id_info_t> idDefs;
};

#define DIAG(INDEX)         \
  position->index += INDEX; \
  DIAGNOSTIC

#if 0
template <>
bool idUsage::isValid<SpvOpUndef>(const spv_instruction_t *inst,
                                  const spv_opcode_desc) {
  assert(0 && "Unimplemented!");
  return false;
}
#endif

template <>
bool idUsage::isValid<SpvOpName>(const spv_instruction_t* inst,
                                 const spv_opcode_desc) {
  auto targetIndex = 1;
  auto target = find(inst->words[targetIndex]);
  spvCheck(!found(target), DIAG(targetIndex) << "OpName Target <id> '"
                                             << inst->words[targetIndex]
                                             << "' is not defined.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpMemberName>(const spv_instruction_t* inst,
                                       const spv_opcode_desc) {
  auto typeIndex = 1;
  auto type = find(inst->words[typeIndex]);
  spvCheck(!found(type), DIAG(typeIndex) << "OpMemberName Type <id> '"
                                         << inst->words[typeIndex]
                                         << "' is not defined.";
           return false);
  spvCheck(SpvOpTypeStruct != type->second.opcode,
           DIAG(typeIndex) << "OpMemberName Type <id> '"
                           << inst->words[typeIndex]
                           << "' is not a struct type.";
           return false);
  auto memberIndex = 2;
  auto member = inst->words[memberIndex];
  auto memberCount = (uint32_t)(type->second.inst->words.size() - 2);
  spvCheck(memberCount <= member, DIAG(memberIndex)
                                      << "OpMemberName Member <id> '"
                                      << inst->words[memberIndex]
                                      << "' index is larger than Type <id> '"
                                      << type->second.id << "'s member count.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpLine>(const spv_instruction_t* inst,
                                 const spv_opcode_desc) {
  auto fileIndex = 1;
  auto file = find(inst->words[fileIndex]);
  spvCheck(!found(file), DIAG(fileIndex) << "OpLine Target <id> '"
                                         << inst->words[fileIndex]
                                         << "' is not defined.";
           return false);
  spvCheck(SpvOpString != file->second.opcode,
           DIAG(fileIndex) << "OpLine Target <id> '" << inst->words[fileIndex]
                           << "' is not an OpString.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpDecorate>(const spv_instruction_t* inst,
                                     const spv_opcode_desc) {
  auto targetIndex = 1;
  auto target = find(inst->words[targetIndex]);
  spvCheck(!found(target), DIAG(targetIndex) << "OpDecorate Target <id> '"
                                             << inst->words[targetIndex]
                                             << "' is not defined.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpMemberDecorate>(const spv_instruction_t* inst,
                                           const spv_opcode_desc) {
  auto structTypeIndex = 1;
  auto structType = find(inst->words[structTypeIndex]);
  spvCheck(!found(structType), DIAG(structTypeIndex)
                                   << "OpMemberDecorate Structure type <id> '"
                                   << inst->words[structTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(SpvOpTypeStruct != structType->second.inst->opcode,
           DIAG(structTypeIndex) << "OpMemberDecorate Structure type <id> '"
                                 << inst->words[structTypeIndex]
                                 << "' is not a struct type.";
           return false);
  auto memberIndex = 2;
  auto member = inst->words[memberIndex];
  auto memberCount = (uint32_t)(structType->second.inst->words.size() - 2);
  spvCheck(memberCount < member, DIAG(memberIndex)
                                     << "OpMemberDecorate Structure type <id> '"
                                     << inst->words[memberIndex]
                                     << "' member count is less than Member";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpGroupDecorate>(const spv_instruction_t* inst,
                                          const spv_opcode_desc) {
  auto decorationGroupIndex = 1;
  auto decorationGroup = find(inst->words[decorationGroupIndex]);
  spvCheck(!found(decorationGroup),
           DIAG(decorationGroupIndex)
               << "OpGroupDecorate Decoration group <id> '"
               << inst->words[decorationGroupIndex] << "' is not defined.";
           return false);
  spvCheck(SpvOpDecorationGroup != decorationGroup->second.opcode,
           DIAG(decorationGroupIndex)
               << "OpGroupDecorate Decoration group <id> '"
               << inst->words[decorationGroupIndex]
               << "' is not a decoration group.";
           return false);
  for (size_t targetIndex = 2; targetIndex < inst->words.size();
       ++targetIndex) {
    auto target = find(inst->words[targetIndex]);
    spvCheck(!found(target), DIAG(targetIndex)
                                 << "OpGroupDecorate Target <id> '"
                                 << inst->words[targetIndex]
                                 << "' is not defined.";
             return false);
  }
  return true;
}

#if 0
template <>
bool idUsage::isValid<SpvOpGroupMemberDecorate>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<SpvOpExtInst>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

template <>
bool idUsage::isValid<SpvOpEntryPoint>(const spv_instruction_t* inst,
                                       const spv_opcode_desc) {
  auto entryPointIndex = 2;
  auto entryPoint = find(inst->words[entryPointIndex]);
  spvCheck(!found(entryPoint), DIAG(entryPointIndex)
                                   << "OpEntryPoint Entry Point <id> '"
                                   << inst->words[entryPointIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(SpvOpFunction != entryPoint->second.opcode,
           DIAG(entryPointIndex) << "OpEntryPoint Entry Point <id> '"
                                 << inst->words[entryPointIndex]
                                 << "' is not a function.";
           return false);
  // TODO: Check the entry point signature is void main(void), may be subject
  // to change
  auto entryPointType = find(entryPoint->second.inst->words[4]);
  spvCheck(!found(entryPointType), assert(0 && "Unreachable!"));
  spvCheck(3 != entryPointType->second.inst->words.size(),
           DIAG(entryPointIndex) << "OpEntryPoint Entry Point <id> '"
                                 << inst->words[entryPointIndex]
                                 << "'s function parameter count is not zero.";
           return false);
  auto returnType = find(entryPoint->second.inst->words[1]);
  spvCheck(!found(returnType), assert(0 && "Unreachable!"));
  spvCheck(SpvOpTypeVoid != returnType->second.opcode,
           DIAG(entryPointIndex) << "OpEntryPoint Entry Point <id> '"
                                 << inst->words[entryPointIndex]
                                 << "'s function return type is not void.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpExecutionMode>(const spv_instruction_t* inst,
                                          const spv_opcode_desc) {
  auto entryPointIndex = 1;
  auto entryPoint = find(inst->words[entryPointIndex]);
  spvCheck(!found(entryPoint), DIAG(entryPointIndex)
                                   << "OpExecutionMode Entry Point <id> '"
                                   << inst->words[entryPointIndex]
                                   << "' is not defined.";
           return false);
  auto entryPointUses = findUses(inst->words[entryPointIndex]);
  spvCheck(!foundUses(entryPointUses), assert(0 && "Unreachable!"));
  bool foundEntryPointUse = false;
  for (auto use : entryPointUses->second) {
    if (SpvOpEntryPoint == use.opcode) {
      foundEntryPointUse = true;
    }
  }
  spvCheck(!foundEntryPointUse, DIAG(entryPointIndex)
                                    << "OpExecutionMode Entry Point <id> '"
                                    << inst->words[entryPointIndex]
                                    << "' is not the Entry Point "
                                       "operand of an OpEntryPoint.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpTypeVector>(const spv_instruction_t* inst,
                                       const spv_opcode_desc) {
  auto componentIndex = 2;
  auto componentType = find(inst->words[componentIndex]);
  spvCheck(!found(componentType), DIAG(componentIndex)
                                      << "OpTypeVector Component Type <id> '"
                                      << inst->words[componentIndex]
                                      << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeIsScalarType(componentType->second.opcode),
           DIAG(componentIndex) << "OpTypeVector Component Type <id> '"
                                << inst->words[componentIndex]
                                << "' is not a scalar type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpTypeMatrix>(const spv_instruction_t* inst,
                                       const spv_opcode_desc) {
  auto columnTypeIndex = 2;
  auto columnType = find(inst->words[columnTypeIndex]);
  spvCheck(!found(columnType), DIAG(columnTypeIndex)
                                   << "OpTypeMatrix Column Type <id> '"
                                   << inst->words[columnTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(SpvOpTypeVector != columnType->second.opcode,
           DIAG(columnTypeIndex) << "OpTypeMatrix Column Type <id> '"
                                 << inst->words[columnTypeIndex]
                                 << "' is not a vector.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpTypeSampler>(const spv_instruction_t*,
                                        const spv_opcode_desc) {
  // OpTypeSampler takes no arguments in Rev31 and beyond.
  return true;
}

template <>
bool idUsage::isValid<SpvOpTypeArray>(const spv_instruction_t* inst,
                                      const spv_opcode_desc) {
  auto elementTypeIndex = 2;
  auto elementType = find(inst->words[elementTypeIndex]);
  spvCheck(!found(elementType), DIAG(elementTypeIndex)
                                    << "OpTypeArray Element Type <id> '"
                                    << inst->words[elementTypeIndex]
                                    << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeGeneratesType(elementType->second.opcode),
           DIAG(elementTypeIndex) << "OpTypeArray Element Type <id> '"
                                  << inst->words[elementTypeIndex]
                                  << "' is not a type.";
           return false);
  auto lengthIndex = 3;
  auto length = find(inst->words[lengthIndex]);
  spvCheck(!found(length), DIAG(lengthIndex) << "OpTypeArray Length <id> '"
                                             << inst->words[lengthIndex]
                                             << "' is not defined.";
           return false);
  spvCheck(SpvOpConstant != length->second.opcode &&
               SpvOpSpecConstant != length->second.opcode,
           DIAG(lengthIndex) << "OpTypeArray Length <id> '"
                             << inst->words[lengthIndex]
                             << "' is not a scalar constant type.";
           return false);

  // NOTE: Check the initialiser value of the constant
  auto constInst = length->second.inst;
  auto constResultTypeIndex = 1;
  auto constResultType = find(constInst->words[constResultTypeIndex]);
  spvCheck(!found(constResultType), DIAG(lengthIndex)
                                        << "OpTypeArray Length <id> '"
                                        << inst->words[constResultTypeIndex]
                                        << "' result type is not defined.";
           return false);
  spvCheck(SpvOpTypeInt != constResultType->second.opcode,
           DIAG(lengthIndex) << "OpTypeArray Length <id> '"
                             << inst->words[lengthIndex]
                             << "' is not a constant integer type.";
           return false);
  if (4 == constInst->words.size()) {
    spvCheck(1 > constInst->words[3], DIAG(lengthIndex)
                                          << "OpTypeArray Length <id> '"
                                          << inst->words[lengthIndex]
                                          << "' value must be at least 1.";
             return false);
  } else if (5 == constInst->words.size()) {
    uint64_t value =
        constInst->words[3] | ((uint64_t)constInst->words[4]) << 32;
    bool signedness = constResultType->second.inst->words[3];
    if (signedness) {
      spvCheck(1 > (int64_t)value, DIAG(lengthIndex)
                                       << "OpTypeArray Length <id> '"
                                       << inst->words[lengthIndex]
                                       << "' value must be at least 1.";
               return false);
    } else {
      spvCheck(1 > value, DIAG(lengthIndex) << "OpTypeArray Length <id> '"
                                            << inst->words[lengthIndex]
                                            << "' value must be at least 1.";
               return false);
    }
  }
  return true;
}

template <>
bool idUsage::isValid<SpvOpTypeRuntimeArray>(const spv_instruction_t* inst,
                                             const spv_opcode_desc) {
  auto elementTypeIndex = 2;
  auto elementType = find(inst->words[elementTypeIndex]);
  spvCheck(!found(elementType), DIAG(elementTypeIndex)
                                    << "OpTypeRuntimeArray Element Type <id> '"
                                    << inst->words[elementTypeIndex]
                                    << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeGeneratesType(elementType->second.opcode),
           DIAG(elementTypeIndex) << "OpTypeRuntimeArray Element Type <id> '"
                                  << inst->words[elementTypeIndex]
                                  << "' is not a type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpTypeStruct>(const spv_instruction_t* inst,
                                       const spv_opcode_desc) {
  for (size_t memberTypeIndex = 2; memberTypeIndex < inst->words.size();
       ++memberTypeIndex) {
    auto memberType = find(inst->words[memberTypeIndex]);
    spvCheck(!found(memberType), DIAG(memberTypeIndex)
                                     << "OpTypeStruct Member Type <id> '"
                                     << inst->words[memberTypeIndex]
                                     << "' is not defined.";
             return false);
    spvCheck(!spvOpcodeGeneratesType(memberType->second.opcode),
             DIAG(memberTypeIndex) << "OpTypeStruct Member Type <id> '"
                                   << inst->words[memberTypeIndex]
                                   << "' is not a type.";
             return false);
  }
  return true;
}

template <>
bool idUsage::isValid<SpvOpTypePointer>(const spv_instruction_t* inst,
                                        const spv_opcode_desc) {
  auto typeIndex = 3;
  auto type = find(inst->words[typeIndex]);
  spvCheck(!found(type), DIAG(typeIndex) << "OpTypePointer Type <id> '"
                                         << inst->words[typeIndex]
                                         << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeGeneratesType(type->second.opcode),
           DIAG(typeIndex) << "OpTypePointer Type <id> '"
                           << inst->words[typeIndex] << "' is not a type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpTypeFunction>(const spv_instruction_t* inst,
                                         const spv_opcode_desc) {
  auto returnTypeIndex = 2;
  auto returnType = find(inst->words[returnTypeIndex]);
  spvCheck(!found(returnType), DIAG(returnTypeIndex)
                                   << "OpTypeFunction Return Type <id> '"
                                   << inst->words[returnTypeIndex]
                                   << "' is not defined";
           return false);
  spvCheck(!spvOpcodeGeneratesType(returnType->second.opcode),
           DIAG(returnTypeIndex) << "OpTypeFunction Return Type <id> '"
                                 << inst->words[returnTypeIndex]
                                 << "' is not a type.";
           return false);
  for (size_t paramTypeIndex = 3; paramTypeIndex < inst->words.size();
       ++paramTypeIndex) {
    auto paramType = find(inst->words[paramTypeIndex]);
    spvCheck(!found(paramType), DIAG(paramTypeIndex)
                                    << "OpTypeFunction Parameter Type <id> '"
                                    << inst->words[paramTypeIndex]
                                    << "' is not defined.";
             return false);
    spvCheck(!spvOpcodeGeneratesType(paramType->second.opcode),
             DIAG(paramTypeIndex) << "OpTypeFunction Parameter Type <id> '"
                                  << inst->words[paramTypeIndex]
                                  << "' is not a type.";
             return false);
  }
  return true;
}

template <>
bool idUsage::isValid<SpvOpTypePipe>(const spv_instruction_t*,
                                     const spv_opcode_desc) {
  // OpTypePipe has no ID arguments.
  return true;
}

template <>
bool idUsage::isValid<SpvOpConstantTrue>(const spv_instruction_t* inst,
                                         const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpConstantTrue Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(SpvOpTypeBool != resultType->second.opcode,
           DIAG(resultTypeIndex) << "OpConstantTrue Result Type <id> '"
                                 << inst->words[resultTypeIndex]
                                 << "' is not a boolean type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpConstantFalse>(const spv_instruction_t* inst,
                                          const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpConstantFalse Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(SpvOpTypeBool != resultType->second.opcode,
           DIAG(resultTypeIndex) << "OpConstantFalse Result Type <id> '"
                                 << inst->words[resultTypeIndex]
                                 << "' is not a boolean type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpConstant>(const spv_instruction_t* inst,
                                     const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpConstant Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeIsScalarType(resultType->second.opcode),
           DIAG(resultTypeIndex)
               << "OpConstant Result Type <id> '"
               << inst->words[resultTypeIndex]
               << "' is not a scalar integer or floating point type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpConstantComposite>(const spv_instruction_t* inst,
                                              const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpConstantComposite Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeIsComposite(resultType->second.opcode),
           DIAG(resultTypeIndex) << "OpConstantComposite Result Type <id> '"
                                 << inst->words[resultTypeIndex]
                                 << "' is not a composite type.";
           return false);

  uint32_t constituentCount = inst->words.size() - 3;
  switch (resultType->second.opcode) {
    case SpvOpTypeVector: {
      auto componentCount = resultType->second.inst->words[3];
      spvCheck(
          componentCount != constituentCount,
          // TODO: Output ID's on diagnostic
          DIAG(inst->words.size() - 1)
              << "OpConstantComposite Constituent <id> count does not match "
                 "Result Type <id> '"
              << resultType->second.id << "'s vector component count.";
          return false);
      auto componentType = find(resultType->second.inst->words[2]);
      spvCheck(!found(componentType), assert(0 && "Unreachable!"));
      for (size_t constituentIndex = 3; constituentIndex < inst->words.size();
           constituentIndex++) {
        auto constituent = find(inst->words[constituentIndex]);
        spvCheck(!found(constituent), assert(0 && "Unreachable!"));
        spvCheck(!spvOpcodeIsConstant(constituent->second.opcode),
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex] << "' is not a constant.";
                 return false);
        auto constituentResultType = find(constituent->second.inst->words[1]);
        spvCheck(!found(constituentResultType), assert(0 && "Unreachable!"));
        spvCheck(componentType->second.opcode !=
                     constituentResultType->second.opcode,
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex]
                     << "'s type does not match Result Type <id> '"
                     << resultType->second.id << "'s vector element type.";
                 return false);
      }
    } break;
    case SpvOpTypeMatrix: {
      auto columnCount = resultType->second.inst->words[3];
      spvCheck(
          columnCount != constituentCount,
          // TODO: Output ID's on diagnostic
          DIAG(inst->words.size() - 1)
              << "OpConstantComposite Constituent <id> count does not match "
                 "Result Type <id> '"
              << resultType->second.id << "'s matrix column count.";
          return false);

      auto columnType = find(resultType->second.inst->words[2]);
      spvCheck(!found(columnType), assert(0 && "Unreachable!"));
      auto componentCount = columnType->second.inst->words[3];
      auto componentType = find(columnType->second.inst->words[2]);
      spvCheck(!found(componentType), assert(0 && "Unreachable!"));

      for (size_t constituentIndex = 3; constituentIndex < inst->words.size();
           constituentIndex++) {
        auto constituent = find(inst->words[constituentIndex]);
        spvCheck(!found(constituent),
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex] << "' is not defined.";
                 return false);
        spvCheck(SpvOpConstantComposite != constituent->second.opcode,
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex]
                     << "' is not a constant composite.";
                 return false);
        auto vector = find(constituent->second.inst->words[1]);
        spvCheck(!found(vector), assert(0 && "Unreachable!"));
        spvCheck(columnType->second.opcode != vector->second.opcode,
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex]
                     << "' type does not match Result Type <id> '"
                     << resultType->second.id << "'s matrix column type.";
                 return false);
        auto vectorComponentType = find(vector->second.inst->words[2]);
        spvCheck(!found(vectorComponentType), assert(0 && "Unreachable!"));
        spvCheck(!spvOpcodeAreTypesEqual(componentType->second.inst,
                                         vectorComponentType->second.inst),
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex]
                     << "' component type does not match Result Type <id> '"
                     << resultType->second.id
                     << "'s matrix column component type.";
                 return false);
        spvCheck(
            componentCount != vector->second.inst->words[3],
            DIAG(constituentIndex)
                << "OpConstantComposite Constituent <id> '"
                << inst->words[constituentIndex]
                << "' vector component count does not match Result Type <id> '"
                << resultType->second.id << "'s vector component count.";
            return false);
      }
    } break;
    case SpvOpTypeArray: {
      auto elementType = find(resultType->second.inst->words[2]);
      spvCheck(!found(elementType), assert(0 && "Unreachable!"));
      auto length = find(resultType->second.inst->words[3]);
      spvCheck(!found(length), assert(0 && "Unreachable!"));
      spvCheck(length->second.inst->words[3] != constituentCount,
               DIAG(inst->words.size() - 1)
                   << "OpConstantComposite Constituent count does not match "
                      "Result Type <id> '"
                   << resultType->second.id << "'s array length.";
               return false);
      for (size_t constituentIndex = 3; constituentIndex < inst->words.size();
           constituentIndex++) {
        auto constituent = find(inst->words[constituentIndex]);
        spvCheck(!found(constituent),
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex] << "' is not defined.";
                 return false);
        spvCheck(!spvOpcodeIsConstant(constituent->second.opcode),
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex] << "' is not a constant.";
                 return false);
        auto constituentType = find(constituent->second.inst->words[1]);
        spvCheck(!found(constituentType), assert(0 && "Unreachable!"));
        spvCheck(!spvOpcodeAreTypesEqual(elementType->second.inst,
                                         constituentType->second.inst),
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex]
                     << "'s type does not match Result Type <id> '"
                     << resultType->second.id << "'s array element type.";
                 return false);
      }
    } break;
    case SpvOpTypeStruct: {
      uint32_t memberCount = resultType->second.inst->words.size() - 2;
      spvCheck(memberCount != constituentCount,
               DIAG(resultTypeIndex)
                   << "OpConstantComposite Constituent <id> '"
                   << inst->words[resultTypeIndex]
                   << "' count does not match Result Type <id> '"
                   << resultType->second.id << "'s struct member count.";
               return false);
      for (uint32_t constituentIndex = 3, memberIndex = 2;
           constituentIndex < inst->words.size();
           constituentIndex++, memberIndex++) {
        auto constituent = find(inst->words[constituentIndex]);
        spvCheck(!found(constituent),
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex] << "' is not define.";
                 return false);
        spvCheck(!spvOpcodeIsConstant(constituent->second.opcode),
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex] << "' is not a constant.";
                 return false);
        auto constituentType = find(constituent->second.inst->words[1]);
        spvCheck(!found(constituentType), assert(0 && "Unreachable!"));

        auto memberType = find(resultType->second.inst->words[memberIndex]);
        spvCheck(!found(memberType), assert(0 && "Unreachable!"));
        spvCheck(!spvOpcodeAreTypesEqual(memberType->second.inst,
                                         constituentType->second.inst),
                 DIAG(constituentIndex)
                     << "OpConstantComposite Constituent <id> '"
                     << inst->words[constituentIndex]
                     << "' type does not match the Result Type <id> '"
                     << resultType->second.id << "'s member type.";
                 return false);
      }
    } break;
    default: { assert(0 && "Unreachable!"); } break;
  }
  return true;
}

template <>
bool idUsage::isValid<SpvOpConstantSampler>(const spv_instruction_t* inst,
                                            const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpConstantSampler Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(SpvOpTypeSampler != resultType->second.opcode,
           DIAG(resultTypeIndex) << "OpConstantSampler Result Type <id> '"
                                 << inst->words[resultTypeIndex]
                                 << "' is not a sampler type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpConstantNull>(const spv_instruction_t* inst,
                                         const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpConstantNull Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  switch (resultType->second.inst->opcode) {
    default: {
      spvCheck(!spvOpcodeIsBasicTypeNullable(resultType->second.inst->opcode),
               DIAG(resultTypeIndex) << "OpConstantNull Result Type <id> '"
                                     << inst->words[resultTypeIndex]
                                     << "' can not be null.";
               return false);
    } break;
    case SpvOpTypeVector: {
      auto type = find(resultType->second.inst->words[2]);
      spvCheck(!found(type), assert(0 && "Unreachable!"));
      spvCheck(!spvOpcodeIsBasicTypeNullable(type->second.inst->opcode),
               DIAG(resultTypeIndex)
                   << "OpConstantNull Result Type <id> '"
                   << inst->words[resultTypeIndex]
                   << "'s vector component type can not be null.";
               return false);
    } break;
    case SpvOpTypeArray: {
      auto type = find(resultType->second.inst->words[2]);
      spvCheck(!found(type), assert(0 && "Unreachable!"));
      spvCheck(!spvOpcodeIsBasicTypeNullable(type->second.inst->opcode),
               DIAG(resultTypeIndex)
                   << "OpConstantNull Result Type <id> '"
                   << inst->words[resultTypeIndex]
                   << "'s array element type can not be null.";
               return false);
    } break;
    case SpvOpTypeMatrix: {
      auto columnType = find(resultType->second.inst->words[2]);
      spvCheck(!found(columnType), assert(0 && "Unreachable!"));
      auto type = find(columnType->second.inst->words[2]);
      spvCheck(!found(type), assert(0 && "Unreachable!"));
      spvCheck(!spvOpcodeIsBasicTypeNullable(type->second.inst->opcode),
               DIAG(resultTypeIndex)
                   << "OpConstantNull Result Type <id> '"
                   << inst->words[resultTypeIndex]
                   << "'s matrix component type cna not be null.";
               return false);
    } break;
    case SpvOpTypeStruct: {
      for (size_t elementIndex = 2;
           elementIndex < resultType->second.inst->words.size();
           ++elementIndex) {
        auto element = find(resultType->second.inst->words[elementIndex]);
        spvCheck(!found(element), assert(0 && "Unreachable!"));
        spvCheck(!spvOpcodeIsBasicTypeNullable(element->second.inst->opcode),
                 DIAG(resultTypeIndex)
                     << "OpConstantNull Result Type <id> '"
                     << inst->words[resultTypeIndex]
                     << "'s struct element type can not be null.";
                 return false);
      }
    } break;
  }
  return true;
}

template <>
bool idUsage::isValid<SpvOpSpecConstantTrue>(const spv_instruction_t* inst,
                                             const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpSpecConstantTrue Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(SpvOpTypeBool != resultType->second.opcode,
           DIAG(resultTypeIndex) << "OpSpecConstantTrue Result Type <id> '"
                                 << inst->words[resultTypeIndex]
                                 << "' is not a boolean type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpSpecConstantFalse>(const spv_instruction_t* inst,
                                              const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpSpecConstantFalse Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(SpvOpTypeBool != resultType->second.opcode,
           DIAG(resultTypeIndex) << "OpSpecConstantFalse Result Type <id> '"
                                 << inst->words[resultTypeIndex]
                                 << "' is not a boolean type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpSpecConstant>(const spv_instruction_t* inst,
                                         const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpSpecConstant Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeIsScalarType(resultType->second.opcode),
           DIAG(resultTypeIndex) << "OpSpecConstant Result Type <id> '"
                                 << inst->words[resultTypeIndex]
                                 << "' is not a scalar type.";
           return false);
  return true;
}

#if 0
template <>
bool idUsage::isValid<SpvOpSpecConstantComposite>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<SpvOpSpecConstantOp>(const spv_instruction_t *inst) {}
#endif

template <>
bool idUsage::isValid<SpvOpVariable>(const spv_instruction_t* inst,
                                     const spv_opcode_desc opcodeEntry) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpVariable Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  spvCheck(SpvOpTypePointer != resultType->second.opcode,
           DIAG(resultTypeIndex) << "OpVariable Result Type <id> '"
                                 << inst->words[resultTypeIndex]
                                 << "' is not a pointer type.";
           return false);
  if (opcodeEntry->numTypes < inst->words.size()) {
    auto initialiserIndex = 4;
    auto initialiser = find(inst->words[initialiserIndex]);
    spvCheck(!found(initialiser), DIAG(initialiserIndex)
                                      << "OpVariable Initializer <id> '"
                                      << inst->words[initialiserIndex]
                                      << "' is not defined.";
             return false);
    spvCheck(!spvOpcodeIsConstant(initialiser->second.opcode),
             DIAG(initialiserIndex) << "OpVariable Initializer <id> '"
                                    << inst->words[initialiserIndex]
                                    << "' is not a constant.";
             return false);
  }
  return true;
}

template <>
bool idUsage::isValid<SpvOpLoad>(const spv_instruction_t* inst,
                                 const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpLoad Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defind.";
           return false);
  auto pointerIndex = 3;
  auto pointer = find(inst->words[pointerIndex]);
  spvCheck(!found(pointer), DIAG(pointerIndex) << "OpLoad Pointer <id> '"
                                               << inst->words[pointerIndex]
                                               << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeIsPointer(pointer->second.opcode),
           DIAG(pointerIndex) << "OpLoad Pointer <id> '"
                              << inst->words[pointerIndex]
                              << "' is not a pointer.";
           return false);
  auto type = find(pointer->second.inst->words[1]);
  spvCheck(!found(type), assert(0 && "Unreachable!"));
  spvCheck(resultType != type, DIAG(resultTypeIndex)
                                   << "OpLoad Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << " does not match Pointer <id> '"
                                   << pointer->second.id << "'s type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpStore>(const spv_instruction_t* inst,
                                  const spv_opcode_desc) {
  auto pointerIndex = 1;
  auto pointer = find(inst->words[pointerIndex]);
  spvCheck(!found(pointer), DIAG(pointerIndex) << "OpStore Pointer <id> '"
                                               << inst->words[pointerIndex]
                                               << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeIsPointer(pointer->second.opcode),
           DIAG(pointerIndex) << "OpStore Pointer <id> '"
                              << inst->words[pointerIndex]
                              << "' is not a pointer.";
           return false);
  auto pointerType = find(pointer->second.inst->words[1]);
  spvCheck(!found(pointerType), assert(0 && "Unreachable!"));
  auto type = find(pointerType->second.inst->words[3]);
  spvCheck(!found(type), assert(0 && "Unreachable!"));
  spvCheck(SpvOpTypeVoid == type->second.opcode,
           DIAG(pointerIndex) << "OpStore Pointer <id> '"
                              << inst->words[pointerIndex]
                              << "'s type is void.";
           return false);

  auto objectIndex = 2;
  auto object = find(inst->words[objectIndex]);
  spvCheck(!found(object), DIAG(objectIndex) << "OpStore Object <id> '"
                                             << inst->words[objectIndex]
                                             << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeIsObject(object->second.opcode),
           DIAG(objectIndex) << "OpStore Object <id> '"
                             << inst->words[objectIndex]
                             << "' in not an object.";
           return false);
  auto objectType = find(object->second.inst->words[1]);
  spvCheck(!found(objectType), assert(0 && "Unreachable!"));
  spvCheck(SpvOpTypeVoid == objectType->second.opcode,
           DIAG(objectIndex) << "OpStore Object <id> '"
                             << inst->words[objectIndex] << "'s type is void.";
           return false);

  spvCheck(!spvOpcodeAreTypesEqual(type->second.inst, objectType->second.inst),
           DIAG(pointerIndex) << "OpStore Pointer <id> '"
                              << inst->words[pointerIndex]
                              << "'s type does not match Object <id> '"
                              << objectType->second.id << "'s type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpCopyMemory>(const spv_instruction_t* inst,
                                       const spv_opcode_desc) {
  auto targetIndex = 1;
  auto target = find(inst->words[targetIndex]);
  spvCheck(!found(target), DIAG(targetIndex) << "OpCopyMemory Target <id> '"
                                             << inst->words[targetIndex]
                                             << "' is not defined.";
           return false);
  auto sourceIndex = 2;
  auto source = find(inst->words[sourceIndex]);
  spvCheck(!found(source), DIAG(targetIndex) << "OpCopyMemory Source <id> '"
                                             << inst->words[targetIndex]
                                             << "' is not defined.";
           return false);
  auto targetPointerType = find(target->second.inst->words[1]);
  spvCheck(!found(targetPointerType), assert(0 && "Unreachable!"));
  auto targetType = find(targetPointerType->second.inst->words[3]);
  spvCheck(!found(targetType), assert(0 && "Unreachable!"));
  auto sourcePointerType = find(source->second.inst->words[1]);
  spvCheck(!found(sourcePointerType), assert(0 && "Unreachable!"));
  auto sourceType = find(sourcePointerType->second.inst->words[3]);
  spvCheck(!found(sourceType), assert(0 && "Unreachable!"));
  spvCheck(
      !spvOpcodeAreTypesEqual(targetType->second.inst, sourceType->second.inst),
      DIAG(sourceIndex) << "OpCopyMemory Target <id> '"
                        << inst->words[sourceIndex]
                        << "'s type does not match Source <id> '"
                        << sourceType->second.id << "'s type.";
      return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpCopyMemorySized>(const spv_instruction_t* inst,
                                            const spv_opcode_desc) {
  auto targetIndex = 1;
  auto target = find(inst->words[targetIndex]);
  spvCheck(!found(target),
           DIAG(targetIndex) << "OpCopyMemorySized Target <id> '"
                             << inst->words[targetIndex] << "' is not defined.";
           return false);
  auto sourceIndex = 2;
  auto source = find(inst->words[sourceIndex]);
  spvCheck(!found(source),
           DIAG(sourceIndex) << "OpCopyMemorySized Source <id> '"
                             << inst->words[sourceIndex] << "' is not defined.";
           return false);
  auto sizeIndex = 3;
  auto size = find(inst->words[sizeIndex]);
  spvCheck(!found(size), DIAG(sizeIndex) << "OpCopyMemorySized, Size <id> '"
                                         << inst->words[sizeIndex]
                                         << "' is not defined.";
           return false);
  auto targetPointerType = find(target->second.inst->words[1]);
  spvCheck(!found(targetPointerType), assert(0 && "Unreachable!"));
  spvCheck(SpvOpTypePointer != targetPointerType->second.opcode,
           DIAG(targetIndex) << "OpCopyMemorySized Target <id> '"
                             << inst->words[targetIndex]
                             << "' is not a pointer.";
           return false);
  auto sourcePointerType = find(source->second.inst->words[1]);
  spvCheck(!found(sourcePointerType), assert(0 && "Unreachable!"));
  spvCheck(SpvOpTypePointer != sourcePointerType->second.opcode,
           DIAG(sourceIndex) << "OpCopyMemorySized Source <id> '"
                             << inst->words[sourceIndex]
                             << "' is not a pointer.";
           return false);
  switch (size->second.opcode) {
    // TODO: The following opcode's are assumed to be valid, refer to the
    // following bug https://cvs.khronos.org/bugzilla/show_bug.cgi?id=13871 for
    // clarification
    case SpvOpConstant:
    case SpvOpSpecConstant: {
      auto sizeType = find(size->second.inst->words[1]);
      spvCheck(!found(sizeType), assert(0 && "Unreachable!"));
      spvCheck(SpvOpTypeInt != sizeType->second.opcode,
               DIAG(sizeIndex) << "OpCopyMemorySized Size <id> '"
                               << inst->words[sizeIndex]
                               << "'s type is not an integer type.";
               return false);
    } break;
    case SpvOpVariable: {
      auto pointerType = find(size->second.inst->words[1]);
      spvCheck(!found(pointerType), assert(0 && "Unreachable!"));
      auto sizeType = find(pointerType->second.inst->words[1]);
      spvCheck(!found(sizeType), assert(0 && "Unreachable!"));
      spvCheck(SpvOpTypeInt != sizeType->second.opcode,
               DIAG(sizeIndex) << "OpCopyMemorySized Size <id> '"
                               << inst->words[sizeIndex]
                               << "'s variable type is not an integer type.";
               return false);
    } break;
    default:
      DIAG(sizeIndex) << "OpCopyMemorySized Size <id> '"
                      << inst->words[sizeIndex]
                      << "' is not a constant or variable.";
      return false;
  }
  // TODO: Check that consant is a least size 1, see the same bug as above for
  // clarification?
  return true;
}

#if 0
template <>
bool idUsage::isValid<SpvOpAccessChain>(const spv_instruction_t *inst,
                                        const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<SpvOpInBoundsAccessChain>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<SpvOpArrayLength>(const spv_instruction_t *inst,
                                        const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<SpvOpImagePointer>(const spv_instruction_t *inst,
                                         const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<SpvOpGenericPtrMemSemantics>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

template <>
bool idUsage::isValid<SpvOpFunction>(const spv_instruction_t* inst,
                                     const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpFunction Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  auto functionTypeIndex = 4;
  auto functionType = find(inst->words[functionTypeIndex]);
  spvCheck(!found(functionType), DIAG(functionTypeIndex)
                                     << "OpFunction Function Type <id> '"
                                     << inst->words[functionTypeIndex]
                                     << "' is not defined.";
           return false);
  spvCheck(SpvOpTypeFunction != functionType->second.opcode,
           DIAG(functionTypeIndex) << "OpFunction Function Type <id> '"
                                   << inst->words[functionTypeIndex]
                                   << "' is not a function type.";
           return false);
  auto returnType = find(functionType->second.inst->words[2]);
  spvCheck(!found(returnType), assert(0 && "Unreachable!"));
  spvCheck(returnType != resultType,
           DIAG(resultTypeIndex) << "OpFunction Result Type <id> '"
                                 << inst->words[resultTypeIndex]
                                 << "' does not match the Function Type <id> '"
                                 << resultType->second.id << "'s return type.";
           return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpFunctionParameter>(const spv_instruction_t* inst,
                                              const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpFunctionParameter Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  auto function = inst - 1;
  // NOTE: Find OpFunction & ensure OpFunctionParameter is not out of place.
  size_t paramIndex = 0;
  while (firstInst != function) {
    spvCheck(SpvOpFunction != function->opcode &&
                 SpvOpFunctionParameter != function->opcode,
             DIAG(0) << "OpFunctionParameter is not preceded by OpFunction or "
                        "OpFunctionParameter sequence.";
             return false);
    if (SpvOpFunction == function->opcode) {
      break;
    } else {
      paramIndex++;
    }
  }
  auto functionType = find(function->words[4]);
  spvCheck(!found(functionType), assert(0 && "Unreachable!"));
  auto paramType = find(functionType->second.inst->words[paramIndex + 3]);
  spvCheck(!found(paramType), assert(0 && "Unreachable!"));
  spvCheck(
      !spvOpcodeAreTypesEqual(resultType->second.inst, paramType->second.inst),
      DIAG(resultTypeIndex) << "OpFunctionParameter Result Type <id> '"
                            << inst->words[resultTypeIndex]
                            << "' does not match the OpTypeFunction parameter "
                               "type of the same index.";
      return false);
  return true;
}

template <>
bool idUsage::isValid<SpvOpFunctionCall>(const spv_instruction_t* inst,
                                         const spv_opcode_desc) {
  auto resultTypeIndex = 1;
  auto resultType = find(inst->words[resultTypeIndex]);
  spvCheck(!found(resultType), DIAG(resultTypeIndex)
                                   << "OpFunctionCall Result Type <id> '"
                                   << inst->words[resultTypeIndex]
                                   << "' is not defined.";
           return false);
  auto functionIndex = 3;
  auto function = find(inst->words[functionIndex]);
  spvCheck(!found(function), DIAG(functionIndex)
                                 << "OpFunctionCall Function <id> '"
                                 << inst->words[functionIndex]
                                 << "' is not defined.";
           return false);
  spvCheck(SpvOpFunction != function->second.opcode,
           DIAG(functionIndex) << "OpFunctionCall Function <id> '"
                               << inst->words[functionIndex]
                               << "' is not a function.";
           return false);
  auto returnType = find(function->second.inst->words[1]);
  spvCheck(!found(returnType), assert(0 && "Unreachable!"));
  spvCheck(
      !spvOpcodeAreTypesEqual(returnType->second.inst, resultType->second.inst),
      DIAG(resultTypeIndex)
          << "OpFunctionCall Result Type <id> '" << inst->words[resultTypeIndex]
          << "'s type does not match Function <id> '" << returnType->second.id
          << "'s return type.";
      return false);
  auto functionType = find(function->second.inst->words[4]);
  spvCheck(!found(functionType), assert(0 && "Unreachable!"));
  auto functionCallArgCount = inst->words.size() - 4;
  auto functionParamCount = functionType->second.inst->words.size() - 3;
  spvCheck(
      functionParamCount != functionCallArgCount,
      DIAG(inst->words.size() - 1)
          << "OpFunctionCall Function <id>'s parameter count does not match "
             "the argument count.";
      return false);
  for (size_t argumentIndex = 4, paramIndex = 3;
       argumentIndex < inst->words.size(); argumentIndex++, paramIndex++) {
    auto argument = find(inst->words[argumentIndex]);
    spvCheck(!found(argument), DIAG(argumentIndex)
                                   << "OpFunctionCall Argument <id> '"
                                   << inst->words[argumentIndex]
                                   << "' is not defined.";
             return false);
    auto argumentType = find(argument->second.inst->words[1]);
    spvCheck(!found(argumentType), assert(0 && "Unreachable!"));
    auto parameterType = find(functionType->second.inst->words[paramIndex]);
    spvCheck(!found(parameterType), assert(0 && "Unreachable!"));
    spvCheck(!spvOpcodeAreTypesEqual(argumentType->second.inst,
                                     parameterType->second.inst),
             DIAG(argumentIndex) << "OpFunctionCall Argument <id> '"
                                 << inst->words[argumentIndex]
                                 << "'s type does not match Function <id> '"
                                 << parameterType->second.id
                                 << "'s parameter type.";
             return false);
  }
  return true;
}

#if 0
template <>
bool idUsage::isValid<OpConvertUToF>(const spv_instruction_t *inst,
                                     const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpConvertFToS>(const spv_instruction_t *inst,
                                     const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpConvertSToF>(const spv_instruction_t *inst,
                                     const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpConvertUToF>(const spv_instruction_t *inst,
                                     const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpUConvert>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSConvert>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFConvert>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpConvertPtrToU>(const spv_instruction_t *inst,
                                       const spv_opcode_desc opcodeEntry) {
}
#endif

#if 0
template <>
bool idUsage::isValid<OpConvertUToPtr>(const spv_instruction_t *inst,
                                       const spv_opcode_desc opcodeEntry) {
}
#endif

#if 0
template <>
bool idUsage::isValid<OpPtrCastToGeneric>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGenericCastToPtr>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpBitcast>(const spv_instruction_t *inst,
                                 const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGenericCastToPtrExplicit>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSatConvertSToU>(const spv_instruction_t *inst) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSatConvertUToS>(const spv_instruction_t *inst) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpVectorExtractDynamic>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpVectorInsertDynamic>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpVectorShuffle>(const spv_instruction_t *inst,
                                       const spv_opcode_desc opcodeEntry) {
}
#endif

#if 0
template <>
bool idUsage::isValid<OpCompositeConstruct>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpCompositeExtract>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpCompositeInsert>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpCopyObject>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpTranspose>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSNegate>(const spv_instruction_t *inst,
                                 const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFNegate>(const spv_instruction_t *inst,
                                 const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpNot>(const spv_instruction_t *inst,
                             const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpIAdd>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFAdd>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpISub>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFSub>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpIMul>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFMul>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpUDiv>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSDiv>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFDiv>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpUMod>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSRem>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSMod>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFRem>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFMod>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpVectorTimesScalar>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpMatrixTimesScalar>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpVectorTimesMatrix>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpMatrixTimesVector>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpMatrixTimesMatrix>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpOuterProduct>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpDot>(const spv_instruction_t *inst,
                             const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpShiftRightLogical>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpShiftRightArithmetic>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpShiftLeftLogical>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpBitwiseOr>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpBitwiseXor>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpBitwiseAnd>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAny>(const spv_instruction_t *inst,
                             const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAll>(const spv_instruction_t *inst,
                             const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpIsNan>(const spv_instruction_t *inst,
                               const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpIsInf>(const spv_instruction_t *inst,
                               const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpIsFinite>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpIsNormal>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSignBitSet>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpLessOrGreater>(const spv_instruction_t *inst,
                                       const spv_opcode_desc opcodeEntry) {
}
#endif

#if 0
template <>
bool idUsage::isValid<OpOrdered>(const spv_instruction_t *inst,
                                 const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpUnordered>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpLogicalOr>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpLogicalXor>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpLogicalAnd>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSelect>(const spv_instruction_t *inst,
                                const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpIEqual>(const spv_instruction_t *inst,
                                const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFOrdEqual>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFUnordEqual>(const spv_instruction_t *inst,
                                     const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpINotEqual>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFOrdNotEqual>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFUnordNotEqual>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpULessThan>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSLessThan>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFOrdLessThan>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFUnordLessThan>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpUGreaterThan>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSGreaterThan>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFOrdGreaterThan>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFUnordGreaterThan>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpULessThanEqual>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSLessThanEqual>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFOrdLessThanEqual>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFUnordLessThanEqual>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpUGreaterThanEqual>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSGreaterThanEqual>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFOrdGreaterThanEqual>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFUnordGreaterThanEqual>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpDPdx>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpDPdy>(const spv_instruction_t *inst,
                              const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFWidth>(const spv_instruction_t *inst,
                                const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpDPdxFine>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpDPdyFine>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFwidthFine>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpDPdxCoarse>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpDPdyCoarse>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpFwidthCoarse>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpPhi>(const spv_instruction_t *inst,
                             const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpLoopMerge>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSelectionMerge>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpBranch>(const spv_instruction_t *inst,
                                const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpBranchConditional>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSwitch>(const spv_instruction_t *inst,
                                const spv_opcode_desc opcodeEntry) {}
#endif

template <>
bool idUsage::isValid<SpvOpReturnValue>(const spv_instruction_t* inst,
                                        const spv_opcode_desc) {
  auto valueIndex = 1;
  auto value = find(inst->words[valueIndex]);
  spvCheck(!found(value), DIAG(valueIndex) << "OpReturnValue Value <id> '"
                                           << inst->words[valueIndex]
                                           << "' is not defined.";
           return false);
  spvCheck(!spvOpcodeIsValue(value->second.opcode),
           DIAG(valueIndex) << "OpReturnValue Value <id> '"
                            << inst->words[valueIndex]
                            << "' does not represent a value.";
           return false);
  auto valueType = find(value->second.inst->words[1]);
  spvCheck(!found(valueType), assert(0 && "Unreachable!"));
  // NOTE: Find OpFunction
  const spv_instruction_t* function = inst - 1;
  while (firstInst != function) {
    spvCheck(SpvOpFunction == function->opcode, break);
    function--;
  }
  spvCheck(SpvOpFunction != function->opcode,
           DIAG(valueIndex) << "OpReturnValue is not in a basic block.";
           return false);
  auto returnType = find(function->words[1]);
  spvCheck(!found(returnType), assert(0 && "Unreachable!"));
  if (SpvOpTypePointer == valueType->second.opcode) {
    auto pointerValueType = find(valueType->second.inst->words[3]);
    spvCheck(!found(pointerValueType), assert(0 && "Unreachable!"));
    spvCheck(!spvOpcodeAreTypesEqual(returnType->second.inst,
                                     pointerValueType->second.inst),
             DIAG(valueIndex)
                 << "OpReturnValue Value <id> '" << inst->words[valueIndex]
                 << "'s pointer type does not match OpFunction's return type.";
             return false);
  } else {
    spvCheck(!spvOpcodeAreTypesEqual(returnType->second.inst,
                                     valueType->second.inst),
             DIAG(valueIndex)
                 << "OpReturnValue Value <id> '" << inst->words[valueIndex]
                 << "'s type does not match OpFunction's return type.";
             return false);
  }
  return true;
}

#if 0
template <>
bool idUsage::isValid<OpLifetimeStart>(const spv_instruction_t *inst,
                                       const spv_opcode_desc opcodeEntry) {
}
#endif

#if 0
template <>
bool idUsage::isValid<OpLifetimeStop>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicInit>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicLoad>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicStore>(const spv_instruction_t *inst,
                                     const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicExchange>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicCompareExchange>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicCompareExchangeWeak>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicIIncrement>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicIDecrement>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicIAdd>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicISub>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicUMin>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicUMax>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicAnd>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicOr>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicXor>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicIMin>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpAtomicIMax>(const spv_instruction_t *inst,
                                    const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpEmitStreamVertex>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpEndStreamPrimitive>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupAsyncCopy>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupWaitEvents>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupAll>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupAny>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupBroadcast>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupIAdd>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupFAdd>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupFMin>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupUMin>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupSMin>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupFMax>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupUMax>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupSMax>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpEnqueueMarker>(const spv_instruction_t *inst,
                                       const spv_opcode_desc opcodeEntry) {
}
#endif

#if 0
template <>
bool idUsage::isValid<OpEnqueueKernel>(const spv_instruction_t *inst,
                                       const spv_opcode_desc opcodeEntry) {
}
#endif

#if 0
template <>
bool idUsage::isValid<OpGetKernelNDrangeSubGroupCount>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGetKernelNDrangeMaxSubGroupSize>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGetKernelWorkGroupSize>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGetKernelPreferredWorkGroupSizeMultiple>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpRetainEvent>(const spv_instruction_t *inst,
                                     const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpReleaseEvent>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpCreateUserEvent>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpIsValidEvent>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpSetUserEventStatus>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpCaptureEventProfilingInfo>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGetDefaultQueue>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpBuildNDRange>(const spv_instruction_t *inst,
                                      const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpReadPipe>(const spv_instruction_t *inst,
                                  const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpWritePipe>(const spv_instruction_t *inst,
                                   const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpReservedReadPipe>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpReservedWritePipe>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpReserveReadPipePackets>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpReserveWritePipePackets>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpCommitReadPipe>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpCommitWritePipe>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpIsValidReserveId>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGetNumPipePackets>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGetMaxPipePackets>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupReserveReadPipePackets>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupReserveWritePipePackets>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupCommitReadPipe>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#if 0
template <>
bool idUsage::isValid<OpGroupCommitWritePipe>(
    const spv_instruction_t *inst, const spv_opcode_desc opcodeEntry) {}
#endif

#undef DIAG

bool idUsage::isValid(const spv_instruction_t* inst) {
  spv_opcode_desc opcodeEntry = nullptr;
  spvCheck(spvOpcodeTableValueLookup(opcodeTable, inst->opcode, &opcodeEntry),
           return false);
#define CASE(OpCode) \
  case Spv##OpCode:  \
    return isValid<Spv##OpCode>(inst, opcodeEntry);
#define FAIL(OpCode)                                     \
  case Spv##OpCode:                                      \
    std::cerr << "Not implemented: " << #OpCode << "\n"; \
    return false;
  switch (inst->opcode) {
    FAIL(OpUndef)
    CASE(OpName)
    CASE(OpMemberName)
    CASE(OpLine)
    CASE(OpDecorate)
    CASE(OpMemberDecorate)
    CASE(OpGroupDecorate)
    FAIL(OpGroupMemberDecorate)
    FAIL(OpExtInst)
    CASE(OpEntryPoint)
    CASE(OpExecutionMode)
    CASE(OpTypeVector)
    CASE(OpTypeMatrix)
    CASE(OpTypeSampler)
    CASE(OpTypeArray)
    CASE(OpTypeRuntimeArray)
    CASE(OpTypeStruct)
    CASE(OpTypePointer)
    CASE(OpTypeFunction)
    CASE(OpTypePipe)
    CASE(OpConstantTrue)
    CASE(OpConstantFalse)
    CASE(OpConstant)
    CASE(OpConstantComposite)
    CASE(OpConstantSampler)
    CASE(OpConstantNull)
    CASE(OpSpecConstantTrue)
    CASE(OpSpecConstantFalse)
    CASE(OpSpecConstant)
    FAIL(OpSpecConstantComposite)
    FAIL(OpSpecConstantOp)
    CASE(OpVariable)
    CASE(OpLoad)
    CASE(OpStore)
    CASE(OpCopyMemory)
    CASE(OpCopyMemorySized)
    FAIL(OpAccessChain)
    FAIL(OpInBoundsAccessChain)
    FAIL(OpArrayLength)
    FAIL(OpGenericPtrMemSemantics)
    CASE(OpFunction)
    CASE(OpFunctionParameter)
    CASE(OpFunctionCall)
    FAIL(OpConvertUToF)
    FAIL(OpConvertFToS)
    FAIL(OpConvertSToF)
    FAIL(OpUConvert)
    FAIL(OpSConvert)
    FAIL(OpFConvert)
    FAIL(OpConvertPtrToU)
    FAIL(OpConvertUToPtr)
    FAIL(OpPtrCastToGeneric)
    FAIL(OpGenericCastToPtr)
    FAIL(OpBitcast)
    FAIL(OpGenericCastToPtrExplicit)
    FAIL(OpSatConvertSToU)
    FAIL(OpSatConvertUToS)
    FAIL(OpVectorExtractDynamic)
    FAIL(OpVectorInsertDynamic)
    FAIL(OpVectorShuffle)
    FAIL(OpCompositeConstruct)
    FAIL(OpCompositeExtract)
    FAIL(OpCompositeInsert)
    FAIL(OpCopyObject)
    FAIL(OpTranspose)
    FAIL(OpSNegate)
    FAIL(OpFNegate)
    FAIL(OpNot)
    FAIL(OpIAdd)
    FAIL(OpFAdd)
    FAIL(OpISub)
    FAIL(OpFSub)
    FAIL(OpIMul)
    FAIL(OpFMul)
    FAIL(OpUDiv)
    FAIL(OpSDiv)
    FAIL(OpFDiv)
    FAIL(OpUMod)
    FAIL(OpSRem)
    FAIL(OpSMod)
    FAIL(OpFRem)
    FAIL(OpFMod)
    FAIL(OpVectorTimesScalar)
    FAIL(OpMatrixTimesScalar)
    FAIL(OpVectorTimesMatrix)
    FAIL(OpMatrixTimesVector)
    FAIL(OpMatrixTimesMatrix)
    FAIL(OpOuterProduct)
    FAIL(OpDot)
    FAIL(OpShiftRightLogical)
    FAIL(OpShiftRightArithmetic)
    FAIL(OpShiftLeftLogical)
    FAIL(OpBitwiseOr)
    FAIL(OpBitwiseXor)
    FAIL(OpBitwiseAnd)
    FAIL(OpAny)
    FAIL(OpAll)
    FAIL(OpIsNan)
    FAIL(OpIsInf)
    FAIL(OpIsFinite)
    FAIL(OpIsNormal)
    FAIL(OpSignBitSet)
    FAIL(OpLessOrGreater)
    FAIL(OpOrdered)
    FAIL(OpUnordered)
    FAIL(OpLogicalOr)
    FAIL(OpLogicalAnd)
    FAIL(OpSelect)
    FAIL(OpIEqual)
    FAIL(OpFOrdEqual)
    FAIL(OpFUnordEqual)
    FAIL(OpINotEqual)
    FAIL(OpFOrdNotEqual)
    FAIL(OpFUnordNotEqual)
    FAIL(OpULessThan)
    FAIL(OpSLessThan)
    FAIL(OpFOrdLessThan)
    FAIL(OpFUnordLessThan)
    FAIL(OpUGreaterThan)
    FAIL(OpSGreaterThan)
    FAIL(OpFOrdGreaterThan)
    FAIL(OpFUnordGreaterThan)
    FAIL(OpULessThanEqual)
    FAIL(OpSLessThanEqual)
    FAIL(OpFOrdLessThanEqual)
    FAIL(OpFUnordLessThanEqual)
    FAIL(OpUGreaterThanEqual)
    FAIL(OpSGreaterThanEqual)
    FAIL(OpFOrdGreaterThanEqual)
    FAIL(OpFUnordGreaterThanEqual)
    FAIL(OpDPdx)
    FAIL(OpDPdy)
    FAIL(OpFwidth)
    FAIL(OpDPdxFine)
    FAIL(OpDPdyFine)
    FAIL(OpFwidthFine)
    FAIL(OpDPdxCoarse)
    FAIL(OpDPdyCoarse)
    FAIL(OpFwidthCoarse)
    FAIL(OpPhi)
    FAIL(OpLoopMerge)
    FAIL(OpSelectionMerge)
    FAIL(OpBranch)
    FAIL(OpBranchConditional)
    FAIL(OpSwitch)
    CASE(OpReturnValue)
    FAIL(OpLifetimeStart)
    FAIL(OpLifetimeStop)
    FAIL(OpAtomicLoad)
    FAIL(OpAtomicStore)
    FAIL(OpAtomicExchange)
    FAIL(OpAtomicCompareExchange)
    FAIL(OpAtomicCompareExchangeWeak)
    FAIL(OpAtomicIIncrement)
    FAIL(OpAtomicIDecrement)
    FAIL(OpAtomicIAdd)
    FAIL(OpAtomicISub)
    FAIL(OpAtomicUMin)
    FAIL(OpAtomicUMax)
    FAIL(OpAtomicAnd)
    FAIL(OpAtomicOr)
    FAIL(OpAtomicSMin)
    FAIL(OpAtomicSMax)
    FAIL(OpEmitStreamVertex)
    FAIL(OpEndStreamPrimitive)
    FAIL(OpGroupAsyncCopy)
    FAIL(OpGroupWaitEvents)
    FAIL(OpGroupAll)
    FAIL(OpGroupAny)
    FAIL(OpGroupBroadcast)
    FAIL(OpGroupIAdd)
    FAIL(OpGroupFAdd)
    FAIL(OpGroupFMin)
    FAIL(OpGroupUMin)
    FAIL(OpGroupSMin)
    FAIL(OpGroupFMax)
    FAIL(OpGroupUMax)
    FAIL(OpGroupSMax)
    FAIL(OpEnqueueMarker)
    FAIL(OpEnqueueKernel)
    FAIL(OpGetKernelNDrangeSubGroupCount)
    FAIL(OpGetKernelNDrangeMaxSubGroupSize)
    FAIL(OpGetKernelWorkGroupSize)
    FAIL(OpGetKernelPreferredWorkGroupSizeMultiple)
    FAIL(OpRetainEvent)
    FAIL(OpReleaseEvent)
    FAIL(OpCreateUserEvent)
    FAIL(OpIsValidEvent)
    FAIL(OpSetUserEventStatus)
    FAIL(OpCaptureEventProfilingInfo)
    FAIL(OpGetDefaultQueue)
    FAIL(OpBuildNDRange)
    FAIL(OpReadPipe)
    FAIL(OpWritePipe)
    FAIL(OpReservedReadPipe)
    FAIL(OpReservedWritePipe)
    FAIL(OpReserveReadPipePackets)
    FAIL(OpReserveWritePipePackets)
    FAIL(OpCommitReadPipe)
    FAIL(OpCommitWritePipe)
    FAIL(OpIsValidReserveId)
    FAIL(OpGetNumPipePackets)
    FAIL(OpGetMaxPipePackets)
    FAIL(OpGroupReserveReadPipePackets)
    FAIL(OpGroupReserveWritePipePackets)
    FAIL(OpGroupCommitReadPipe)
    FAIL(OpGroupCommitWritePipe)
    default:
      return true;
  }
#undef FAIL
#undef CASE
}
}  // anonymous namespace

spv_result_t spvValidateInstructionIDs(
    const spv_instruction_t* pInsts, const uint64_t instCount,
    const spv_id_info_t* pIdUses, const uint64_t idUsesCount,
    const spv_id_info_t* pIdDefs, const uint64_t idDefsCount,
    const spv_opcode_table opcodeTable, const spv_operand_table operandTable,
    const spv_ext_inst_table extInstTable, spv_position position,
    spv_diagnostic* pDiag) {
  idUsage idUsage(opcodeTable, operandTable, extInstTable, pIdUses, idUsesCount,
                  pIdDefs, idDefsCount, pInsts, instCount, position, pDiag);
  for (uint64_t instIndex = 0; instIndex < instCount; ++instIndex) {
    spvCheck(!idUsage.isValid(&pInsts[instIndex]), return SPV_ERROR_INVALID_ID);
    position->index += pInsts[instIndex].words.size();
  }
  return SPV_SUCCESS;
}
