// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "validate.h"
#include "validate_types.h"

#include "binary.h"
#include "diagnostic.h"
#include "instruction.h"
#include "libspirv/libspirv.h"
#include "opcode.h"
#include "operand.h"
#include "spirv_constant.h"
#include "spirv_endian.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <functional>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using std::function;
using std::map;
using std::ostream_iterator;
using std::placeholders::_1;
using std::string;
using std::stringstream;
using std::transform;
using std::unordered_set;
using std::vector;

using libspirv::ValidationState_t;
using libspirv::kLayoutFunctionDeclarations;
using libspirv::kLayoutFunctionDefinitions;
using libspirv::kLayoutMemoryModel;
using libspirv::FunctionDecl::kFunctionDeclDeclaration;
using libspirv::FunctionDecl::kFunctionDeclDefinition;

#define spvCheckReturn(expression) \
  if (spv_result_t error = (expression)) return error;

#if 0
spv_result_t spvValidateOperandsString(const uint32_t* words,
                                       const uint16_t wordCount,
                                       spv_position position,
                                       spv_diagnostic* pDiagnostic) {
  const char* str = (const char*)words;
  uint64_t strWordCount = strlen(str) / sizeof(uint32_t) + 1;
  if (strWordCount < wordCount) {
    DIAGNOSTIC << "Instruction word count is too short, string extends past "
                  "end of instruction.";
    return SPV_WARNING;
  }
  return SPV_SUCCESS;
}

spv_result_t spvValidateOperandsLiteral(const uint32_t* words,
                                        const uint32_t length,
                                        const uint16_t maxLength,
                                        spv_position position,
                                        spv_diagnostic* pDiagnostic) {
  // NOTE: A literal could either be a number consuming up to 2 words or a
  // null terminated string.
  (void)words;
  (void)length;
  (void)maxLength;
  (void)position;
  (void)pDiagnostic;
  return SPV_UNSUPPORTED;
}

spv_result_t spvValidateOperandValue(const spv_operand_type_t type,
                                     const uint32_t word,
                                     const spv_operand_table operandTable,
                                     spv_position position,
                                     spv_diagnostic* pDiagnostic) {
  switch (type) {
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_TYPE_ID:
    case SPV_OPERAND_TYPE_RESULT_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID: {
      // NOTE: ID's are validated in SPV_VALIDATION_LEVEL_1, this is
      // SPV_VALIDATION_LEVEL_0
    } break;
    case SPV_OPERAND_TYPE_LITERAL_INTEGER: {
      // NOTE: Implicitly valid as they are encoded as 32 bit value
    } break;
    case SPV_OPERAND_TYPE_SOURCE_LANGUAGE:
    case SPV_OPERAND_TYPE_EXECUTION_MODEL:
    case SPV_OPERAND_TYPE_ADDRESSING_MODEL:
    case SPV_OPERAND_TYPE_MEMORY_MODEL:
    case SPV_OPERAND_TYPE_EXECUTION_MODE:
    case SPV_OPERAND_TYPE_STORAGE_CLASS:
    case SPV_OPERAND_TYPE_DIMENSIONALITY:
    case SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE:
    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_FP_ROUNDING_MODE:
    case SPV_OPERAND_TYPE_LINKAGE_TYPE:
    case SPV_OPERAND_TYPE_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE:
    case SPV_OPERAND_TYPE_DECORATION:
    case SPV_OPERAND_TYPE_BUILT_IN:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
    case SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_SCOPE_ID:
    case SPV_OPERAND_TYPE_GROUP_OPERATION:
    case SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS:
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO: {
      spv_operand_desc operandEntry = nullptr;
      spv_result_t error =
          spvOperandTableValueLookup(operandTable, type, word, &operandEntry);
      if (error) {
        DIAGNOSTIC << "Invalid '" << spvOperandTypeStr(type) << "' operand '"
                   << word << "'.";
        return error;
      }
    } break;
    default:
      assert(0 && "Invalid operand types should already have been caught!");
  }
  return SPV_SUCCESS;
}

spv_result_t spvValidateBasic(const spv_instruction_t* pInsts,
                              const uint64_t instCount,
                              const spv_opcode_table opcodeTable,
                              const spv_operand_table operandTable,
                              spv_position position,
                              spv_diagnostic* pDiagnostic) {
  for (uint64_t instIndex = 0; instIndex < instCount; ++instIndex) {
    const uint32_t* words = pInsts[instIndex].words.data();
    uint16_t wordCount;
    SpvOp opcode;
    spvOpcodeSplit(words[0], &wordCount, &opcode);

    spv_opcode_desc opcodeEntry = nullptr;
    if (spvOpcodeTableValueLookup(opcodeTable, opcode, &opcodeEntry)) {
      DIAGNOSTIC << "Invalid Opcode '" << opcode << "'.";
      return SPV_ERROR_INVALID_BINARY;
    }
    position->index++;

    if (opcodeEntry->numTypes > wordCount) {
      DIAGNOSTIC << "Instruction word count '" << wordCount
                 << "' is not small, expected at least '"
                 << opcodeEntry->numTypes << "'.";
      return SPV_ERROR_INVALID_BINARY;
    }

    spv_operand_desc operandEntry = nullptr;
    for (uint16_t index = 1; index < pInsts[instIndex].words.size();
         ++index, position->index++) {
      const uint32_t word = words[index];

      // TODO(dneto): This strategy is inadequate for dealing with operations
      // with varying kinds or numbers of logical operands.  See the definition
      // of spvBinaryOperandInfo for more.
      // We should really parse the instruction and capture and use
      // the elaborated list of logical operands generated as a side effect
      // of the parse.
      spv_operand_type_t type = spvBinaryOperandInfo(
          word, index, opcodeEntry, operandTable, &operandEntry);

      if (SPV_OPERAND_TYPE_LITERAL_STRING == type) {
        spvCheckReturn(spvValidateOperandsString(
            words + index, wordCount - index, position, pDiagnostic));
        // NOTE: String literals are always at the end of Opcodes
        break;
      } else if (SPV_OPERAND_TYPE_LITERAL_INTEGER == type) {
        // spvCheckReturn(spvValidateOperandsNumber(
        //    words + index, wordCount - index, 2, position, pDiagnostic));
      } else {
        spvCheckReturn(spvValidateOperandValue(type, word, operandTable,
                                               position, pDiagnostic));
      }
    }
  }

  return SPV_SUCCESS;
}
#endif

spv_result_t spvValidateIDs(const spv_instruction_t* pInsts,
                            const uint64_t count, const uint32_t bound,
                            const spv_opcode_table opcodeTable,
                            const spv_operand_table operandTable,
                            const spv_ext_inst_table extInstTable,
                            spv_position position,
                            spv_diagnostic* pDiagnostic) {
  std::vector<spv_id_info_t> idUses;
  std::vector<spv_id_info_t> idDefs;

  for (uint64_t instIndex = 0; instIndex < count; ++instIndex) {
    const uint32_t* words = pInsts[instIndex].words.data();
    SpvOp opcode;
    spvOpcodeSplit(words[0], nullptr, &opcode);

    spv_opcode_desc opcodeEntry = nullptr;
    if (spvOpcodeTableValueLookup(opcodeTable, opcode, &opcodeEntry)) {
      DIAGNOSTIC << "Invalid Opcode '" << opcode << "'.";
      return SPV_ERROR_INVALID_BINARY;
    }

    spv_operand_desc operandEntry = nullptr;
    position->index++;  // NOTE: Account for Opcode word
    for (uint16_t index = 1; index < pInsts[instIndex].words.size();
         ++index, position->index++) {
      const uint32_t word = words[index];

      spv_operand_type_t type = spvBinaryOperandInfo(
          word, index, opcodeEntry, operandTable, &operandEntry);

      if (SPV_OPERAND_TYPE_RESULT_ID == type || SPV_OPERAND_TYPE_ID == type) {
        if (0 == word) {
          DIAGNOSTIC << "Invalid ID of '0' is not allowed.";
          return SPV_ERROR_INVALID_ID;
        }
        if (bound < word) {
          DIAGNOSTIC << "Invalid ID '" << word << "' exceeds the bound '"
                     << bound << "'.";
          return SPV_ERROR_INVALID_ID;
        }
      }

      if (SPV_OPERAND_TYPE_RESULT_ID == type) {
        idDefs.push_back(
            {word, opcodeEntry->opcode, &pInsts[instIndex], *position});
      }

      if (SPV_OPERAND_TYPE_ID == type) {
        idUses.push_back({word, opcodeEntry->opcode, nullptr, *position});
      }
    }
  }

  // NOTE: Error on redefined ID
  for (size_t outerIndex = 0; outerIndex < idDefs.size(); ++outerIndex) {
    for (size_t innerIndex = 0; innerIndex < idDefs.size(); ++innerIndex) {
      if (outerIndex == innerIndex) {
        continue;
      }
      if (idDefs[outerIndex].id == idDefs[innerIndex].id) {
        DIAGNOSTIC << "Multiply defined ID '" << idDefs[outerIndex].id << "'.";
        return SPV_ERROR_INVALID_ID;
      }
    }
  }

  // NOTE: Validate ID usage, including use of undefined ID's
  position->index = SPV_INDEX_INSTRUCTION;
  if (spvValidateInstructionIDs(pInsts, count, idUses.data(), idUses.size(),
                                idDefs.data(), idDefs.size(), opcodeTable,
                                operandTable, extInstTable, position,
                                pDiagnostic))
    return SPV_ERROR_INVALID_ID;

  return SPV_SUCCESS;
}

namespace {

// TODO(umar): Validate header
// TODO(umar): The Id bound should be validated also. But you can only do that
// after you've seen all the instructions in the module.
// TODO(umar): The binary parser validates the magic word, and the length of the
// header, but nothing else.
spv_result_t setHeader(void* user_data, spv_endianness_t endian, uint32_t magic,
                       uint32_t version, uint32_t generator, uint32_t id_bound,
                       uint32_t reserved) {
  (void)user_data;
  (void)endian;
  (void)magic;
  (void)version;
  (void)generator;
  (void)id_bound;
  (void)reserved;
  return SPV_SUCCESS;
}

// Performs SSA validation on the IDs of an instruction. The
// can_have_forward_declared_ids  functor should return true if the
// instruction operand's ID can be forward referenced.
//
// TODO(umar): Use dominators to correctly validate SSA. For example, the result
// id from a 'then' block cannot dominate its usage in the 'else' block. This
// is not yet performed by this funciton.
spv_result_t SsaPass(ValidationState_t& _,
                     function<bool(unsigned)> can_have_forward_declared_ids,
                     const spv_parsed_instruction_t* inst) {
  if (_.is_enabled(SPV_VALIDATE_SSA_BIT)) {
    for (unsigned i = 0; i < inst->num_operands; i++) {
      const spv_parsed_operand_t& operand = inst->operands[i];
      const spv_operand_type_t& type = operand.type;
      const uint32_t* operand_ptr = inst->words + operand.offset;

      auto ret = SPV_ERROR_INTERNAL;
      switch (type) {
        case SPV_OPERAND_TYPE_RESULT_ID:
          _.removeIfForwardDeclared(*operand_ptr);
          ret = _.defineId(*operand_ptr);
          break;
        case SPV_OPERAND_TYPE_ID:
        case SPV_OPERAND_TYPE_TYPE_ID:
        case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
        case SPV_OPERAND_TYPE_SCOPE_ID:
          if (_.isDefinedId(*operand_ptr)) {
            ret = SPV_SUCCESS;
          } else if (can_have_forward_declared_ids(i)) {
            ret = _.forwardDeclareId(*operand_ptr);
          } else {
            ret = _.diag(SPV_ERROR_INVALID_ID) << "ID "
                                               << _.getIdName(*operand_ptr)
                                               << " has not been defined";
          }
          break;
        default:
          ret = SPV_SUCCESS;
          break;
      }
      if (SPV_SUCCESS != ret) {
        return ret;
      }
    }
  }
  return SPV_SUCCESS;
}

// This funciton takes the opcode of an instruction and returns
// a function object that will return true if the index
// of the operand can be forwarad declared. This function will
// used in the SSA validation stage of the pipeline
function<bool(unsigned)> getCanBeForwardDeclaredFunction(SpvOp opcode) {
  function<bool(unsigned index)> out;
  switch (opcode) {
    case SpvOpExecutionMode:
    case SpvOpEntryPoint:
    case SpvOpName:
    case SpvOpMemberName:
    case SpvOpSelectionMerge:
    case SpvOpDecorate:
    case SpvOpMemberDecorate:
    case SpvOpBranch:
    case SpvOpLoopMerge:
      out = [](unsigned) { return true; };
      break;
    case SpvOpGroupDecorate:
    case SpvOpGroupMemberDecorate:
    case SpvOpBranchConditional:
    case SpvOpSwitch:
      out = [](unsigned index) { return index != 0; };
      break;

    case SpvOpFunctionCall:
      out = [](unsigned index) { return index == 2; };
      break;

    case SpvOpPhi:
      out = [](unsigned index) { return index > 1; };
      break;

    case SpvOpEnqueueKernel:
      out = [](unsigned index) { return index == 8; };
      break;

    case SpvOpGetKernelNDrangeSubGroupCount:
    case SpvOpGetKernelNDrangeMaxSubGroupSize:
      out = [](unsigned index) { return index == 3; };
      break;

    case SpvOpGetKernelWorkGroupSize:
    case SpvOpGetKernelPreferredWorkGroupSizeMultiple:
      out = [](unsigned index) { return index == 2; };
      break;

    default:
      out = [](unsigned) { return false; };
      break;
  }
  return out;
}

// Improves diagnostic messages by collecting names of IDs
// NOTE: This function returns void and is not involved in validation
void DebugInstructionPass(ValidationState_t& _,
                          const spv_parsed_instruction_t* inst) {
  switch (inst->opcode) {
    case SpvOpName: {
      const uint32_t target = *(inst->words + inst->operands[0].offset);
      const char* str =
          reinterpret_cast<const char*>(inst->words + inst->operands[1].offset);
      _.assignNameToId(target, str);
    } break;
    case SpvOpMemberName: {
      const uint32_t target = *(inst->words + inst->operands[0].offset);
      const char* str =
          reinterpret_cast<const char*>(inst->words + inst->operands[2].offset);
      _.assignNameToId(target, str);
    } break;
    case SpvOpSourceContinued:
    case SpvOpSource:
    case SpvOpSourceExtension:
    case SpvOpString:
    case SpvOpLine:
    case SpvOpNoLine:

    default:
      break;
  }
}

// TODO(umar): Check linkage capabilities for function declarations
// TODO(umar): Better error messages
// NOTE: This function does not handle CFG related validation
// Performs logical layout validation. See Section 2.4
spv_result_t ModuleLayoutPass(ValidationState_t& _,
                              const spv_parsed_instruction_t* inst) {
  if (_.is_enabled(SPV_VALIDATE_LAYOUT_BIT)) {
    SpvOp opcode = inst->opcode;

    if (_.getLayoutStage() < kLayoutFunctionDeclarations) {
      // Module scoped instructions are processed by determining if the opcode
      // is part of the current stage. If it is not then the next stage is
      // checked.
      while (_.isOpcodeInCurrentLayoutStage(opcode) == false) {
        _.progressToNextLayoutStageOrder();

        if (_.getLayoutStage() == kLayoutMemoryModel &&
            opcode != SpvOpMemoryModel) {
          return _.diag(SPV_ERROR_INVALID_LAYOUT)
                 << spvOpcodeString(opcode)
                 << " cannot appear before the memory model instruction";
        }

        if (_.getLayoutStage() == kLayoutFunctionDeclarations) {
          // All module stages have been processed. Recursivly call
          // ModuleLayoutPass to process the next section of the module
          return ModuleLayoutPass(_, inst);
        }
      }

      if (opcode == SpvOpVariable) {
        const uint32_t* storage_class = inst->words + inst->operands[2].offset;
        if (*storage_class == SpvStorageClassFunction) {
          return _.diag(SPV_ERROR_INVALID_LAYOUT)
                 << "Variables cannot have a function[7] storage class "
                    "outside of a function";
        }
      }
    } else {
      // This ensures no module instructions are called during function
      // declarations
      switch (opcode) {
        case SpvOpCapability:
        case SpvOpExtension:
        case SpvOpExtInstImport:
        case SpvOpMemoryModel:
        case SpvOpEntryPoint:
        case SpvOpExecutionMode:
        case SpvOpSourceContinued:
        case SpvOpSource:
        case SpvOpSourceExtension:
        case SpvOpString:
        case SpvOpName:
        case SpvOpMemberName:
        case SpvOpDecorate:
        case SpvOpMemberDecorate:
        case SpvOpGroupDecorate:
        case SpvOpGroupMemberDecorate:
        case SpvOpDecorationGroup:
        case SpvOpTypeVoid:
        case SpvOpTypeBool:
        case SpvOpTypeInt:
        case SpvOpTypeFloat:
        case SpvOpTypeVector:
        case SpvOpTypeMatrix:
        case SpvOpTypeImage:
        case SpvOpTypeSampler:
        case SpvOpTypeSampledImage:
        case SpvOpTypeArray:
        case SpvOpTypeRuntimeArray:
        case SpvOpTypeStruct:
        case SpvOpTypeOpaque:
        case SpvOpTypePointer:
        case SpvOpTypeFunction:
        case SpvOpTypeEvent:
        case SpvOpTypeDeviceEvent:
        case SpvOpTypeReserveId:
        case SpvOpTypeQueue:
        case SpvOpTypePipe:
        case SpvOpTypeForwardPointer:
        case SpvOpConstantTrue:
        case SpvOpConstantFalse:
        case SpvOpConstant:
        case SpvOpConstantComposite:
        case SpvOpConstantSampler:
        case SpvOpConstantNull:
        case SpvOpSpecConstantTrue:
        case SpvOpSpecConstantFalse:
        case SpvOpSpecConstant:
        case SpvOpSpecConstantComposite:
        case SpvOpSpecConstantOp:
          return _.diag(SPV_ERROR_INVALID_LAYOUT) << "Invalid Layout";
        case SpvOpVariable: {
          const uint32_t* storage_class =
              inst->words + inst->operands[2].offset;
          if (*storage_class != SpvStorageClassFunction)
            return _.diag(SPV_ERROR_INVALID_LAYOUT)
                   << "All Variable instructions in a function must have a "
                      "storage class of function[7]";
        } break;
        default:
          break;
      }
      if (_.getLayoutStage() == kLayoutFunctionDeclarations) {
        switch (opcode) {
          case SpvOpFunction:
            if (_.in_function_body()) {
              return _.diag(SPV_ERROR_INVALID_LAYOUT)
                     << "Cannot declare a function in a function body";
            }
            spvCheckReturn(_.get_functions().RegisterFunction(
                inst->result_id, inst->type_id,
                inst->words[inst->operands[2].offset],
                inst->words[inst->operands[3].offset]));
            break;
          case SpvOpFunctionParameter:
            if (_.in_function_body() == false) {
              return _.diag(SPV_ERROR_INVALID_LAYOUT) << "Function parameter "
                                                         "instructions must be "
                                                         "in a function body";
            }
            spvCheckReturn(_.get_functions().RegisterFunctionParameter(
                inst->result_id, inst->type_id));
            break;
          case SpvOpLine:  // ??
            break;
          case SpvOpLabel:
            if (_.in_function_body() == false) {
              return _.diag(SPV_ERROR_INVALID_LAYOUT)
                     << "Label instructions must be in a function body";
            }
            _.progressToNextLayoutStageOrder();
            spvCheckReturn(_.get_functions().RegisterSetFunctionDeclType(
                kFunctionDeclDefinition));
            break;
          case SpvOpFunctionEnd:
            assert(_.get_functions().get_block_count() ==
                       0  // NOTE: This should not happen
                   &&
                   "Function contains blocks in function declaration section.");
            if (_.in_function_body() == false) {
              return _.diag(SPV_ERROR_INVALID_LAYOUT)
                     << "Function end instructions must be in a function body";
            }
            spvCheckReturn(_.get_functions().RegisterSetFunctionDeclType(
                kFunctionDeclDeclaration));
            spvCheckReturn(_.get_functions().RegisterFunctionEnd());
            break;
          default:
            return _.diag(SPV_ERROR_INVALID_LAYOUT)
                   << "A function must begin with a label";
            break;
        }
      }
      // NOTE: Function definitions are handled by the CFGPass
    }
  }
  return SPV_SUCCESS;
}

// TODO(umar): Support for merge instructions
// TODO(umar): Structured control flow checks
spv_result_t CfgPass(ValidationState_t& _,
                     const spv_parsed_instruction_t* inst) {
  if (_.getLayoutStage() == kLayoutFunctionDefinitions) {
    SpvOp opcode = inst->opcode;
    switch (opcode) {
      case SpvOpFunction:
        spvCheckReturn(_.get_functions().RegisterFunction(
            inst->result_id, inst->type_id,
            inst->words[inst->operands[2].offset],
            inst->words[inst->operands[3].offset]));
        spvCheckReturn(_.get_functions().RegisterSetFunctionDeclType(
            kFunctionDeclDefinition));
        break;
      case SpvOpFunctionParameter:
        spvCheckReturn(_.get_functions().RegisterFunctionParameter(
            inst->result_id, inst->type_id));
        break;
      case SpvOpFunctionEnd:
        if (_.get_functions().get_block_count() == 0)
          return _.diag(SPV_ERROR_INVALID_LAYOUT) << "Function declarations "
                                                     "must appear before "
                                                     "function definitions.";
        spvCheckReturn(_.get_functions().RegisterFunctionEnd());
        break;
      case SpvOpLabel:
        spvCheckReturn(_.get_functions().RegisterBlock(inst->result_id));
        break;
      case SpvOpBranch:
      case SpvOpBranchConditional:
      case SpvOpSwitch:
      case SpvOpKill:
      case SpvOpReturn:
      case SpvOpReturnValue:
      case SpvOpUnreachable:
        spvCheckReturn(_.get_functions().RegisterBlockEnd());
        break;
      default:
        if (_.in_block() == false) {
          return _.diag(SPV_ERROR_INVALID_LAYOUT) << spvOpcodeString(opcode)
                                                  << " must appear in a block";
        }
        break;
    }
  }
  return SPV_SUCCESS;
}

spv_result_t ProcessInstructions(void* user_data,
                                 const spv_parsed_instruction_t* inst) {
  ValidationState_t& _ = *(reinterpret_cast<ValidationState_t*>(user_data));
  _.incrementInstructionCount();

  auto can_have_forward_declared_ids =
      getCanBeForwardDeclaredFunction(inst->opcode);

  DebugInstructionPass(_, inst);

  // TODO(umar): Perform data rules pass
  // TODO(umar): Perform instruction validation pass
  spvCheckReturn(ModuleLayoutPass(_, inst));
  spvCheckReturn(CfgPass(_, inst));
  spvCheckReturn(SsaPass(_, can_have_forward_declared_ids, inst));

  return SPV_SUCCESS;
}

}  // anonymous namespace

spv_result_t spvValidate(const spv_const_context context,
                         const spv_const_binary binary, const uint32_t options,
                         spv_diagnostic* pDiagnostic) {
  if (!pDiagnostic) return SPV_ERROR_INVALID_DIAGNOSTIC;

  spv_endianness_t endian;
  spv_position_t position = {};
  if (spvBinaryEndianness(binary, &endian)) {
    DIAGNOSTIC << "Invalid SPIR-V magic number.";
    return SPV_ERROR_INVALID_BINARY;
  }

  spv_header_t header;
  if (spvBinaryHeaderGet(binary, endian, &header)) {
    DIAGNOSTIC << "Invalid SPIR-V header.";
    return SPV_ERROR_INVALID_BINARY;
  }

  // NOTE: Parse the module and perform inline validation checks. These
  // checks do not require the the knowledge of the whole module.
  ValidationState_t vstate(pDiagnostic, options);
  spvCheckReturn(spvBinaryParse(context, &vstate, binary->code, binary->wordCount,
                                setHeader, ProcessInstructions, pDiagnostic));

  // TODO(umar): Add validation checks which require the parsing of the entire
  // module. Use the information from the processInstructions pass to make
  // the checks.

  if (vstate.unresolvedForwardIdCount() > 0) {
    stringstream ss;
    vector<uint32_t> ids = vstate.unresolvedForwardIds();

    transform(begin(ids), end(ids), ostream_iterator<string>(ss, " "),
              bind(&ValidationState_t::getIdName, vstate, _1));

    auto id_str = ss.str();
    return vstate.diag(SPV_ERROR_INVALID_ID)
           << "The following forward referenced IDs have not be defined:\n"
           << id_str.substr(0, id_str.size() - 1);
  }

  // NOTE: Copy each instruction for easier processing
  std::vector<spv_instruction_t> instructions;
  uint64_t index = SPV_INDEX_INSTRUCTION;
  while (index < binary->wordCount) {
    uint16_t wordCount;
    SpvOp opcode;
    spvOpcodeSplit(spvFixWord(binary->code[index], endian), &wordCount,
                   &opcode);
    spv_instruction_t inst;
    spvInstructionCopy(&binary->code[index], opcode, wordCount, endian, &inst);
    instructions.push_back(inst);
    index += wordCount;
  }

  if (spvIsInBitfield(SPV_VALIDATE_ID_BIT, options)) {
    position.index = SPV_INDEX_INSTRUCTION;
    spvCheckReturn(
        spvValidateIDs(instructions.data(), instructions.size(), header.bound,
                       context->opcode_table, context->operand_table,
                       context->ext_inst_table, &position, pDiagnostic));
  }

  return SPV_SUCCESS;
}
