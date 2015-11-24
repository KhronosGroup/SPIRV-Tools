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

#include "validate.h"

#include "binary.h"
#include "diagnostic.h"
#include "endian.h"
#include "instruction.h"
#include "libspirv/libspirv.h"
#include "opcode.h"
#include "operand.h"
#include "spirv_constant.h"

#include <cassert>
#include <cstdio>
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

using std::vector;
using std::unordered_set;

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

// TODO(umar): Move this class to another file
class ValidationState_t {
 public:
  ValidationState_t(spv_diagnostic* diag)
      : diagnostic_(diag), instruction_counter_(0) {}

  spv_result_t definedIds(uint32_t id) {
    if (defined_ids_.find(id) == std::end(defined_ids_)) {
      defined_ids_.insert(id);
    } else {
      return diag(SPV_ERROR_INVALID_ID)
             << "ID cannot be assigned multiple times";
    }
    return SPV_SUCCESS;
  }

  spv_result_t forwardDeclareId(uint32_t id) {
    unresolved_forward_ids_.insert(id);
    return SPV_SUCCESS;
  }

  spv_result_t removeIfForwardDeclared(uint32_t id) {
    unresolved_forward_ids_.erase(id);
    return SPV_SUCCESS;
  }

  size_t unresolvedForwardIdCount() const {
    return unresolved_forward_ids_.size();
  }

  //
  bool isDefinedId(uint32_t id) const {
    return defined_ids_.find(id) != std::end(defined_ids_);
  }

  // Increments the instruction count. Used for diagnostic
  int incrementInstructionCount() { return instruction_counter_++; }

  libspirv::DiagnosticStream diag(spv_result_t error_code) const {
    return libspirv::DiagnosticStream({0, 0, (uint64_t)(instruction_counter_)},
                                      diagnostic_, error_code);
  }

 private:
  spv_diagnostic* diagnostic_;
  // Tracks the number of instructions evaluated by the validator
  int instruction_counter_;

  // All IDs which have been defined
  unordered_set<uint32_t> defined_ids_;

  // IDs which have been forward declared but have not been defined
  unordered_set<uint32_t> unresolved_forward_ids_;
};

spv_result_t SSAPass(ValidationState_t& _, bool can_have_forward_declared_ids,
                     const spv_parsed_instruction_t* inst) {
  for (int i = 0; i < inst->num_operands; i++) {
    const spv_parsed_operand_t& operand = inst->operands[i];
    const spv_operand_type_t& type = operand.type;
    const uint32_t* operand_ptr = inst->words + operand.offset;

    auto ret = SPV_ERROR_INTERNAL;
    switch (type) {
      case SPV_OPERAND_TYPE_RESULT_ID: {
        _.removeIfForwardDeclared(*operand_ptr);
        ret = _.definedIds(*operand_ptr);
        break;
      }
      case SPV_OPERAND_TYPE_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
      case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      case SPV_OPERAND_TYPE_SCOPE_ID: {
        if (can_have_forward_declared_ids) {
          if (_.isDefinedId(*operand_ptr)) {
            ret = SPV_SUCCESS;
          } else {
            ret = _.forwardDeclareId(*operand_ptr);
          }
        } else {
          if (_.isDefinedId(*operand_ptr)) {
            ret = SPV_SUCCESS;
          } else {
            ret = _.diag(SPV_ERROR_INVALID_ID) << "ID " << *operand_ptr
                                               << " has not been declared";
          }
        }
        break;
      }
      default:
        ret = SPV_SUCCESS;
        break;
    }
    if (SPV_SUCCESS != ret) {
      return ret;
    }
  }
  return SPV_SUCCESS;
}

spv_result_t pushInstructions(void* user_data,
                              const spv_parsed_instruction_t* inst) {
  ValidationState_t& _ = *(reinterpret_cast<ValidationState_t*>(user_data));
  _.incrementInstructionCount();

  vector<SpvOp> instructions_with_forward_ids = {
      SpvOpExecutionMode, SpvOpEntryPoint, SpvOpPhi, SpvOpFunctionCall,
      SpvOpName, SpvOpMemberName, SpvOpDecorate, SpvOpMemberDecorate,
      SpvOpGroupDecorate};

  bool can_have_forward_declared_ids = false;
  for (auto& instruction : instructions_with_forward_ids) {
    can_have_forward_declared_ids |= inst->opcode == instruction;
  }

  return SSAPass(_, can_have_forward_declared_ids, inst);

}

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

  ValidationState_t vstate(pDiagnostic);
  auto err = spvBinaryParse(context, &vstate, binary->code, binary->wordCount,
                            setHeader, pushInstructions, pDiagnostic);

  if (vstate.unresolvedForwardIdCount() > 0) {
    // TODO(umar): print undefined IDs
    return vstate.diag(SPV_ERROR_INVALID_ID)
           << "Some forward referenced IDs have not be defined. \n";
  }
  if (err) {
    return err;
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
