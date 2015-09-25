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

#include <libspirv/libspirv.h>
#include "binary.h"
#include "diagnostic.h"
#include "ext_inst.h"
#include "opcode.h"
#include "operand.h"

#include <assert.h>
#include <string.h>

#include <sstream>

// Binary API

enum {
  I32_ENDIAN_LITTLE = 0x03020100ul,
  I32_ENDIAN_BIG = 0x00010203ul,
};

static const union {
  unsigned char bytes[4];
  uint32_t value;
} o32_host_order = {{0, 1, 2, 3}};

#define I32_ENDIAN_HOST (o32_host_order.value)

spv_result_t spvBinaryEndianness(const spv_binary binary,
                                 spv_endianness_t *pEndian) {
  if (!binary->code || !binary->wordCount) return SPV_ERROR_INVALID_BINARY;
  if (!pEndian) return SPV_ERROR_INVALID_POINTER;

  uint8_t bytes[4];
  memcpy(bytes, binary->code, sizeof(uint32_t));

  if (0x03 == bytes[0] && 0x02 == bytes[1] && 0x23 == bytes[2] &&
      0x07 == bytes[3]) {
    *pEndian = SPV_ENDIANNESS_LITTLE;
    return SPV_SUCCESS;
  }

  if (0x07 == bytes[0] && 0x23 == bytes[1] && 0x02 == bytes[2] &&
      0x03 == bytes[3]) {
    *pEndian = SPV_ENDIANNESS_BIG;
    return SPV_SUCCESS;
  }

  return SPV_ERROR_INVALID_BINARY;
}

uint32_t spvFixWord(const uint32_t word, const spv_endianness_t endian) {
  if ((SPV_ENDIANNESS_LITTLE == endian && I32_ENDIAN_HOST == I32_ENDIAN_BIG) ||
      (SPV_ENDIANNESS_BIG == endian && I32_ENDIAN_HOST == I32_ENDIAN_LITTLE)) {
    return (word & 0x000000ff) << 24 | (word & 0x0000ff00) << 8 |
           (word & 0x00ff0000) >> 8 | (word & 0xff000000) >> 24;
  }

  return word;
}

uint64_t spvFixDoubleWord(const uint32_t low, const uint32_t high,
                          const spv_endianness_t endian) {
  return (uint64_t(spvFixWord(high, endian)) << 32) | spvFixWord(low, endian);
}

spv_result_t spvBinaryHeaderGet(const spv_binary binary,
                                const spv_endianness_t endian,
                                spv_header_t *pHeader) {
  if (!binary->code || !binary->wordCount) return SPV_ERROR_INVALID_BINARY;
  if (!pHeader) return SPV_ERROR_INVALID_POINTER;

  // TODO: Validation checking?
  pHeader->magic = spvFixWord(binary->code[SPV_INDEX_MAGIC_NUMBER], endian);
  pHeader->version = spvFixWord(binary->code[SPV_INDEX_VERSION_NUMBER], endian);
  pHeader->generator =
      spvFixWord(binary->code[SPV_INDEX_GENERATOR_NUMBER], endian);
  pHeader->bound = spvFixWord(binary->code[SPV_INDEX_BOUND], endian);
  pHeader->schema = spvFixWord(binary->code[SPV_INDEX_SCHEMA], endian);
  pHeader->instructions = &binary->code[SPV_INDEX_INSTRUCTION];

  return SPV_SUCCESS;
}

spv_result_t spvBinaryHeaderSet(spv_binary_t *binary, const uint32_t bound) {
  if (!binary) return SPV_ERROR_INVALID_BINARY;
  if (!binary->code || !binary->wordCount) return SPV_ERROR_INVALID_BINARY;

  binary->code[SPV_INDEX_MAGIC_NUMBER] = SPV_MAGIC_NUMBER;
  binary->code[SPV_INDEX_VERSION_NUMBER] = SPV_VERSION_NUMBER;
  binary->code[SPV_INDEX_GENERATOR_NUMBER] = SPV_GENERATOR_KHRONOS;
  binary->code[SPV_INDEX_BOUND] = bound;
  binary->code[SPV_INDEX_SCHEMA] = 0;  // NOTE: Reserved

  return SPV_SUCCESS;
}

spv_result_t spvBinaryEncodeU32(const uint32_t value, spv_instruction_t *pInst,
                                const spv_position position,
                                spv_diagnostic *pDiagnostic) {
  if (pInst->wordCount + 1 > SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX) {
    DIAGNOSTIC << "Instruction word count '"
               << SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX << "' exceeded.";
    return SPV_ERROR_INVALID_TEXT;
  }

  pInst->words[pInst->wordCount++] = (uint32_t)value;
  return SPV_SUCCESS;
}

spv_result_t spvBinaryEncodeU64(const uint64_t value, spv_instruction_t *pInst,
                                const spv_position position,
                                spv_diagnostic *pDiagnostic) {
  if (pInst->wordCount + 2 > SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX) {
    DIAGNOSTIC << "Instruction word count '"
               << SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX << "' exceeded.";
    return SPV_ERROR_INVALID_TEXT;
  }

  uint32_t low = (uint32_t)(0x00000000ffffffff & value);
  uint32_t high = (uint32_t)((0xffffffff00000000 & value) >> 32);
  pInst->words[pInst->wordCount++] = low;
  pInst->words[pInst->wordCount++] = high;
  return SPV_SUCCESS;
}

spv_result_t spvBinaryEncodeString(const char *str, spv_instruction_t *pInst,
                                   const spv_position position,
                                   spv_diagnostic *pDiagnostic) {
  size_t length = strlen(str);
  size_t wordCount = (length / 4) + 1;
  if ((sizeof(uint32_t) * pInst->wordCount) + length >
      sizeof(uint32_t) * SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX) {
    DIAGNOSTIC << "Instruction word count '"
               << SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX << "'exceeded.";
    return SPV_ERROR_INVALID_TEXT;
  }

  char *dest = (char *)&pInst->words[pInst->wordCount];
  strncpy(dest, str, length);
  pInst->wordCount += (uint16_t)wordCount;

  return SPV_SUCCESS;
}

// TODO(dneto): This API is not powerful enough in the case that the
// number and type of operands are not known until partway through parsing
// the operation.  This happens when enum operands might have different number
// of operands, or with extended instructions.
spv_operand_type_t spvBinaryOperandInfo(const uint32_t word,
                                        const uint16_t operandIndex,
                                        const spv_opcode_desc opcodeEntry,
                                        const spv_operand_table operandTable,
                                        spv_operand_desc *pOperandEntry) {
  spv_operand_type_t type;
  if (operandIndex < opcodeEntry->numTypes) {
    // NOTE: Do operand table lookup to set operandEntry if successful
    uint16_t index = operandIndex - 1;
    type = opcodeEntry->operandTypes[index];
    spv_operand_desc entry = nullptr;
    if (!spvOperandTableValueLookup(operandTable, type, word, &entry)) {
      if (SPV_OPERAND_TYPE_NONE != entry->operandTypes[0]) {
        *pOperandEntry = entry;
      }
    }
  } else if (*pOperandEntry) {
    // NOTE: Use specified operand entry operand type for this word
    uint16_t index = operandIndex - opcodeEntry->numTypes;
    type = (*pOperandEntry)->operandTypes[index];
  } else if (OpSwitch == opcodeEntry->opcode) {
    // NOTE: OpSwitch is a special case which expects a list of paired extra
    // operands
    assert(0 &&
           "This case is previously untested, remove this assert and ensure it "
           "is behaving correctly!");
    uint16_t lastIndex = opcodeEntry->numTypes - 1;
    uint16_t index = lastIndex + ((operandIndex - lastIndex) % 2);
    type = opcodeEntry->operandTypes[index];
  } else {
    // NOTE: Default to last operand type in opcode entry
    uint16_t index = opcodeEntry->numTypes - 1;
    type = opcodeEntry->operandTypes[index];
  }
  return type;
}

spv_result_t spvBinaryDecodeOperand(
    const Op opcode, const spv_operand_type_t type, const uint32_t *words,
    uint16_t numWords, const spv_endianness_t endian, const uint32_t options,
    const spv_operand_table operandTable, const spv_ext_inst_table extInstTable,
    spv_operand_pattern_t *pExpectedOperands, spv_ext_inst_type_t *pExtInstType,
    out_stream &stream, spv_position position, spv_diagnostic *pDiagnostic) {
  if (!words || !position) return SPV_ERROR_INVALID_POINTER;
  if (!pDiagnostic) return SPV_ERROR_INVALID_DIAGNOSTIC;

  bool print = spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_PRINT, options);
  bool color =
      print && spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_COLOR, options);

  switch (type) {
    case SPV_OPERAND_TYPE_EXECUTION_SCOPE:
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_ID_IN_OPTIONAL_TUPLE:
    case SPV_OPERAND_TYPE_OPTIONAL_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS:
    case SPV_OPERAND_TYPE_RESULT_ID: {
      if (color) {
        if (type == SPV_OPERAND_TYPE_RESULT_ID) {
          stream.get() << clr::blue();
        } else {
          stream.get() << clr::yellow();
        }
      }
      stream.get() << "%" << spvFixWord(words[0], endian);
      stream.get() << ((color) ? clr::reset() : "");
      position->index++;
    } break;
    case SPV_OPERAND_TYPE_LITERAL_NUMBER: {
      // NOTE: Special case for extended instruction use
      if (OpExtInst == opcode) {
        spv_ext_inst_desc extInst;
        if (spvExtInstTableValueLookup(extInstTable, *pExtInstType, words[0],
                                       &extInst)) {
          DIAGNOSTIC << "Invalid extended instruction '" << words[0] << "'.";
          return SPV_ERROR_INVALID_BINARY;
        }
        spvPrependOperandTypes(extInst->operandTypes, pExpectedOperands);
        stream.get() << (color ? clr::red() : "");
        stream.get() << extInst->name;
        stream.get() << (color ? clr::reset() : "");
        position->index++;
        break;
      }
    }  // Fall through for the general case.
    case SPV_OPERAND_TYPE_MULTIWORD_LITERAL_NUMBER:
    case SPV_OPERAND_TYPE_LITERAL:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL:
    case SPV_OPERAND_TYPE_LITERAL_IN_OPTIONAL_TUPLE: {
      // TODO: Need to support multiple word literals
      stream.get() << (color ? clr::red() : "");
      if (numWords > 2) {
        DIAGNOSTIC << "Literal numbers larger than 64-bit not supported yet.";
        return SPV_UNSUPPORTED;
      } else if (numWords == 2) {
        stream.get() << spvFixDoubleWord(words[0], words[1], endian);
        position->index += 2;
      } else {
        stream.get() << spvFixWord(words[0], endian);
        position->index++;
      }
      stream.get() << (color ? clr::reset() : "");
    } break;
    case SPV_OPERAND_TYPE_LITERAL_STRING:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING: {
      const char *string = (const char *)words;
      uint64_t stringOperandCount = (strlen(string) / 4) + 1;

      // NOTE: Special case for extended instruction import
      if (OpExtInstImport == opcode) {
        *pExtInstType = spvExtInstImportTypeGet(string);
        if (SPV_EXT_INST_TYPE_NONE == *pExtInstType) {
          DIAGNOSTIC << "Invalid extended instruction import'" << string
                     << "'.";
          return SPV_ERROR_INVALID_BINARY;
        }
      }

      stream.get() << "\"";
      stream.get() << (color ? clr::green() : "");
      stream.get() << string;
      stream.get() << (color ? clr::reset() : "");
      stream.get() << "\"";
      position->index += stringOperandCount;
    } break;
    case SPV_OPERAND_TYPE_CAPABILITY:
    case SPV_OPERAND_TYPE_SOURCE_LANGUAGE:
    case SPV_OPERAND_TYPE_EXECUTION_MODEL:
    case SPV_OPERAND_TYPE_ADDRESSING_MODEL:
    case SPV_OPERAND_TYPE_MEMORY_MODEL:
    case SPV_OPERAND_TYPE_EXECUTION_MODE:
    case SPV_OPERAND_TYPE_OPTIONAL_EXECUTION_MODE:
    case SPV_OPERAND_TYPE_STORAGE_CLASS:
    case SPV_OPERAND_TYPE_DIMENSIONALITY:
    case SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE:
    case SPV_OPERAND_TYPE_FP_ROUNDING_MODE:
    case SPV_OPERAND_TYPE_LINKAGE_TYPE:
    case SPV_OPERAND_TYPE_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE:
    case SPV_OPERAND_TYPE_DECORATION:
    case SPV_OPERAND_TYPE_BUILT_IN:
    case SPV_OPERAND_TYPE_GROUP_OPERATION:
    case SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS:
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO: {
      spv_operand_desc entry;
      if (spvOperandTableValueLookup(operandTable, type,
                                     spvFixWord(words[0], endian), &entry)) {
        DIAGNOSTIC << "Invalid " << spvOperandTypeStr(type) << " operand '"
                   << words[0] << "'.";
        return SPV_ERROR_INVALID_TEXT; // TODO(dneto): Surely this is invalid binary.
      }
      stream.get() << entry->name;
      // Prepare to accept operands to this operand, if needed.
      spvPrependOperandTypes(entry->operandTypes, pExpectedOperands);
      position->index++;
    } break;
    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_OPTIONAL_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL: {
      // This operand is a mask.
      // Scan it from least significant bit to most significant bit.  For each
      // set bit, emit the name of that bit and prepare to parse its operands,
      // if any.
      uint32_t remaining_word = spvFixWord(words[0], endian);
      uint32_t mask;
      int num_emitted = 0;
      for (mask = 1; remaining_word; mask <<= 1) {
        if (remaining_word & mask) {
          remaining_word ^= mask;
          spv_operand_desc entry;
          if (spvOperandTableValueLookup(operandTable, type, mask, &entry)) {
            DIAGNOSTIC << "Invalid " << spvOperandTypeStr(type) << " operand '"
                       << words[0] << "'.";
            return SPV_ERROR_INVALID_BINARY;
          }
          if (num_emitted) stream.get() << "|";
          stream.get() << entry->name;
          num_emitted++;
        }
      }
      if (!num_emitted) {
        // An operand value of 0 was provided, so represent it by the name
        // of the 0 value. In many cases, that's "None".
        spv_operand_desc entry;
        if (SPV_SUCCESS ==
            spvOperandTableValueLookup(operandTable, type, 0, &entry)) {
          stream.get() << entry->name;
          // Prepare for its operands, if any.
          spvPrependOperandTypes(entry->operandTypes, pExpectedOperands);
        }
      }
      // Prepare for subsequent operands, if any.
      // Scan from MSB to LSB since we can only prepend operands to a pattern.
      remaining_word = spvFixWord(words[0], endian);
      for (mask = (1u << 31); remaining_word; mask >>= 1) {
        if (remaining_word & mask) {
          remaining_word ^= mask;
          spv_operand_desc entry;
          if (SPV_SUCCESS ==
              spvOperandTableValueLookup(operandTable, type, mask, &entry)) {
            spvPrependOperandTypes(entry->operandTypes, pExpectedOperands);
          }
        }
      }
      position->index++;
    } break;
    default: {
      DIAGNOSTIC << "Invalid binary operand '" << type << "'";
      return SPV_ERROR_INVALID_BINARY;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t spvBinaryDecodeOpcode(
    spv_instruction_t *pInst, const spv_endianness_t endian,
    const uint32_t options, const spv_opcode_table opcodeTable,
    const spv_operand_table operandTable, const spv_ext_inst_table extInstTable,
    spv_assembly_syntax_format_t format, out_stream &stream,
    spv_position position, spv_diagnostic *pDiagnostic) {
  if (!pInst || !position) return SPV_ERROR_INVALID_POINTER;
  if (!opcodeTable || !operandTable || !extInstTable)
    return SPV_ERROR_INVALID_TABLE;
  if (!pDiagnostic) return SPV_ERROR_INVALID_DIAGNOSTIC;

  spv_position_t instructionStart = *position;

  uint16_t wordCount;
  Op opcode;
  spvOpcodeSplit(spvFixWord(pInst->words[0], endian), &wordCount, &opcode);

  spv_opcode_desc opcodeEntry;
  if (spvOpcodeTableValueLookup(opcodeTable, opcode, &opcodeEntry)) {
    DIAGNOSTIC << "Invalid Opcode '" << opcode << "'.";
    return SPV_ERROR_INVALID_BINARY;
  }

  // See if there are enough required words.
  // Some operands in the operand types are optional or could be zero length.
  // The optional and zero length opeands must be at the end of the list.
  if (opcodeEntry->numTypes > wordCount &&
      !spvOperandIsOptional(opcodeEntry->operandTypes[wordCount])) {
    uint16_t numRequired;
    for (numRequired = 0;
         numRequired < opcodeEntry->numTypes &&
         !spvOperandIsOptional(opcodeEntry->operandTypes[numRequired]);
         numRequired++)
      ;
    DIAGNOSTIC << "Invalid instruction Op" << opcodeEntry->name
               << " word count '" << wordCount << "', expected at least '"
               << numRequired << "'.";
    return SPV_ERROR_INVALID_BINARY;
  }

  const bool isAssigmentFormat =
      SPV_ASSEMBLY_SYNTAX_FORMAT_ASSIGNMENT == format;

  // For Canonical Assembly Format, all words are written to stream in order.
  // For Assignment Assembly Format, <result-id> and the equal sign are written
  // to stream first, while the rest are written to no_result_id_stream. After
  // processing all words, all words in no_result_id_stream are transcribed to
  // stream.

  std::stringstream no_result_id_strstream;
  out_stream no_result_id_stream(no_result_id_strstream);
  (isAssigmentFormat ? no_result_id_stream.get() : stream.get())
      << "Op" << opcodeEntry->name;

  const int16_t result_id_index = spvOpcodeResultIdIndex(opcodeEntry);
  position->index++;

  // Maintains the ordered list of expected operand types.
  // For many instructions we only need the {numTypes, operandTypes}
  // entries in opcodeEntry.  However, sometimes we need to modify
  // the list as we parse the operands. This occurs when an operand
  // has its own logical operands (such as the LocalSize operand for
  // ExecutionMode), or for extended instructions that may have their
  // own operands depending on the selected extended instruction.
  spv_operand_pattern_t expectedOperands(
      opcodeEntry->operandTypes,
      opcodeEntry->operandTypes + opcodeEntry->numTypes);

  for (uint16_t index = 1; index < wordCount; ++index) {
    const uint64_t currentPosIndex = position->index;
    const bool currentIsResultId = result_id_index == index - 1;

    if (expectedOperands.empty()) {
      DIAGNOSTIC << "Invalid instruction Op" << opcodeEntry->name
                 << " starting at word " << instructionStart.index
                 << ": expected no more operands after " << index
                 << " words, but word count is " << wordCount << ".";
      return SPV_ERROR_INVALID_BINARY;
    }

    spv_operand_type_t type = spvTakeFirstMatchableOperand(&expectedOperands);

    if (isAssigmentFormat) {
      if (!currentIsResultId) no_result_id_stream.get() << " ";
    } else {
      stream.get() << " ";
    }

    uint16_t numWords = 1;
    if (type == SPV_OPERAND_TYPE_MULTIWORD_LITERAL_NUMBER) {
      // Make sure this is the last operand for this instruction.
      if (expectedOperands.empty()) {
        numWords = wordCount - index;
      } else {
        // TODO(antiagainst): This may not be an error. The exact design has not
        // been settled yet.
        DIAGNOSTIC << "Multiple word literal numbers can only appear as the "
                      "last operand of an instruction.";
        return SPV_ERROR_INVALID_BINARY;
      }
    }

    if (spvBinaryDecodeOperand(
            opcodeEntry->opcode, type, pInst->words + index, numWords, endian,
            options, operandTable, extInstTable, &expectedOperands,
            &pInst->extInstType,
            (isAssigmentFormat && !currentIsResultId ? no_result_id_stream
                                                     : stream),
            position, pDiagnostic)) {
      DIAGNOSTIC << "UNEXPLAINED ERROR";
      return SPV_ERROR_INVALID_BINARY;
    }
    if (isAssigmentFormat && currentIsResultId) stream.get() << " = ";
    index += (uint16_t)(position->index - currentPosIndex - 1);
  }
  // TODO(dneto): There's an opportunity for a more informative message.
  if (!expectedOperands.empty() &&
      !spvOperandIsOptional(expectedOperands.front())) {
    DIAGNOSTIC << "Invalid instruction Op" << opcodeEntry->name
               << " starting at word " << instructionStart.index
               << ": expected more operands after " << wordCount << " words.";
    return SPV_ERROR_INVALID_BINARY;
  }

  stream.get() << no_result_id_strstream.str();

  return SPV_SUCCESS;
}

spv_result_t spvBinaryToText(uint32_t *code, const uint64_t wordCount,
                             const uint32_t options,
                             const spv_opcode_table opcodeTable,
                             const spv_operand_table operandTable,
                             const spv_ext_inst_table extInstTable,
                             spv_text *pText, spv_diagnostic *pDiagnostic) {
  return spvBinaryToTextWithFormat(
      code, wordCount, options, opcodeTable, operandTable, extInstTable,
      SPV_ASSEMBLY_SYNTAX_FORMAT_DEFAULT, pText, pDiagnostic);
}

spv_result_t spvBinaryToTextWithFormat(
    uint32_t *code, const uint64_t wordCount, const uint32_t options,
    const spv_opcode_table opcodeTable, const spv_operand_table operandTable,
    const spv_ext_inst_table extInstTable, spv_assembly_syntax_format_t format,
    spv_text *pText, spv_diagnostic *pDiagnostic) {
  spv_binary_t binary = {code, wordCount};

  spv_position_t position = {};
  if (!binary.code || !binary.wordCount) {
    DIAGNOSTIC << "Binary stream is empty.";
    return SPV_ERROR_INVALID_BINARY;
  }
  if (!opcodeTable || !operandTable || !extInstTable)
    return SPV_ERROR_INVALID_TABLE;
  if (pText && spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_PRINT, options))
    return SPV_ERROR_INVALID_POINTER;
  if (!pText && !spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_PRINT, options))
    return SPV_ERROR_INVALID_POINTER;
  if (!pDiagnostic) return SPV_ERROR_INVALID_DIAGNOSTIC;

  spv_endianness_t endian;
  if (spvBinaryEndianness(&binary, &endian)) {
    DIAGNOSTIC << "Invalid SPIR-V magic number '" << std::hex << binary.code[0]
               << "'.";
    return SPV_ERROR_INVALID_BINARY;
  }

  spv_header_t header;
  if (spvBinaryHeaderGet(&binary, endian, &header)) {
    DIAGNOSTIC << "Invalid SPIR-V header.";
    return SPV_ERROR_INVALID_BINARY;
  }

  bool print = spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_PRINT, options);
  bool color =
      print && spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_COLOR, options);

  std::stringstream sstream;
  out_stream stream(sstream);
  if (print) {
    stream = out_stream();
  }

  if (color) {
    stream.get() << clr::grey();
  }
  stream.get() << "; SPIR-V\n"
               << "; Version: " << header.version << "\n"
               << "; Generator: " << spvGeneratorStr(header.generator) << "\n"
               << "; Bound: " << header.bound << "\n"
               << "; Schema: " << header.schema << "\n";
  if (color) {
    stream.get() << clr::reset();
  }

  const uint32_t *words = binary.code;
  position.index = SPV_INDEX_INSTRUCTION;
  spv_ext_inst_type_t extInstType = SPV_EXT_INST_TYPE_NONE;
  while (position.index < binary.wordCount) {
    uint64_t index = position.index;
    uint16_t wordCount;
    Op opcode;
    spvOpcodeSplit(spvFixWord(words[position.index], endian), &wordCount,
                   &opcode);

    spv_instruction_t inst = {};
    inst.extInstType = extInstType;
    spvInstructionCopy(&words[position.index], opcode, wordCount, endian,
                       &inst);

    if (spvBinaryDecodeOpcode(&inst, endian, options, opcodeTable, operandTable,
                              extInstTable, format, stream, &position,
                              pDiagnostic))
      return SPV_ERROR_INVALID_BINARY;
    extInstType = inst.extInstType;

    if ((index + wordCount) != position.index) {
      DIAGNOSTIC << "Invalid word count.";
      return SPV_ERROR_INVALID_BINARY;
    }

    stream.get() << "\n";
  }

  if (!print) {
    size_t length = sstream.str().size();
    char *str = new char[length + 1];
    if (!str) return SPV_ERROR_OUT_OF_MEMORY;
    strncpy(str, sstream.str().c_str(), length + 1);
    spv_text text = new spv_text_t();
    if (!text) return SPV_ERROR_OUT_OF_MEMORY;
    text->str = str;
    text->length = length;
    *pText = text;
  }

  return SPV_SUCCESS;
}

void spvBinaryDestroy(spv_binary binary) {
  if (!binary) return;
  if (binary->code) {
    delete[] binary->code;
  }
  delete binary;
}
