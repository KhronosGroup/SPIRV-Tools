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
#include "text.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>
#include <unordered_map>

// Structures

struct spv_named_id_table_t {
  std::unordered_map<std::string, uint32_t> namedIds;
};

// Text API

std::string getWord(const char *str) {
  size_t index = 0;
  while (true) {
    switch (str[index]) {
      case '\0':
      case '\t':
      case '\n':
      case ' ':
        break;
      default:
        index++;
    }
  }
  return std::string(str, str + index);
}

spv_named_id_table spvNamedIdTableCreate() {
  return new spv_named_id_table_t();
}

void spvNamedIdTableDestory(spv_named_id_table table) { delete table; }

uint32_t spvNamedIdAssignOrGet(spv_named_id_table table, const char *textValue,
                               uint32_t *pBound) {
  if (table->namedIds.end() == table->namedIds.find(textValue)) {
    table->namedIds[textValue] = *pBound;
  }
  return table->namedIds[textValue];
}

int32_t spvTextIsNamedId(const char *textValue) {
  // TODO: Strengthen the parsing of textValue to only include allow names that
  // match: ([a-z]|[A-Z])(_|[a-z]|[A-Z]|[0-9])*
  switch (textValue[0]) {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return false;
    default:
      break;
  }
  return true;
}

spv_result_t spvTextAdvanceLine(const spv_text text, spv_position position) {
  while (true) {
    switch (text->str[position->index]) {
      case '\0':
        return SPV_END_OF_STREAM;
      case '\n':
        position->column = 0;
        position->line++;
        position->index++;
        return SPV_SUCCESS;
      default:
        position->column++;
        position->index++;
        break;
    }
  }
}

spv_result_t spvTextAdvance(const spv_text text, spv_position position) {
  // NOTE: Consume white space, otherwise don't advance.
  switch (text->str[position->index]) {
    case '\0':
      return SPV_END_OF_STREAM;
    case ';':
      if (spv_result_t error = spvTextAdvanceLine(text, position)) return error;
      return spvTextAdvance(text, position);
    case ' ':
    case '\t':
      position->column++;
      position->index++;
      return spvTextAdvance(text, position);
    case '\n':
      position->column = 0;
      position->line++;
      position->index++;
      return spvTextAdvance(text, position);
    default:
      break;
  }

  return SPV_SUCCESS;
}

spv_result_t spvTextWordGet(const spv_text text,
                            const spv_position startPosition, std::string &word,
                            spv_position endPosition) {
  spvCheck(!text->str || !text->length, return SPV_ERROR_INVALID_TEXT);
  spvCheck(!startPosition || !endPosition, return SPV_ERROR_INVALID_POINTER);

  *endPosition = *startPosition;

  // NOTE: Assumes first character is not white space!
  while (true) {
    switch (text->str[endPosition->index]) {
      case ' ':
      case '\t':
      case '\n':
      case '\0': {  // NOTE: End of word found!
        word.assign(text->str + startPosition->index,
                    (size_t)(endPosition->index - startPosition->index));
        return SPV_SUCCESS;
      }
      default:
        break;
    }

    endPosition->column++;
    endPosition->index++;
  }
}

// Returns true if the string at the given position in text starts with "Op".
static bool spvStartsWithOp(const spv_text text, const spv_position position) {
  if (text->length < position->index + 2) return false;
  return ('O' == text->str[position->index] &&
          'p' == text->str[position->index + 1]);
}

// Returns true if a new instruction begins at the given position in text.
static bool spvTextIsStartOfNewInst(const spv_text text,
                                    const spv_position position) {
  spv_position_t nextPosition = *position;
  if (spvTextAdvance(text, &nextPosition)) return false;
  if (spvStartsWithOp(text, position)) return true;

  std::string word;
  spv_position_t startPosition = nextPosition;
  if (spvTextWordGet(text, &startPosition, word, &nextPosition)) return false;
  if ('%' != word.front()) return false;

  if (spvTextAdvance(text, &nextPosition)) return false;
  startPosition = nextPosition;
  if (spvTextWordGet(text, &startPosition, word, &nextPosition)) return false;
  if ("=" != word) return false;

  if (spvTextAdvance(text, &nextPosition)) return false;
  startPosition = nextPosition;
  if (spvStartsWithOp(text, &startPosition)) return true;
  return false;
}

spv_result_t spvTextStringGet(const spv_text text,
                              const spv_position startPosition,
                              std::string &string, spv_position endPosition) {
  spvCheck(!text->str || !text->length, return SPV_ERROR_INVALID_TEXT);
  spvCheck(!startPosition || !endPosition, return SPV_ERROR_INVALID_POINTER);

  spvCheck('"' != text->str[startPosition->index],
           return SPV_ERROR_INVALID_TEXT);

  *endPosition = *startPosition;

  // NOTE: Assumes first character is not white space
  while (true) {
    endPosition->column++;
    endPosition->index++;

    switch (text->str[endPosition->index]) {
      case '"': {
        endPosition->column++;
        endPosition->index++;

        string.assign(text->str + startPosition->index,
                      (size_t)(endPosition->index - startPosition->index));

        return SPV_SUCCESS;
      }
      case '\n':
      case '\0':
        return SPV_ERROR_INVALID_TEXT;
      default:
        break;
    }
  }
}

spv_result_t spvTextToUInt32(const char *textValue, uint32_t *pValue) {
  char *endPtr = nullptr;
  *pValue = strtoul(textValue, &endPtr, 0);
  if (0 == *pValue && textValue == endPtr) {
    return SPV_ERROR_INVALID_TEXT;
  }
  return SPV_SUCCESS;
}

spv_result_t spvTextToLiteral(const char *textValue, spv_literal_t *pLiteral) {
  bool isSigned = false;
  bool isFloat = false;
  bool isString = false;

  if ('-' == textValue[0]) {
    isSigned = true;
  }

  for (uint64_t index = 0; index < strlen(textValue); ++index) {
    switch (textValue[index]) {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        break;
      case '.':
        isFloat = true;
        break;
      default:
        isString = true;
        break;
    }
  }

  if (isString) {
    pLiteral->type = SPV_LITERAL_TYPE_STRING;
    strncpy(pLiteral->value.str, textValue, strlen(textValue));
  } else if (isFloat) {
    double d = strtod(textValue, nullptr);
    float f = (float)d;
    if (d == (double)f) {
      pLiteral->type = SPV_LITERAL_TYPE_FLOAT_32;
      pLiteral->value.f = f;
    } else {
      pLiteral->type = SPV_LITERAL_TYPE_FLOAT_64;
      pLiteral->value.d = d;
    }
  } else if (isSigned) {
    int64_t i64 = strtoll(textValue, nullptr, 10);
    int32_t i32 = (int32_t)i64;
    if (i64 == (int64_t)i32) {
      pLiteral->type = SPV_LITERAL_TYPE_INT_32;
      pLiteral->value.i32 = i32;
    } else {
      pLiteral->type = SPV_LITERAL_TYPE_INT_64;
      pLiteral->value.i64 = i64;
    }
  } else {
    uint64_t u64 = strtoull(textValue, nullptr, 10);
    uint32_t u32 = (uint32_t)u64;
    if (u64 == (uint64_t)u32) {
      pLiteral->type = SPV_LITERAL_TYPE_UINT_32;
      pLiteral->value.u32 = u32;
    } else {
      pLiteral->type = SPV_LITERAL_TYPE_UINT_64;
      pLiteral->value.u64 = u64;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t spvTextEncodeOperand(
    const spv_operand_type_t type, const char *textValue,
    const spv_operand_table operandTable, const spv_ext_inst_table extInstTable,
    spv_named_id_table namedIdTable, spv_instruction_t *pInst,
    const spv_operand_type_t **ppExtraOperands, uint32_t *pBound,
    const spv_position position, spv_diagnostic *pDiagnostic) {
  // NOTE: Handle immediate int in the stream
  if ('!' == textValue[0]) {
    const char *begin = textValue + 1;
    char *end = nullptr;
    uint32_t immediateInt = strtoul(begin, &end, 0);
    size_t size = strlen(textValue);
    size_t length = (end - begin);
    spvCheck(size - 1 != length, DIAGNOSTIC << "Invalid immediate integer '"
                                            << textValue << "'.";
             return SPV_ERROR_INVALID_TEXT);
    position->column += size;
    position->index += size;
    pInst->words[pInst->wordCount] = immediateInt;
    pInst->wordCount += 1;
    return SPV_SUCCESS;
  }

  switch (type) {
    case SPV_OPERAND_TYPE_ID: {
      if ('%' == textValue[0]) {
        textValue++;
      }
      // TODO: Force all ID's to be prefixed with '%'.
      uint32_t id = 0;
      if (spvTextIsNamedId(textValue)) {
        id = spvNamedIdAssignOrGet(namedIdTable, textValue, pBound);
      } else {
        spvCheck(spvTextToUInt32(textValue, &id),
                 DIAGNOSTIC << "Invalid ID '" << textValue << "'.";
                 return SPV_ERROR_INVALID_TEXT);
      }
      pInst->words[pInst->wordCount++] = id;
      if (*pBound <= id) {
        *pBound = id + 1;
      }
    } break;
    case SPV_OPERAND_TYPE_RESULT_ID: {
      if ('%' == textValue[0]) {
        textValue++;
      }
      // TODO: Force all Result ID's to be prefixed with '%'.
      uint32_t id = 0;
      if (spvTextIsNamedId(textValue)) {
        id = spvNamedIdAssignOrGet(namedIdTable, textValue, pBound);
      } else {
        spvCheck(spvTextToUInt32(textValue, &id),
                 DIAGNOSTIC << "Invalid result ID '" << textValue << "'.";
                 return SPV_ERROR_INVALID_TEXT);
      }
      pInst->words[pInst->wordCount++] = id;
      if (*pBound <= id) {
        *pBound = id + 1;
      }
    } break;
    case SPV_OPERAND_TYPE_LITERAL_NUMBER: {
      // NOTE: Special case for extension instruction lookup
      if (OpExtInst == pInst->opcode) {
        spv_ext_inst_desc extInst;
        spvCheck(spvExtInstTableNameLookup(extInstTable, pInst->extInstType,
                                           textValue, &extInst),
                 DIAGNOSTIC << "Invalid extended instruction name '"
                            << textValue << "'.";
                 return SPV_ERROR_INVALID_TEXT);
        pInst->words[pInst->wordCount++] = extInst->ext_inst;
        *ppExtraOperands = extInst->operandTypes;
        return SPV_SUCCESS;
      }

      // TODO: Literal numbers can be any number up to 64 bits wide. This
      // includes integers and floating point numbers.
      spvCheck(spvTextToUInt32(textValue, &pInst->words[pInst->wordCount++]),
               DIAGNOSTIC << "Invalid literal number '" << textValue << "'.";
               return SPV_ERROR_INVALID_TEXT);
    } break;
    case SPV_OPERAND_TYPE_LITERAL: {
      spv_literal_t literal = {};
      spvCheck(spvTextToLiteral(textValue, &literal),
               DIAGNOSTIC << "Invalid literal '" << textValue << "'.";
               return SPV_ERROR_INVALID_TEXT);
      switch (literal.type) {
        case SPV_LITERAL_TYPE_INT_32:
          spvCheck(spvBinaryEncodeU32((uint32_t)literal.value.i32, pInst,
                                      position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
          break;
        case SPV_LITERAL_TYPE_INT_64: {
          spvCheck(spvBinaryEncodeU64((uint64_t)literal.value.i64, pInst,
                                      position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        case SPV_LITERAL_TYPE_UINT_32: {
          spvCheck(spvBinaryEncodeU32(literal.value.u32, pInst, position,
                                      pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        case SPV_LITERAL_TYPE_UINT_64: {
          spvCheck(spvBinaryEncodeU64((uint64_t)literal.value.u64, pInst,
                                      position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        case SPV_LITERAL_TYPE_FLOAT_32: {
          spvCheck(spvBinaryEncodeU32((uint32_t)literal.value.f, pInst,
                                      position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        case SPV_LITERAL_TYPE_FLOAT_64: {
          spvCheck(spvBinaryEncodeU64((uint64_t)literal.value.d, pInst,
                                      position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        case SPV_LITERAL_TYPE_STRING: {
          spvCheck(spvBinaryEncodeString(literal.value.str, pInst, position,
                                         pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        default:
          DIAGNOSTIC << "Invalid literal '" << textValue << "'";
          return SPV_ERROR_INVALID_TEXT;
      }
    } break;
    case SPV_OPERAND_TYPE_LITERAL_STRING: {
      size_t len = strlen(textValue);
      spvCheck('"' != textValue[0] && '"' != textValue[len - 1],
               DIAGNOSTIC << "Invalid literal string '" << textValue
                          << "', expected quotes.";
               return SPV_ERROR_INVALID_TEXT);
      // NOTE: Strip quotes
      std::string text(textValue + 1, len - 2);

      // NOTE: Special case for extended instruction library import
      if (OpExtInstImport == pInst->opcode) {
        pInst->extInstType = spvExtInstImportTypeGet(text.c_str());
      }

      spvCheck(
          spvBinaryEncodeString(text.c_str(), pInst, position, pDiagnostic),
          return SPV_ERROR_INVALID_TEXT);
    } break;
    default: {
      // NOTE: All non literal operands are handled here using the operand
      // table.
      spv_operand_desc entry;
      spvCheck(spvOperandTableNameLookup(operandTable, type, textValue, &entry),
               DIAGNOSTIC << "Invalid " << spvOperandTypeStr(type) << " '"
                          << textValue << "'.";
               return SPV_ERROR_INVALID_TEXT;);
      spvCheck(spvBinaryEncodeU32(entry->value, pInst, position, pDiagnostic),
               DIAGNOSTIC << "Invalid " << spvOperandTypeStr(type) << " '"
                          << textValue << "'.";
               return SPV_ERROR_INVALID_TEXT;);
      if (ppExtraOperands && entry->operandTypes[0] != SPV_OPERAND_TYPE_NONE) {
        *ppExtraOperands = entry->operandTypes;
      }
    } break;
  }
  return SPV_SUCCESS;
}

spv_result_t spvTextEncodeOpcode(
    const spv_text text, const spv_opcode_table opcodeTable,
    const spv_operand_table operandTable, const spv_ext_inst_table extInstTable,
    spv_named_id_table namedIdTable, uint32_t *pBound, spv_instruction_t *pInst,
    spv_position position, spv_diagnostic *pDiagnostic) {
  // An assembly instruction has two possible formats:
  // 1. <opcode> <operand>.., e.g., "OpMemoryModel Physical64 OpenCL".
  // 2. <result-id> = <opcode> <operand>.., e.g., "%void = OpTypeVoid".

  // Assume it's the first format at the beginning.
  std::string opcodeName;
  spv_position_t nextPosition = {};
  spv_result_t error =
      spvTextWordGet(text, position, opcodeName, &nextPosition);
  spvCheck(error, return error);

  // NOTE: Handle insertion of an immediate integer into the binary stream
  bool immediate = false;
  spvCheck('!' == text->str[position->index], immediate = true);
  if (immediate) {
    const char *begin = opcodeName.data() + 1;
    char *end = nullptr;
    uint32_t immediateInt = strtoul(begin, &end, 0);
    size_t size = opcodeName.size() - 1;
    spvCheck(size != (size_t)(end - begin),
             DIAGNOSTIC << "Invalid immediate integer '" << opcodeName << "'.";
             return SPV_ERROR_INVALID_TEXT);
    position->column += opcodeName.size();
    position->index += opcodeName.size();
    pInst->words[0] = immediateInt;
    pInst->wordCount = 1;
    return SPV_SUCCESS;
  }

  // Handle value generating instructions (the second format above) here.
  std::string result_id;
  spv_position_t result_id_position = {};
  // If the word we get doesn't start with "Op", assume it's an <result-id>
  // from now.
  spvCheck(!spvStartsWithOp(text, position), result_id = opcodeName);
  if (!result_id.empty()) {
    spvCheck('%' != result_id.front(),
             DIAGNOSTIC << "Expected <opcode> or <result-id> at the beginning "
                           "of an instruction, found '"
                        << result_id << "'.";
             return SPV_ERROR_INVALID_TEXT);
    result_id_position = *position;
    *position = nextPosition;
    spvCheck(spvTextAdvance(text, position),
             DIAGNOSTIC << "Expected '=', found end of stream.";
             return SPV_ERROR_INVALID_TEXT);
    // The '=' sign.
    std::string equal_sign;
    error = spvTextWordGet(text, position, equal_sign, &nextPosition);
    spvCheck("=" != equal_sign, DIAGNOSTIC << "'=' expected after result id.";
             return SPV_ERROR_INVALID_TEXT);

    // The <opcode> after the '=' sign.
    *position = nextPosition;
    spvCheck(spvTextAdvance(text, position),
             DIAGNOSTIC << "Expected opcode, found end of stream.";
             return SPV_ERROR_INVALID_TEXT);
    error = spvTextWordGet(text, position, opcodeName, &nextPosition);
    spvCheck(error, return error);
    spvCheck(!spvStartsWithOp(text, position),
             DIAGNOSTIC << "Invalid Opcode prefix '" << opcodeName << "'.";
             return SPV_ERROR_INVALID_TEXT);
  }

  // NOTE: The table contains Opcode names without the "Op" prefix.
  const char *pInstName = opcodeName.data() + 2;

  spv_opcode_desc opcodeEntry;
  error = spvOpcodeTableNameLookup(opcodeTable, pInstName, &opcodeEntry);
  spvCheck(error, DIAGNOSTIC << "Invalid Opcode name '"
                             << getWord(text->str + position->index) << "'";
           return error);
  pInst->opcode = opcodeEntry->opcode;
  *position = nextPosition;
  pInst->wordCount++;

  // Get the arugment index for <result-id>. Used for handling the <result-id>
  // for value generating instructions below.
  const int16_t result_id_index = spvOpcodeResultIdIndex(opcodeEntry);

  // NOTE: Process the fixed size operands
  const spv_operand_type_t *extraOperandTypes = nullptr;
  for (int32_t operandIndex = 0; operandIndex < (opcodeEntry->wordCount - 1);
       ++operandIndex) {
    if (operandIndex == result_id_index && !result_id.empty()) {
      // Handling the <result-id> for value generating instructions.
      error = spvTextEncodeOperand(
          SPV_OPERAND_TYPE_RESULT_ID, result_id.c_str(), operandTable,
          extInstTable, namedIdTable, pInst, &extraOperandTypes, pBound,
          &result_id_position, pDiagnostic);
      spvCheck(error, return error);
      continue;
    }
    spvCheck(spvTextAdvance(text, position),
             DIAGNOSTIC << "Expected operand, found end of stream.";
             return SPV_ERROR_INVALID_TEXT);

    std::string operandValue;
    error = spvTextWordGet(text, position, operandValue, &nextPosition);
    spvCheck(error, return error);

    error = spvTextEncodeOperand(
        opcodeEntry->operandTypes[operandIndex], operandValue.c_str(),
        operandTable, extInstTable, namedIdTable, pInst, &extraOperandTypes,
        pBound, position, pDiagnostic);
    spvCheck(error, return error);

    *position = nextPosition;
  }

  if (spvOpcodeIsVariable(opcodeEntry)) {
    if (!extraOperandTypes) {
      // NOTE: Handle variable length not defined by an immediate previously
      // encountered in the Opcode.
      spv_operand_type_t type =
          opcodeEntry->operandTypes[opcodeEntry->wordCount - 1];

      while (!spvTextAdvance(text, position)) {
        // NOTE: If this is the end of the current instruction stream and we
        // break out of this loop.
        if (spvTextIsStartOfNewInst(text, position)) break;

        std::string textValue;
        spvTextWordGet(text, position, textValue, &nextPosition);

        if (SPV_OPERAND_TYPE_LITERAL_STRING == type) {
          spvCheck(spvTextAdvance(text, position),
                   DIAGNOSTIC << "Invalid string, found end of stream.";
                   return SPV_ERROR_INVALID_TEXT);

          std::string string;
          spvCheck(spvTextStringGet(text, position, string, &nextPosition),
                   DIAGNOSTIC << "Invalid string, new line or end of stream.";
                   return SPV_ERROR_INVALID_TEXT);
          spvCheck(spvTextEncodeOperand(type, string.c_str(), operandTable,
                                        extInstTable, namedIdTable, pInst,
                                        nullptr, pBound, position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } else {
          spvCheck(spvTextEncodeOperand(type, textValue.c_str(), operandTable,
                                        extInstTable, namedIdTable, pInst,
                                        nullptr, pBound, position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        }
        *position = nextPosition;
      }
    } else {
      // NOTE: Process the variable size operands defined by an immediate
      // previously encountered in the Opcode.
      uint64_t extraOperandsIndex = 0;
      while (extraOperandTypes[extraOperandsIndex]) {
        spvCheck(spvTextAdvance(text, position),
                 DIAGNOSTIC << "Expected operand, found end of stream.";
                 return SPV_ERROR_INVALID_TEXT);

        std::string operandValue;
        error = spvTextWordGet(text, position, operandValue, &nextPosition);

        error = spvTextEncodeOperand(extraOperandTypes[extraOperandsIndex],
                                     operandValue.c_str(), operandTable,
                                     extInstTable, namedIdTable, pInst, nullptr,
                                     pBound, position, pDiagnostic);
        spvCheck(error, return error);

        *position = nextPosition;

        extraOperandsIndex++;
      }
    }
  }

  pInst->words[0] = spvOpcodeMake(pInst->wordCount, opcodeEntry->opcode);

  return SPV_SUCCESS;
}

spv_result_t spvTextToBinary(const spv_text text,
                             const spv_opcode_table opcodeTable,
                             const spv_operand_table operandTable,
                             const spv_ext_inst_table extInstTable,
                             spv_binary *pBinary, spv_diagnostic *pDiagnostic) {
  spv_position_t position = {};
  spvCheck(!text->str || !text->length, DIAGNOSTIC << "Text stream is empty.";
           return SPV_ERROR_INVALID_TEXT);
  spvCheck(!opcodeTable || !operandTable || !extInstTable,
           return SPV_ERROR_INVALID_TABLE);
  spvCheck(!pBinary, return SPV_ERROR_INVALID_POINTER);
  spvCheck(!pDiagnostic, return SPV_ERROR_INVALID_DIAGNOSTIC);

  // NOTE: Ensure diagnostic is zero initialised
  *pDiagnostic = {};

  uint32_t bound = 1;

  std::vector<spv_instruction_t> instructions;

  spvCheck(spvTextAdvance(text, &position), DIAGNOSTIC
                                                << "Text stream is empty.";
           return SPV_ERROR_INVALID_TEXT);

  spv_named_id_table namedIdTable = spvNamedIdTableCreate();
  spvCheck(!namedIdTable, return SPV_ERROR_OUT_OF_MEMORY);

  spv_ext_inst_type_t extInstType = SPV_EXT_INST_TYPE_NONE;
  while (text->length > position.index) {
    spv_instruction_t inst = {};
    inst.extInstType = extInstType;

    spvCheck(spvTextEncodeOpcode(text, opcodeTable, operandTable, extInstTable,
                                 namedIdTable, &bound, &inst, &position,
                                 pDiagnostic),
             spvNamedIdTableDestory(namedIdTable);
             return SPV_ERROR_INVALID_TEXT);
    extInstType = inst.extInstType;

    instructions.push_back(inst);

    spvCheck(spvTextAdvance(text, &position), break);
  }

  spvNamedIdTableDestory(namedIdTable);

  size_t totalSize = SPV_INDEX_INSTRUCTION;
  for (auto &inst : instructions) {
    totalSize += inst.wordCount;
  }

  uint32_t *data = new uint32_t[totalSize];
  spvCheck(!data, return SPV_ERROR_OUT_OF_MEMORY);
  uint64_t currentIndex = SPV_INDEX_INSTRUCTION;
  for (auto &inst : instructions) {
    memcpy(data + currentIndex, inst.words, sizeof(uint32_t) * inst.wordCount);
    currentIndex += inst.wordCount;
  }

  spv_binary binary = new spv_binary_t();
  spvCheck(!binary, delete[] data; return SPV_ERROR_OUT_OF_MEMORY);
  binary->code = data;
  binary->wordCount = totalSize;

  spv_result_t error = spvBinaryHeaderSet(binary, bound);
  spvCheck(error, spvBinaryDestroy(binary); return error);

  *pBinary = binary;

  return SPV_SUCCESS;
}

void spvTextDestroy(spv_text text) {
  spvCheck(!text, return );
  if (text->str) {
    delete[] text->str;
  }
  delete text;
}
