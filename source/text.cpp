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
#include <utils/bitwisecast.h>

#include "binary.h"
#include "diagnostic.h"
#include "ext_inst.h"
#include "opcode.h"
#include "operand.h"
#include "text.h"

#include <assert.h>
#include <stdio.h>
#include <cstdlib>
#include <string.h>

#include <string>
#include <vector>
#include <unordered_map>

using spvutils::BitwiseCast;

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

  bool quoting = false;
  bool escaping = false;

  // NOTE: Assumes first character is not white space!
  while (true) {
    const char ch = text->str[endPosition->index];
    if (ch == '\\')
      escaping = !escaping;
    else {
      switch (ch) {
        case '"':
          if (!escaping) quoting = !quoting;
          break;
        case ' ':
        case ';':
        case '\t':
        case '\n':
          if (escaping || quoting) break;
          // Fall through.
        case '\0': {  // NOTE: End of word found!
          word.assign(text->str + startPosition->index,
                      (size_t)(endPosition->index - startPosition->index));
          return SPV_SUCCESS;
        }
        default:
          break;
      }
      escaping = false;
    }

    endPosition->column++;
    endPosition->index++;
  }
}

namespace {

// Returns true if the string at the given position in text starts with "Op".
bool spvStartsWithOp(const spv_text text, const spv_position position) {
  if (text->length < position->index + 3) return false;
  char ch0 = text->str[position->index];
  char ch1 = text->str[position->index + 1];
  char ch2 = text->str[position->index + 2];
  return ('O' == ch0 && 'p' == ch1 && ('A' <= ch2 && ch2 <= 'Z'));
}

} // anonymous namespace

// Returns true if a new instruction begins at the given position in text.
bool spvTextIsStartOfNewInst(const spv_text text,
                             const spv_position position) {
  spv_position_t nextPosition = *position;
  if (spvTextAdvance(text, &nextPosition)) return false;
  if (spvStartsWithOp(text, &nextPosition)) return true;

  std::string word;
  spv_position_t startPosition = *position;
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
  int numPeriods = 0;
  bool isString = false;

  const size_t len = strlen(textValue);
  if (len == 0) return SPV_FAILED_MATCH;

  for (uint64_t index = 0; index < len; ++index) {
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
        numPeriods++;
        break;
      case '-':
        if (index == 0) {
          isSigned = true;
        } else {
          isString = true;
        }
        break;
      default:
        isString = true;
        index = len; // break out of the loop too.
        break;
    }
  }

  pLiteral->type = spv_literal_type_t(99);

  if (isString || numPeriods > 1 || (isSigned && len==1)) {
    // TODO(dneto): Allow escaping.
    if (len < 2 || textValue[0] != '"' || textValue[len - 1] != '"')
      return SPV_FAILED_MATCH;
    pLiteral->type = SPV_LITERAL_TYPE_STRING;
    // Need room for the null-terminator.
    if (len >= sizeof(pLiteral->value.str)) return SPV_ERROR_OUT_OF_MEMORY;
    strncpy(pLiteral->value.str, textValue+1, len-2);
    pLiteral->value.str[len-2] = 0;
  } else if (numPeriods == 1) {
    double d = std::strtod(textValue, nullptr);
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
    spv_operand_pattern_t* pExpectedOperands, uint32_t *pBound,
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
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_ID_IN_OPTIONAL_TUPLE:
    case SPV_OPERAND_TYPE_OPTIONAL_ID:
    case SPV_OPERAND_TYPE_RESULT_ID: {
      if ('%' == textValue[0]) {
        textValue++;
      }
      // TODO: Force all ID's to be prefixed with '%'.
      uint32_t id = 0;
      if (spvTextIsNamedId(textValue)) {
        id = spvNamedIdAssignOrGet(namedIdTable, textValue, pBound);
      } else {
        spvCheck(spvTextToUInt32(textValue, &id),
                 DIAGNOSTIC
                     << "Invalid "
                     << ((type == SPV_OPERAND_TYPE_RESULT_ID) ? "result " : "")
                     << "ID '" << textValue << "'.";
                 return (spvOperandIsOptional(type) ? SPV_FAILED_MATCH
                                                    : SPV_ERROR_INVALID_TEXT));
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

        // Prepare to parse the operands for the extended instructions.
        spvPrependOperandTypes(extInst->operandTypes, pExpectedOperands);

        return SPV_SUCCESS;
      }

      // TODO: Literal numbers can be any number up to 64 bits wide. This
      // includes integers and floating point numbers.
      // TODO(dneto): Suggest using spvTextToLiteral and looking for an
      // appropriate result type.
      spvCheck(spvTextToUInt32(textValue, &pInst->words[pInst->wordCount++]),
               DIAGNOSTIC << "Invalid literal number '" << textValue << "'.";
               return SPV_ERROR_INVALID_TEXT);
    } break;
    case SPV_OPERAND_TYPE_LITERAL:
    case SPV_OPERAND_TYPE_LITERAL_IN_OPTIONAL_TUPLE:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL: {
      spv_literal_t literal = {};
      // TODO(dneto): Is return code different for optional operands?
      spvCheck(spvTextToLiteral(textValue, &literal),
               DIAGNOSTIC << "Invalid literal '" << textValue << "'.";
               return SPV_ERROR_INVALID_TEXT);
      switch (literal.type) {
        // We do not have to print diagnostics here because spvBinaryEncode*
        // prints diagnostic messages on failure.
        case SPV_LITERAL_TYPE_INT_32:
          spvCheck(spvBinaryEncodeU32(BitwiseCast<uint32_t>(literal.value.i32),
                                      pInst, position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
          break;
        case SPV_LITERAL_TYPE_INT_64: {
          spvCheck(spvBinaryEncodeU64(BitwiseCast<uint64_t>(literal.value.i64),
                                      pInst, position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        case SPV_LITERAL_TYPE_UINT_32: {
          spvCheck(spvBinaryEncodeU32(literal.value.u32, pInst, position,
                                      pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        case SPV_LITERAL_TYPE_UINT_64: {
          spvCheck(spvBinaryEncodeU64(BitwiseCast<uint64_t>(literal.value.u64),
                                      pInst, position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        case SPV_LITERAL_TYPE_FLOAT_32: {
          spvCheck(spvBinaryEncodeU32(BitwiseCast<uint32_t>(literal.value.f),
                                      pInst, position, pDiagnostic),
                   return SPV_ERROR_INVALID_TEXT);
        } break;
        case SPV_LITERAL_TYPE_FLOAT_64: {
          spvCheck(spvBinaryEncodeU64(BitwiseCast<uint64_t>(literal.value.d),
                                      pInst, position, pDiagnostic),
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
    case SPV_OPERAND_TYPE_LITERAL_STRING:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING: {
      size_t len = strlen(textValue);
      spvCheck('"' != textValue[0] && '"' != textValue[len - 1],
               if (spvOperandIsOptional(type))
                 return SPV_FAILED_MATCH;
               DIAGNOSTIC << "Invalid literal string '" << textValue
                          << "', expected quotes.";
               return SPV_ERROR_INVALID_TEXT;);
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
    case SPV_OPERAND_TYPE_OPTIONAL_IMAGE:
      assert(0 && " Handle optional optional image operands");
      break;
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

      // Prepare to parse the operands for this logical operand.
      spvPrependOperandTypes(entry->operandTypes, pExpectedOperands);
    } break;
  }
  return SPV_SUCCESS;
}

spv_result_t spvTextEncodeOpcode(
    const spv_text text, spv_assembly_syntax_format_t format,
    const spv_opcode_table opcodeTable, const spv_operand_table operandTable,
    const spv_ext_inst_table extInstTable, spv_named_id_table namedIdTable,
    uint32_t *pBound, spv_instruction_t *pInst, spv_position position,
    spv_diagnostic *pDiagnostic) {
  // An assembly instruction has two possible formats:
  // 1(CAF): <opcode> <operand>..., e.g., "OpTypeVoid %void".
  // 2(AAF): <result-id> = <opcode> <operand>..., e.g., "%void = OpTypeVoid".

  std::string firstWord;
  spv_position_t nextPosition = {};
  spv_result_t error = spvTextWordGet(text, position, firstWord, &nextPosition);
  spvCheck(error, DIAGNOSTIC << "Internal Error"; return error);

  // NOTE: Handle insertion of an immediate integer into the binary stream
  if ('!' == text->str[position->index]) {
    const char *begin = firstWord.data() + 1;
    char *end = nullptr;
    uint32_t immediateInt = strtoul(begin, &end, 0);
    size_t size = firstWord.size() - 1;
    spvCheck(size != (size_t)(end - begin),
             DIAGNOSTIC << "Invalid immediate integer '" << firstWord << "'.";
             return SPV_ERROR_INVALID_TEXT);
    position->column += firstWord.size();
    position->index += firstWord.size();
    pInst->words[0] = immediateInt;
    pInst->wordCount = 1;
    return SPV_SUCCESS;
  }

  std::string opcodeName;
  std::string result_id;
  spv_position_t result_id_position = {};
  if (spvStartsWithOp(text, position)) {
    opcodeName = firstWord;
  } else {
    // If the first word of this instruction is not an opcode, we must be
    // processing AAF now.
    spvCheck(
        SPV_ASSEMBLY_SYNTAX_FORMAT_ASSIGNMENT != format,
        DIAGNOSTIC
            << "Expected <opcode> at the beginning of an instruction, found '"
            << firstWord << "'.";
        return SPV_ERROR_INVALID_TEXT);

    result_id = firstWord;
    spvCheck('%' != result_id.front(),
             DIAGNOSTIC << "Expected <opcode> or <result-id> at the beginning "
                           "of an instruction, found '"
                        << result_id << "'.";
             return SPV_ERROR_INVALID_TEXT);
    result_id_position = *position;

    // The '=' sign.
    *position = nextPosition;
    spvCheck(spvTextAdvance(text, position),
             DIAGNOSTIC << "Expected '=', found end of stream.";
             return SPV_ERROR_INVALID_TEXT);
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
    spvCheck(error, DIAGNOSTIC << "Internal Error"; return error);
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
  if (SPV_ASSEMBLY_SYNTAX_FORMAT_ASSIGNMENT == format) {
    // If this instruction has <result-id>, check it follows AAF.
    spvCheck(opcodeEntry->hasResult && result_id.empty(),
             DIAGNOSTIC << "Expected <result-id> at the beginning of an "
                           "instruction, found '"
                        << firstWord << "'.";
             return SPV_ERROR_INVALID_TEXT);
  }
  pInst->opcode = opcodeEntry->opcode;
  *position = nextPosition;
  pInst->wordCount++;

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

  while (!expectedOperands.empty()) {
    const spv_operand_type_t type = expectedOperands.front();
    expectedOperands.pop_front();

    // Expand optional tuples lazily.
    if (spvExpandOperandSequenceOnce(type, &expectedOperands))
      continue;

    if (type == SPV_OPERAND_TYPE_RESULT_ID && !result_id.empty()) {
      // Handle the <result-id> for value generating instructions.
      // We've already consumed it from the text stream.  Here
      // we inject its words into the instruction.
      error = spvTextEncodeOperand(
          SPV_OPERAND_TYPE_RESULT_ID, result_id.c_str(), operandTable,
          extInstTable, namedIdTable, pInst, nullptr, pBound,
          &result_id_position, pDiagnostic);
      spvCheck(error, return error);
    } else {
      // Find the next word.
      error = spvTextAdvance(text, position);
      if (error == SPV_END_OF_STREAM) {
        if (spvOperandIsOptional(type)) {
          // This would have been the last potential operand for the instruction,
          // and we didn't find one.  We're finished parsing this instruction.
          break;
        } else {
          DIAGNOSTIC << "Expected operand, found end of stream.";
          return SPV_ERROR_INVALID_TEXT;
        }
      }
      assert(error == SPV_SUCCESS && "Somebody added another way to fail");

      if (spvTextIsStartOfNewInst(text, position)) {
        if (spvOperandIsOptional(type)) {
          break;
        } else {
          DIAGNOSTIC << "Expected operand, found next instruction instead.";
          return SPV_ERROR_INVALID_TEXT;
        }
      }

      std::string operandValue;
      error = spvTextWordGet(text, position, operandValue, &nextPosition);
      spvCheck(error, DIAGNOSTIC << "Internal Error"; return error);

      error = spvTextEncodeOperand(
          type, operandValue.c_str(),
          operandTable, extInstTable, namedIdTable, pInst, &expectedOperands,
          pBound, position, pDiagnostic);

      if (error == SPV_FAILED_MATCH && spvOperandIsOptional(type))
        return SPV_SUCCESS;

      spvCheck(error, return error);

      *position = nextPosition;
    }
  }

  pInst->words[0] = spvOpcodeMake(pInst->wordCount, opcodeEntry->opcode);

  return SPV_SUCCESS;
}

namespace {

// Translates a given assembly language module into binary form.
// If a diagnostic is generated, it is not yet marked as being
// for a text-based input.
spv_result_t spvTextToBinaryInternal(const spv_text text,
                                     spv_assembly_syntax_format_t format,
                                     const spv_opcode_table opcodeTable,
                                     const spv_operand_table operandTable,
                                     const spv_ext_inst_table extInstTable,
                                     spv_binary *pBinary,
                                     spv_diagnostic *pDiagnostic) {
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

    spvCheck(spvTextEncodeOpcode(text, format, opcodeTable, operandTable,
                                 extInstTable, namedIdTable, &bound, &inst,
                                 &position, pDiagnostic),
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

} // anonymous namespace

spv_result_t spvTextToBinary(const char *input_text,
                             const uint64_t input_text_size,
                             const spv_opcode_table opcodeTable,
                             const spv_operand_table operandTable,
                             const spv_ext_inst_table extInstTable,
                             spv_binary *pBinary, spv_diagnostic *pDiagnostic) {
  return spvTextWithFormatToBinary(
      input_text, input_text_size, SPV_ASSEMBLY_SYNTAX_FORMAT_DEFAULT,
      opcodeTable, operandTable, extInstTable, pBinary, pDiagnostic);
}

spv_result_t spvTextWithFormatToBinary(
    const char *input_text, const uint64_t input_text_size,
    spv_assembly_syntax_format_t format, const spv_opcode_table opcodeTable,
    const spv_operand_table operandTable, const spv_ext_inst_table extInstTable,
    spv_binary *pBinary, spv_diagnostic *pDiagnostic) {
  spv_text_t text = {input_text, input_text_size};

  spv_result_t result =
      spvTextToBinaryInternal(&text, format, opcodeTable, operandTable,
                              extInstTable, pBinary, pDiagnostic);
  if (pDiagnostic && *pDiagnostic) (*pDiagnostic)->isTextSource = true;

  return result;
}

void spvTextDestroy(spv_text text) {
  spvCheck(!text, return );
  if (text->str) {
    delete[] text->str;
  }
  delete text;
}
