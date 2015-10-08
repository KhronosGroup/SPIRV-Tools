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

#include "text_handler.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "binary.h"
#include "ext_inst.h"
#include "instruction.h"
#include "opcode.h"
#include "text.h"

namespace {

/// @brief Advance text to the start of the next line
///
/// @param[in] text to be parsed
/// @param[in,out] position position text has been advanced to
///
/// @return result code
spv_result_t advanceLine(spv_text text, spv_position position) {
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

/// @brief Advance text to first non white space character
/// If a null terminator is found during the text advance SPV_END_OF_STREAM is
/// returned, SPV_SUCCESS otherwise. No error checking is performed on the
/// parameters, its the users responsibility to ensure these are non null.
///
/// @param[in] text to be parsed
/// @param[in,out] position text has been advanced to
///
/// @return result code
spv_result_t advance(spv_text text, spv_position position) {
  // NOTE: Consume white space, otherwise don't advance.
  switch (text->str[position->index]) {
    case '\0':
      return SPV_END_OF_STREAM;
    case ';':
      if (spv_result_t error = advanceLine(text, position)) return error;
      return advance(text, position);
    case ' ':
    case '\t':
      position->column++;
      position->index++;
      return advance(text, position);
    case '\n':
      position->column = 0;
      position->line++;
      position->index++;
      return advance(text, position);
    default:
      break;
  }
  return SPV_SUCCESS;
}

/// @brief Fetch the next word from the text stream.
///
/// A word ends at the next comment or whitespace.  However, double-quoted
/// strings remain intact, and a backslash always escapes the next character.
///
/// @param[in] text stream to read from
/// @param[in] position current position in text stream
/// @param[out] word returned word
/// @param[out] endPosition one past the end of the returned word
///
/// @return result code
spv_result_t getWord(spv_text text, spv_position position, std::string &word,
                     spv_position endPosition) {
  if (!text->str || !text->length) return SPV_ERROR_INVALID_TEXT;
  if (!position || !endPosition) return SPV_ERROR_INVALID_POINTER;

  *endPosition = *position;

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
          word.assign(text->str + position->index,
                      (size_t)(endPosition->index - position->index));
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

// Returns true if the characters in the text as position represent
// the start of an Opcode.
bool startsWithOp(spv_text text, spv_position position) {
  if (text->length < position->index + 3) return false;
  char ch0 = text->str[position->index];
  char ch1 = text->str[position->index + 1];
  char ch2 = text->str[position->index + 2];
  return ('O' == ch0 && 'p' == ch1 && ('A' <= ch2 && ch2 <= 'Z'));
}

/// @brief Parses a mask expression string for the given operand type.
///
/// A mask expression is a sequence of one or more terms separated by '|',
/// where each term a named enum value for the given type.  No whitespace
/// is permitted.
///
/// On success, the value is written to pValue.
///
/// @param[in] operandTable operand lookup table
/// @param[in] type of the operand
/// @param[in] textValue word of text to be parsed
/// @param[out] pValue where the resulting value is written
///
/// @return result code
spv_result_t spvTextParseMaskOperand(const spv_operand_table operandTable,
                                     const spv_operand_type_t type,
                                     const char *textValue, uint32_t *pValue) {
  if (textValue == nullptr) return SPV_ERROR_INVALID_TEXT;
  size_t text_length = strlen(textValue);
  if (text_length == 0) return SPV_ERROR_INVALID_TEXT;
  const char *text_end = textValue + text_length;

  // We only support mask expressions in ASCII, so the separator value is a
  // char.
  const char separator = '|';

  // Accumulate the result by interpreting one word at a time, scanning
  // from left to right.
  uint32_t value = 0;
  const char *begin = textValue;  // The left end of the current word.
  const char *end = nullptr;  // One character past the end of the current word.
  do {
    end = std::find(begin, text_end, separator);

    spv_operand_desc entry = nullptr;
    if (spvOperandTableNameLookup(operandTable, type, begin, end - begin,
                                  &entry)) {
      return SPV_ERROR_INVALID_TEXT;
    }
    value |= entry->value;

    // Advance to the next word by skipping over the separator.
    begin = end + 1;
  } while (end != text_end);

  *pValue = value;
  return SPV_SUCCESS;
}

}  // anonymous namespace

namespace libspirv {

bool AssemblyGrammar::isValid() const {
  return operandTable_ && opcodeTable_ && extInstTable_;
}

spv_result_t AssemblyGrammar::lookupOpcode(const char *name,
                                           spv_opcode_desc *desc) const {
  return spvOpcodeTableNameLookup(opcodeTable_, name, desc);
}
spv_result_t AssemblyGrammar::lookupOperand(spv_operand_type_t type,
                                            const char *name, size_t name_len,
                                            spv_operand_desc *desc) const {
  return spvOperandTableNameLookup(operandTable_, type, name, name_len, desc);
}

spv_result_t AssemblyGrammar::parseMaskOperand(const spv_operand_type_t type,
                                               const char *textValue,
                                               uint32_t *pValue) const {
  return spvTextParseMaskOperand(operandTable_, type, textValue, pValue);
}
spv_result_t AssemblyGrammar::lookupExtInst(spv_ext_inst_type_t type,
                                            const char *textValue,
                                            spv_ext_inst_desc *extInst) const {
  return spvExtInstTableNameLookup(extInstTable_, type, textValue, extInst);
}

void AssemblyGrammar::prependOperandTypesForMask(
    const spv_operand_type_t type, const uint32_t mask,
    spv_operand_pattern_t *pattern) const {
  spvPrependOperandTypesForMask(operandTable_, type, mask, pattern);
}

// This represents all of the data that is only valid for the duration of
// a single compilation.
uint32_t AssemblyContext::spvNamedIdAssignOrGet(const char *textValue) {
  if (named_ids_.end() == named_ids_.find(textValue)) {
    named_ids_[std::string(textValue)] = bound_++;
  }
  return named_ids_[textValue];
}
uint32_t AssemblyContext::getBound() const { return bound_; }

spv_result_t AssemblyContext::advance() {
  return ::advance(text_, &current_position_);
}

spv_result_t AssemblyContext::getWord(std::string &word,
                                      spv_position endPosition) {
  return ::getWord(text_, &current_position_, word, endPosition);
}

bool AssemblyContext::startsWithOp() {
  return ::startsWithOp(text_, &current_position_);
}

bool AssemblyContext::isStartOfNewInst() {
  spv_position_t nextPosition = current_position_;
  if (::advance(text_, &nextPosition)) return false;
  if (::startsWithOp(text_, &nextPosition)) return true;

  std::string word;
  spv_position_t startPosition = current_position_;
  if (::getWord(text_, &startPosition, word, &nextPosition)) return false;
  if ('%' != word.front()) return false;

  if (::advance(text_, &nextPosition)) return false;
  startPosition = nextPosition;
  if (::getWord(text_, &startPosition, word, &nextPosition)) return false;
  if ("=" != word) return false;

  if (::advance(text_, &nextPosition)) return false;
  startPosition = nextPosition;
  if (::startsWithOp(text_, &startPosition)) return true;
  return false;
}
char AssemblyContext::peek() const {
  return text_->str[current_position_.index];
}

bool AssemblyContext::hasText() const {
  return text_->length > current_position_.index;
}
std::string AssemblyContext::getWord() const {
  size_t index = current_position_.index;
  while (true) {
    switch (text_->str[index]) {
      case '\0':
      case '\t':
      case '\v':
      case '\r':
      case '\n':
      case ' ':
        return std::string(text_->str, text_->str + index);
      default:
        index++;
    }
  }
  assert(0 && "Unreachable");
  return "";  // Make certain compilers happy.
}

void AssemblyContext::seekForward(uint32_t size) {
  current_position_.index += size;
  current_position_.column += size;
}

spv_result_t AssemblyContext::binaryEncodeU32(const uint32_t value,
                                                     spv_instruction_t *pInst) {
  spvInstructionAddWord(pInst, value);
  return SPV_SUCCESS;
}

spv_result_t AssemblyContext::binaryEncodeU64(const uint64_t value,
                                                     spv_instruction_t *pInst) {
  uint32_t low = (uint32_t)(0x00000000ffffffff & value);
  uint32_t high = (uint32_t)((0xffffffff00000000 & value) >> 32);
  spv_result_t err = binaryEncodeU32(low, pInst);
  if (err != SPV_SUCCESS) {
    return err;
  }
  return binaryEncodeU32(high, pInst);
}

spv_result_t AssemblyContext::binaryEncodeString(
    const char *value, spv_instruction_t *pInst) {
  const size_t length = strlen(value);
  const size_t wordCount = (length / 4) + 1;
  const size_t oldWordCount = pInst->words.size();
  const size_t newWordCount = oldWordCount + wordCount;

  // TODO(dneto): We can just defer this check until later.
  if (newWordCount > SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX) {
    diagnostic() << "Instruction too long: more than "
             << SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX << " words.";
    return SPV_ERROR_INVALID_TEXT;
  }

  pInst->words.resize(newWordCount);

  // Make sure all the bytes in the last word are 0, in case we only
  // write a partial word at the end.
  pInst->words.back() = 0;

  char *dest = (char *)&pInst->words[oldWordCount];
  strncpy(dest, value, length);

  return SPV_SUCCESS;
}
}

