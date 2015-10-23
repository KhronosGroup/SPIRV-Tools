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
#include <cstdlib>
#include <cstring>
#include <tuple>

#include "binary.h"
#include "ext_inst.h"
#include "instruction.h"
#include "opcode.h"
#include "text.h"
#include "util/bitutils.h"

namespace {

using spvutils::BitwiseCast;

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

const IdType kUnknownType = {0, false, IdTypeClass::kBottom};

bool AssemblyGrammar::isValid() const {
  return operandTable_ && opcodeTable_ && extInstTable_;
}

spv_result_t AssemblyGrammar::lookupOpcode(const char *name,
                                           spv_opcode_desc *desc) const {
  return spvOpcodeTableNameLookup(opcodeTable_, name, desc);
}

spv_result_t AssemblyGrammar::lookupOpcode(Op opcode,
                                           spv_opcode_desc *desc) const {
  return spvOpcodeTableValueLookup(opcodeTable_, opcode, desc);
}

spv_result_t AssemblyGrammar::lookupOperand(spv_operand_type_t type,
                                            const char *name, size_t name_len,
                                            spv_operand_desc *desc) const {
  return spvOperandTableNameLookup(operandTable_, type, name, name_len, desc);
}

spv_result_t AssemblyGrammar::lookupOperand(spv_operand_type_t type,
                                            uint32_t operand,
                                            spv_operand_desc *desc) const {
  return spvOperandTableValueLookup(operandTable_, type, operand, desc);
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

spv_result_t AssemblyGrammar::lookupExtInst(spv_ext_inst_type_t type,
                                            uint32_t firstWord,
                                            spv_ext_inst_desc *extInst) const {
  return spvExtInstTableValueLookup(extInstTable_, type, firstWord, extInst);
}

void AssemblyGrammar::prependOperandTypesForMask(
    const spv_operand_type_t type, const uint32_t mask,
    spv_operand_pattern_t *pattern) const {
  spvPrependOperandTypesForMask(operandTable_, type, mask, pattern);
}

// TODO(dneto): Reorder AssemblyContext definitions to match declaration order.

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
  uint64_t index = current_position_.index;
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
  uint32_t low = uint32_t(0x00000000ffffffff & value);
  uint32_t high = uint32_t((0xffffffff00000000 & value) >> 32);
  binaryEncodeU32(low, pInst);
  binaryEncodeU32(high, pInst);
  return SPV_SUCCESS;
}

spv_result_t AssemblyContext::binaryEncodeNumericLiteral(
    const char *val, spv_result_t error_code, const IdType &type,
    spv_instruction_t *pInst) {
  const bool is_bottom = type.type_class == libspirv::IdTypeClass::kBottom;
  const bool is_floating = libspirv::isScalarFloating(type);
  const bool is_integer = libspirv::isScalarIntegral(type);

  if (!is_bottom && !is_floating && !is_integer) {
    return diagnostic(SPV_ERROR_INTERNAL)
           << "The expected type is not a scalar integer or float type";
  }

  // If this is bottom, but looks like a float, we should treat it like a
  // float.
  const bool looks_like_float = is_bottom && strchr(val, '.');

  // If we explicitly expect a floating-point number, we should handle that
  // first.
  if (is_floating || looks_like_float)
    return binaryEncodeFloatingPointLiteral(val, error_code, type, pInst);

  return binaryEncodeIntegerLiteral(val, error_code, type, pInst);
}

spv_result_t AssemblyContext::binaryEncodeString(
    const char *value, spv_instruction_t *pInst) {
  const size_t length = strlen(value);
  const size_t wordCount = (length / 4) + 1;
  const size_t oldWordCount = pInst->words.size();
  const size_t newWordCount = oldWordCount + wordCount;

  // TODO(dneto): We can just defer this check until later.
  if (newWordCount > SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX) {
    return diagnostic() << "Instruction too long: more than "
                        << SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX << " words.";
  }

  pInst->words.resize(newWordCount);

  // Make sure all the bytes in the last word are 0, in case we only
  // write a partial word at the end.
  pInst->words.back() = 0;

  char *dest = (char *)&pInst->words[oldWordCount];
  strncpy(dest, value, length);

  return SPV_SUCCESS;
}

spv_result_t AssemblyContext::recordTypeDefinition(
    const spv_instruction_t *pInst) {
  uint32_t value = pInst->words[1];
  if (types_.find(value) != types_.end()) {
    return diagnostic()
           << "Value " << value << " has already been used to generate a type";
  }

  if (pInst->opcode == OpTypeInt) {
    if (pInst->words.size() != 4)
      return diagnostic() << "Invalid OpTypeInt instruction";
    types_[value] = {pInst->words[2], pInst->words[3] != 0,
                     IdTypeClass::kScalarIntegerType};
  } else if (pInst->opcode == OpTypeFloat) {
    if (pInst->words.size() != 3)
      return diagnostic() << "Invalid OpTypeFloat instruction";
    types_[value] = {pInst->words[2], false, IdTypeClass::kScalarFloatType};
  } else {
    types_[value] = {0, false, IdTypeClass::kOtherType};
  }
  return SPV_SUCCESS;
}

IdType AssemblyContext::getTypeOfTypeGeneratingValue(uint32_t value) const {
  auto type = types_.find(value);
  if (type == types_.end()) {
    return kUnknownType;
  }
  return std::get<1>(*type);
}

IdType AssemblyContext::getTypeOfValueInstruction(uint32_t value) const {
  auto type_value = value_types_.find(value);
  if (type_value == value_types_.end()) {
    return { 0, false, IdTypeClass::kBottom};
  }
  return getTypeOfTypeGeneratingValue(std::get<1>(*type_value));
}

spv_result_t AssemblyContext::recordTypeIdForValue(uint32_t value,
                                                   uint32_t type) {
  bool successfully_inserted = false;
  std::tie(std::ignore, successfully_inserted) =
      value_types_.insert(std::make_pair(value, type));
  if (!successfully_inserted)
    return diagnostic() << "Value is being defined a second time";
  return SPV_SUCCESS;
}

spv_result_t AssemblyContext::binaryEncodeFloatingPointLiteral(
    const char *val, spv_result_t error_code, const IdType &type,
    spv_instruction_t *pInst) {
  const auto bit_width = assumedBitWidth(type);
  switch (bit_width) {
    case 16:
      return diagnostic(SPV_ERROR_INTERNAL)
             << "Unsupported yet: 16-bit float constants.";
    case 32: {
      float fVal;
      if (auto error = parseNumber(val, error_code, &fVal,
                                   "Invalid 32-bit float literal: "))
        return error;
      return binaryEncodeU32(BitwiseCast<uint32_t>(fVal), pInst);
    } break;
    case 64: {
      double dVal;
      if (auto error = parseNumber(val, error_code, &dVal,
                                   "Invalid 64-bit float literal: "))
        return error;
      return binaryEncodeU64(BitwiseCast<uint64_t>(dVal), pInst);
    } break;
    default:
      break;
  }
  return diagnostic() << "Unsupported " << bit_width << "-bit float literals";
}

spv_result_t AssemblyContext::binaryEncodeIntegerLiteral(
    const char *val, spv_result_t error_code, const IdType &type,
    spv_instruction_t *pInst) {
  const bool is_bottom = type.type_class == libspirv::IdTypeClass::kBottom;
  const auto bit_width = assumedBitWidth(type);

  if (bit_width > 64)
    return diagnostic(SPV_ERROR_INTERNAL) << "Unsupported " << bit_width
                                          << "-bit integer literals";

  // Either we are expecting anything or integer.
  bool is_negative = val[0] == '-';
  bool can_be_signed = is_bottom || type.isSigned;

  if (is_negative && !can_be_signed) {
    return diagnostic()
           << "Cannot put a negative number in an unsigned literal";
  }

  const bool is_hex = val[0] == '0' && (val[1] == 'x' || val[1] == 'X');

  uint64_t decoded_bits;
  if (is_negative) {
    int64_t decoded_signed = 0;
    if (auto error = parseNumber(val, error_code, &decoded_signed,
                                 "Invalid signed integer literal: "))
      return error;
    if (auto error = checkRangeAndIfHexThenSignExtend(
            decoded_signed, error_code, type, is_hex, &decoded_signed))
      return error;
    decoded_bits = decoded_signed;
  } else {
    // There's no leading minus sign, so parse it as an unsigned integer.
    if (auto error = parseNumber(val, error_code, &decoded_bits,
                                 "Invalid unsigned integer literal: "))
      return error;
    if (auto error = checkRangeAndIfHexThenSignExtend(
            decoded_bits, error_code, type, is_hex, &decoded_bits))
      return error;
  }
  if (bit_width > 32) {
    return binaryEncodeU64(decoded_bits, pInst);
  } else {
    return binaryEncodeU32(uint32_t(decoded_bits), pInst);
  }
}

template <typename T>
spv_result_t AssemblyContext::checkRangeAndIfHexThenSignExtend(
    T value, spv_result_t error_code, const IdType &type, bool is_hex,
    T *updated_value_for_hex) {
  // The encoded result has three regions of bits that are of interest, from
  // least to most significant:
  //   - magnitude bits, where the magnitude of the number would be stored if
  //     we were using a signed-magnitude representation.
  //   - an optional sign bit
  //   - overflow bits, up to bit 63 of a 64-bit number
  // For example:
  //   Type                Overflow      Sign       Magnitude
  //   ---------------     --------      ----       ---------
  //   unsigned 8 bit      8-63          n/a        0-7
  //   signed 8 bit        8-63          7          0-6
  //   unsigned 16 bit     16-63         n/a        0-15
  //   signed 16 bit       16-63         15         0-14

  // We'll use masks to define the three regions.
  // At first we'll assume the number is unsigned.
  const uint32_t bit_width = assumedBitWidth(type);
  uint64_t magnitude_mask =
      (bit_width == 64) ? -1 : ((uint64_t(1) << bit_width) - 1);
  uint64_t sign_mask = 0;
  uint64_t overflow_mask = ~magnitude_mask;

  if (value < 0 || type.isSigned) {
    // Accommodate the sign bit.
    magnitude_mask >>= 1;
    sign_mask = magnitude_mask + 1;
  }

  bool failed = false;
  if (value < 0) {
    // The top bits must all be 1 for a negative signed value.
    failed = ((value & overflow_mask) != overflow_mask) ||
             ((value & sign_mask) != sign_mask);
  } else {
    if (is_hex) {
      // Hex values are a bit special. They decode as unsigned values, but
      // may represent a negative number.  In this case, the overflow bits
      // should be zero.
      failed = (value & overflow_mask);
    } else {
      // Check overflow in the ordinary case.
      failed = (value & magnitude_mask) != value;
    }
  }

  if (failed) {
    return diagnostic(error_code)
           << "Integer " << (is_hex ? std::hex : std::dec) << std::showbase
           << value << " does not fit in a " << std::dec << bit_width << "-bit "
           << (type.isSigned ? "signed" : "unsigned") << " integer";
  }

  // Sign extend hex the number.
  if (is_hex && (value & sign_mask))
    *updated_value_for_hex = (value | overflow_mask);

  return SPV_SUCCESS;
}
} // namespace libspirv
