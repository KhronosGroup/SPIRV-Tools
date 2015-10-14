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

#ifndef _LIBSPIRV_UTIL_TEXT_HANDLER_H_
#define _LIBSPIRV_UTIL_TEXT_HANDLER_H_

#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>

#include <libspirv/libspirv.h>
#include "diagnostic.h"
#include "instruction.h"
#include "operand.h"
#include "text.h"

namespace libspirv {
// Structures

// This is a lattice for tracking types.
enum class IdTypeClass {
  kBottom = 0, // We have no information yet.
  kScalarIntegerType,
  kScalarFloatType,
  kOtherType
};


// Contains ID type information that needs to be tracked across all Ids.
// Bitwidth is only valid when type_class is kScalarIntegerType or
// kScalarFloatType.
struct IdType {
  uint32_t bitwidth;  // Safe to assume that we will not have > 2^32 bits.
  bool isSigned; // This is only significant if type_class is integral.
  IdTypeClass type_class;
};

// A value representing an unknown type.
extern const IdType kUnknownType;

// Returns true if the type is a scalar integer type.
inline bool isScalarIntegral(const IdType& type) {
  return type.type_class == IdTypeClass::kScalarIntegerType;
}

// Returns true if the type is a scalar floating point type.
inline bool isScalarFloating(const IdType& type) {
  return type.type_class == IdTypeClass::kScalarFloatType;
}

// Returns the number of bits in the type.
// This is only valid for bottom, scalar integer, and scalar floating
// classes.  For bottom, assume 32 bits.
inline int assumedBitWidth(const IdType& type) {
  switch(type.type_class) {
    case IdTypeClass::kBottom:
      return 32;
    case IdTypeClass::kScalarIntegerType:
    case IdTypeClass::kScalarFloatType:
      return type.bitwidth;
    default:
      break;
  }
  // We don't care about this case.
  return 0;
}

// Encapsulates the grammar to use for SPIR-V assembly.
// Contains methods to query for valid instructions and operands.
class AssemblyGrammar {
 public:
  AssemblyGrammar(const spv_operand_table operand_table,
                  const spv_opcode_table opcode_table,
                  const spv_ext_inst_table ext_inst_table)
      : operandTable_(operand_table),
        opcodeTable_(opcode_table),
        extInstTable_(ext_inst_table) {}

  // Returns true if the compilation_data has been initialized with valid data.
  bool isValid() const;

  // Fills in the desc parameter with the information about the opcode
  // of the given name. Returns SPV_SUCCESS if the opcode was found, and
  // SPV_ERROR_INVALID_LOOKUP if the opcode does not exist.
  spv_result_t lookupOpcode(const char *name, spv_opcode_desc *desc) const;

  // Fills in the desc parameter with the information about the opcode
  // of the valid. Returns SPV_SUCCESS if the opcode was found, and
  // SPV_ERROR_INVALID_LOOKUP if the opcode does not exist.
  spv_result_t lookupOpcode(Op opcode, spv_opcode_desc *desc) const;

  // Fills in the desc parameter with the information about the given
  // operand. Returns SPV_SUCCESS if the operand was found, and
  // SPV_ERROR_INVALID_LOOKUP otherwise.
  spv_result_t lookupOperand(spv_operand_type_t type, const char *name,
                             size_t name_len, spv_operand_desc *desc) const;

  // Parses a mask expression string for the given operand type.
  //
  // A mask expression is a sequence of one or more terms separated by '|',
  // where each term is a named enum value for a given type. No whitespace
  // is permitted.
  //
  // On success, the value is written to pValue, and SPV_SUCCESS is returend.
  // The operand type is defined by the type parameter, and the text to be
  // parsed is defined by the textValue parameter.
  spv_result_t parseMaskOperand(const spv_operand_type_t type,
                                const char *textValue, uint32_t *pValue) const;


  // Writes the extended operand with the given type and text to the *extInst
  // parameter.
  // Returns SPV_SUCCESS if the value could be found.
  spv_result_t lookupExtInst(spv_ext_inst_type_t type, const char *textValue,
                             spv_ext_inst_desc *extInst) const;

  // Inserts the operands expected after the given typed mask onto the front
  // of the given pattern.
  //
  // Each set bit in the mask represents zero or more operand types that should
  // be prepended onto the pattern. Opearnds for a less significant bit always
  // appear before operands for a more significatn bit.
  //
  // If a set bit is unknown, then we assume it has no operands.
  void prependOperandTypesForMask(const spv_operand_type_t type,
                                  const uint32_t mask,
                                  spv_operand_pattern_t *pattern) const;

 private:
  const spv_operand_table operandTable_;
  const spv_opcode_table opcodeTable_;
  const spv_ext_inst_table extInstTable_;
};

// Encapsulates the data used during the assembly of a SPIR-V module.
class AssemblyContext {
 public:
  AssemblyContext(spv_text text, spv_diagnostic *diagnostic)
      : current_position_({}),
        pDiagnostic_(diagnostic),
        text_(text),
        bound_(1) {}

  // Assigns a new integer value to the given text ID, or returns the previously
  // assigned integer value if the ID has been seen before.
  uint32_t spvNamedIdAssignOrGet(const char *textValue);

  // Returns the largest largest numeric ID that has been assigned.
  uint32_t getBound() const;

  // Advances position to point to the next word in the input stream.
  // Returns SPV_SUCCESS on success.
  spv_result_t advance();

  // Sets word to the next word in the input text. Fills endPosition with
  // the next location past the end of the word.
  spv_result_t getWord(std::string &word, spv_position endPosition);

  // Returns the next word in the input stream. It is invalid to call this
  // method if position has been set to a location in the stream that does not
  // exist. If there are no subsequent words, the empty string will be returend.
  std::string getWord() const;

  // Returns true if the next word in the input is the start of a new Opcode.
  bool startsWithOp();

  // Returns true if the next word in the input is the start of a new
  // instruction.
  bool isStartOfNewInst();

  // Returns a diagnostic object initialized with current position in the input
  // stream, and for the given error code. Any data written to this object will
  // show up in pDiagnsotic on destruction.
  DiagnosticStream diagnostic(spv_result_t error) {
    return DiagnosticStream(&current_position_, pDiagnostic_, error);
  }

  // Returns a diagnostic object with the default assembly error code.
  DiagnosticStream diagnostic() {
    // The default failure for assembly is invalid text.
    return diagnostic(SPV_ERROR_INVALID_TEXT);
  }

  // Returns then next characted in the input stream.
  char peek() const;

  // Returns true if there is more text in the input stream.
  bool hasText() const;

  // Seeks the input stream forward by 'size' characters.
  void seekForward(uint32_t size);

  // Sets the current position in the input stream to the given position.
  void setPosition(const spv_position_t &newPosition) {
    current_position_ = newPosition;
  }

  // Returns the current position in the input stream.
  const spv_position_t &position() const { return current_position_; }

  // Appends the given 32-bit value to the given instruction.
  // Returns SPV_SUCCESS if the value could be correctly inserted in the
  // instruction.
  spv_result_t binaryEncodeU32(const uint32_t value, spv_instruction_t *pInst);

  // Appends the given string to the given instruction.
  // Returns SPV_SUCCESS if the value could be correctly inserted in the
  // instruction.
  spv_result_t binaryEncodeString(const char *value, spv_instruction_t *pInst);

  // Appends the given numeric literal to the given instruction.
  // Validates and respects the bitwidth supplied in the IdType argument.
  // If the type is of class kBottom the value will be encoded as a
  // 32-bit integer.
  // Returns SPV_SUCCESS if the value could be correctly added to the
  // instruction.  Returns the given error code on failure, and emits
  // a diagnotic if that error code is not SPV_FAILED_MATCH.
  spv_result_t binaryEncodeNumericLiteral(const char *numeric_literal,
                                          spv_result_t error_code,
                                          const IdType &type,
                                          spv_instruction_t *pInst);

  // Returns the IdType associated with this type-generating value.
  // If the type has not been previously recorded with recordTypeDefinition,
  // kUnknownType  will be returned.
  IdType getTypeOfTypeGeneratingValue(uint32_t value) const;

  // Returns the IdType that represents the return value of this Value
  // generating instruction.
  // If the value has not been recorded with recordTypeIdForValue, or the type
  // could not be determined kUnknownType will be returned.
  IdType getTypeOfValueInstruction(uint32_t value) const;

  // Tracks the type-defining instruction. The result of the tracking can
  // later be queried using getValueType.
  // pInst is expected to be completely filled in by the time this instruction
  // is called.
  // Returns SPV_SUCCESS on success, or SPV_ERROR_INVALID_VALUE on error.
  spv_result_t recordTypeDefinition(const spv_instruction_t* pInst);

  // Tracks the relationship between the value and its type.
  spv_result_t recordTypeIdForValue(uint32_t value, uint32_t type);

  // Parses a numeric value of a given type from the given text.  The number
  // should take up the entire string, and should be within bounds for the
  // target type.  On success, returns SPV_SUCCESS and populates the object
  // referenced by value_pointer. On failure, returns the given error code,
  // and emits a diagnostic if that error code is not SPV_FAILED_MATCH.
  template <typename T>
  spv_result_t parseNumber(const char *text, spv_result_t error_code,
                           T *value_pointer,
                           const char *error_message_fragment) {
    // C++11 doesn't define std::istringstream(int8_t&), so calling this method
    // with a single-byte type leads to implementation-defined behaviour.
    // Similarly for uint8_t.
    static_assert(sizeof(T) > 1, "Don't use a single-byte type this parse method");

    std::istringstream text_stream(text);
    // Allow both decimal and hex input for integers.
    // It also allows octal input, but we don't care about that case.
    text_stream >> std::setbase(0);
    text_stream >> *value_pointer;
    bool ok = true;

    // We should have read something.
    ok = (text[0] != 0) && !text_stream.bad();
    // It should have been all the text.
    ok = ok && text_stream.eof();
    // It should have been in range.
    ok = ok && !text_stream.fail();
    // Work around a bug in the GNU C++11 library. It will happily parse
    // "-1" for uint16_t as 65535.
    if (ok && !std::is_signed<T>::value && (text[0] == '-') &&
        *value_pointer != 0) {
      ok = false;
      // Match expected error behaviour of std::istringstream::operator>>
      // on failure to parse.
      *value_pointer = 0;
    }

    if (ok) return SPV_SUCCESS;
    return diagnostic(error_code) << error_message_fragment << text;
  }

 private:
  // Appends the given floating point literal to the given instruction.
  // Returns SPV_SUCCESS if the value was correctly parsed.  Otherwise
  // returns the given error code, and emits a diagnostic if that error
  // code is not SPV_FAILED_MATCH.
  // Only 32 and 64 bit floating point numbers are supported.
  spv_result_t binaryEncodeFloatingPointLiteral(const char *numeric_literal,
                                                spv_result_t error_code,
                                                const IdType& type,
                                                spv_instruction_t *pInst);

  // Appends the given integer literal to the given instruction.
  // Returns SPV_SUCCESS if the value was correctly parsed.  Otherwise
  // returns the given error code, and emits a diagnostic if that error
  // code is not SPV_FAILED_MATCH.
  // Integers up to 64 bits are supported.
  spv_result_t binaryEncodeIntegerLiteral(const char *numeric_literal,
                                          spv_result_t error_code,
                                          const IdType &type,
                                          spv_instruction_t *pInst);

  // Returns SPV_SUCCESS if the given value fits within the target scalar
  // integral type.  The target type may have an unusual bit width.
  // If the value was originally specified as a hexadecimal number, then
  // the overflow bits should be zero.  If it was hex and the target type is
  // signed, then return the sign-extended value through the
  // updated_value_for_hex pointer argument.
  // On failure, return the given error code and emit a diagnostic if that error
  // code is not SPV_FAILED_MATCH.
  template <typename T>
  spv_result_t checkRangeAndIfHexThenSignExtend(T value, spv_result_t error_code,
                                                const IdType &type, bool is_hex,
                                                T *updated_value_for_hex);

  // Writes the given 64-bit literal value into the instruction.
  // return SPV_SUCCESS if the value could be written in the instruction.
  spv_result_t binaryEncodeU64(const uint64_t value, spv_instruction_t *pInst);
  // Maps ID names to their corresponding numerical ids.
  using spv_named_id_table = std::unordered_map<std::string, uint32_t>;
  // Maps type-defining IDs to their IdType.
  using spv_id_to_type_map = std::unordered_map<uint32_t, IdType>;
  // Maps Ids to the id of their type.
  using spv_id_to_type_id = std::unordered_map<uint32_t, uint32_t>;

  spv_named_id_table named_ids_;
  spv_id_to_type_map types_;
  spv_id_to_type_id value_types_;
  spv_position_t current_position_;
  spv_diagnostic *pDiagnostic_;
  spv_text text_;
  uint32_t bound_;
};
}
#endif  // _LIBSPIRV_UTIL_TEXT_HANDLER_H_

