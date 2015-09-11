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

#ifndef _LIBSPIRV_UTIL_TEXT_H_
#define _LIBSPIRV_UTIL_TEXT_H_

#include <libspirv/libspirv.h>
#include "operand.h"

#include <string>

// Structures

typedef enum spv_literal_type_t {
  SPV_LITERAL_TYPE_INT_32,
  SPV_LITERAL_TYPE_INT_64,
  SPV_LITERAL_TYPE_UINT_32,
  SPV_LITERAL_TYPE_UINT_64,
  SPV_LITERAL_TYPE_FLOAT_32,
  SPV_LITERAL_TYPE_FLOAT_64,
  SPV_LITERAL_TYPE_STRING,
  SPV_FORCE_32_BIT_ENUM(spv_literal_type_t)
} spv_literal_type_t;

typedef struct spv_literal_t {
  spv_literal_type_t type;
  union value_t {
    int32_t i32;
    int64_t i64;
    uint32_t u32;
    uint64_t u64;
    float f;
    double d;
    char str[SPV_LIMIT_LITERAL_STRING_MAX];
  } value;
} spv_literal_t;

struct spv_named_id_table_t;

// Types

typedef spv_named_id_table_t *spv_named_id_table;

// Functions

/// @brief Returns the word at the beginning of the given string.
///
/// A word ends at the first space, tab, form feed, carriage return, newline,
/// or at the end of the string.
///
/// @param[in] str the source string
///
/// @return word as a string
std::string spvGetWord(const char* str);

/// @brief Advance text to the start of the next line
///
/// @param[in] text to be parsed
/// @param[in,out] pPosition position text has been advanced to
///
/// @return result code
spv_result_t spvTextAdvanceLine(const spv_text text, spv_position_t *pPosition);

/// @brief Advance text to first non white space character
///
/// If a null terminator is found during the text advance SPV_END_OF_STREAM is
/// returned, SPV_SUCCESS otherwise. No error checking is performed on the
/// parameters, its the users responsibility to ensure these are non null.
///
/// @param[in] text to be parsed
/// @param[in,out] pPosition position text has been advanced to
///
/// @return result code
spv_result_t spvTextAdvance(const spv_text text, spv_position_t *pPosition);

/// @brief Fetch the next word from the text stream.
///
/// A word ends at the next comment or whitespace.  However, double-quoted
/// strings remain intact, and a backslash always escapes the next character.
///
/// @param[in] text stream to read from
/// @param[in] startPosition current position in text stream
/// @param[out] word returned word
/// @param[out] endPosition one past the end of the returned word
///
/// @return result code
spv_result_t spvTextWordGet(const spv_text text,
                            const spv_position startPosition, std::string &word,
                            spv_position endPosition);

/// @brief Returns true if the given text can start a new instruction.
///
/// @param[in] text stream to read from
/// @param[in] startPosition current position in text stream
///
/// @return result code
bool spvTextIsStartOfNewInst(const spv_text text,
                             const spv_position startPosition);

/// @brief Fetch a string, including quotes, from the text stream
///
/// @param[in] text stream to read from
/// @param[in] startPosition current position in text stream
/// @param[out] string returned string
/// @param[out] endPosition one past the end of the return string
///
/// @return result code
spv_result_t spvTextStringGet(const spv_text text,
                              const spv_position startPosition,
                              std::string &string, spv_position endPosition);

/// @brief Convert the input text to a unsigned 32 bit integer
///
/// @param[in] textValue input text to parse
/// @param[out] pValue the returned integer
///
/// @return result code
spv_result_t spvTextToUInt32(const char *textValue, uint32_t *pValue);

/// @brief Convert the input text to one of the number types.
///
/// String literals must be surrounded by double-quotes ("), which are
/// then stripped.
///
/// @param[in] textValue input text to parse
/// @param[out] pLiteral the returned literal number
///
/// @return result code
spv_result_t spvTextToLiteral(const char *textValue, spv_literal_t *pLiteral);

/// @brief Create a named ID table
///
/// @return named ID table
spv_named_id_table spvNamedIdTableCreate();

/// @brief Free a named ID table
///
/// @param table named ID table
void spvNamedIdTableDestory(spv_named_id_table table);

/// @brief Lookup or assign a named ID
///
/// @param table named ID table
/// @param textValue name value
/// @param pBound upper ID bound, used for assigning new ID's
///
/// @return the new ID assossiated with the named ID
uint32_t spvNamedIdAssignOrGet(spv_named_id_table table, const char *textValue,
                               uint32_t *pBound);

/// @brief Determine if a name has an assossiated ID
///
/// @param textValue name value
///
/// @return zero on failure, non-zero otherwise
int32_t spvTextIsNamedId(const char *textValue);

/// @brief Translate an Opcode operand to binary form
///
/// @param[in] type of the operand
/// @param[in] textValue word of text to be parsed
/// @param[in] operandTable operand lookup table
/// @param[in,out] namedIdTable table of named ID's
/// @param[out] pInst return binary Opcode
/// @param[in,out] pExpectedOperands the operand types expected
/// @param[in,out] pBound current highest defined ID value
/// @param[in] pPosition used in diagnostic on error
/// @param[out] pDiagnostic populated on error
///
/// @return result code
spv_result_t spvTextEncodeOperand(
    const spv_operand_type_t type, const char *textValue,
    const spv_operand_table operandTable, const spv_ext_inst_table extInstTable,
    spv_named_id_table namedIdTable, spv_instruction_t *pInst,
    spv_operand_pattern_t* pExpectedOperands, uint32_t *pBound,
    const spv_position_t *pPosition, spv_diagnostic *pDiagnostic);

/// @brief Translate single Opcode and operands to binary form
///
/// @param[in] text stream to translate
/// @param[in] format the assembly syntax format of text
/// @param[in] opcodeTable Opcode lookup table
/// @param[in] operandTable operand lookup table
/// @param[in,out] namedIdTable table of named ID's
/// @param[in,out] pBound current highest defined ID value
/// @param[out] pInst returned binary Opcode
/// @param[in,out] pPosition in the text stream
/// @param[out] pDiagnostic populated on failure
///
/// @return result code
spv_result_t spvTextEncodeOpcode(
    const spv_text text, spv_assembly_syntax_format_t format,
    const spv_opcode_table opcodeTable, const spv_operand_table operandTable,
    const spv_ext_inst_table extInstTable, spv_named_id_table namedIdTable,
    uint32_t *pBound, spv_instruction_t *pInst, spv_position_t *pPosition,
    spv_diagnostic *pDiagnostic);

#endif
