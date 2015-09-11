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

#ifndef _LIBSPIRV_UTIL_BINARY_H_
#define _LIBSPIRV_UTIL_BINARY_H_

#include <libspirv/libspirv.h>
#include "operand.h"
#include "print.h"

// Functions

/// @brief Fix the endianness of a word
///
/// @param[in] word whos endianness should be fixed
/// @param[in] endian the desired endianness
///
/// @return word with host endianness correction
uint32_t spvFixWord(const uint32_t word, const spv_endianness_t endian);

/// @brief Determine the endianness of the SPV binary
///
/// Gets the endianness of the SPV source. Returns SPV_ENDIANNESS_UNKNOWN if
/// the
/// SPV magic number is invalid, otherwise the determined endianness.
///
/// @param[in] binary the binary module
/// @param[out] pEndian return the endianness of the SPV module
///
/// @return result code
spv_result_t spvBinaryEndianness(const spv_binary binary,
                                 spv_endianness_t *pEndian);

/// @brief Grab the header from the SPV module
///
/// @param[in] binary the binary module
/// @param[in] endian the endianness of the module
/// @param[out] pHeader the returned header
///
/// @return result code
spv_result_t spvBinaryHeaderGet(const spv_binary binary,
                                const spv_endianness_t endian,
                                spv_header_t *pHeader);

/// @brief Populate a binary stream with this generators header
///
/// @param[in,out] binary the binary stream
/// @param[in] bound the upper ID bound
///
/// @return result code
spv_result_t spvBinaryHeaderSet(spv_binary binary, const uint32_t bound);

/// @brief Append a single word into a binary stream
///
/// @param[in] value the word to encode
/// @param[in] pInst the stream to append to
/// @param[in,out] position position in the binary
/// @param[out] pDiagnostic contains diagnostic on failure
///
/// @return result code
spv_result_t spvBinaryEncodeU32(const uint32_t value, spv_instruction_t *pInst,
                                const spv_position position,
                                spv_diagnostic *pDiagnostic);

/// @brief Append two related words into the binary stream
///
/// @param[in] value the two words to encode
/// @param[in] pInst the stream to append to
/// @param[in,out] position position in the binary
/// @param[out] pDiagnostic contains diagnostic on failure
///
/// @return result code
spv_result_t spvBinaryEncodeU64(const uint64_t value, spv_instruction_t *pInst,
                                const spv_position position,
                                spv_diagnostic *pDiagnostic);

/// @brief Append a string literal in the binary stream
///
/// @param[in] str the string to encode
/// @param[in] pInst the stream to append to
/// @param[in,out] position position in the binary
/// @param[out] pDiagnostic contains diagnostic on failure
///
/// @return result code
spv_result_t spvBinaryEncodeString(const char *str, spv_instruction_t *pInst,
                                   const spv_position position,
                                   spv_diagnostic *pDiagnostic);

/// @brief Determine the type of the desired operand
///
/// @param[in] word the operand value
/// @param[in] index the word index in the instruction
/// @param[in] opcodeEntry table of specified Opcodes
/// @param[in] operandTable table of specified operands
/// @param[in,out] pOperandEntry the entry in the operand table
///
/// @return type returned
spv_operand_type_t spvBinaryOperandInfo(const uint32_t word,
                                        const uint16_t index,
                                        const spv_opcode_desc opcodeEntry,
                                        const spv_operand_table operandTable,
                                        spv_operand_desc *pOperandEntry);

/// @brief Translate a binary operand to the textual form
///
/// @param[in] opcode of the current instruction
/// @param[in] type type of the operand to decode
/// @param[in] words the binary stream of words
/// @param[in] endian the endianness of the stream
/// @param[in] options bitfield of spv_binary_to_text_options_t values
/// @param[in] operandTable table of specified operands
/// @param[in,out] pExpectedOperands the expected operand types
/// @param[in,out] pExtInstType type of extended instruction library
/// @param[in,out] stream the text output stream
/// @param[in,out] position position in the binary stream
/// @param[out] pDiag return diagnostic on error
///
/// @return result code
spv_result_t spvBinaryDecodeOperand(
    const Op opcode, const spv_operand_type_t type, const uint32_t *words,
    const spv_endianness_t endian, const uint32_t options,
    const spv_operand_table operandTable, const spv_ext_inst_table extInstTable,
    spv_operand_pattern_t *pExpectedOperands, spv_ext_inst_type_t *pExtInstType,
    out_stream &stream, spv_position position, spv_diagnostic *pDiagnostic);

/// @brief Translate binary Opcode stream to textual form
///
/// @param[in] pInst the Opcode instruction stream
/// @param[in] endian the endianness of the stream
/// @param[in] options bitfield of spv_binary_to_text_options_t values
/// @param[in] opcodeTable table of specified Opcodes
/// @param[in] operandTable table of specified operands
/// @param[in] format the assembly syntax format to decode into
/// @param[out] stream output text stream
/// @param[in,out] position position in the stream
/// @param[out] pDiag return diagnostic on error
///
/// @return result code
spv_result_t spvBinaryDecodeOpcode(
    spv_instruction_t *pInst, const spv_endianness_t endian,
    const uint32_t options, const spv_opcode_table opcodeTable,
    const spv_operand_table operandTable, const spv_ext_inst_table extInstTable,
    spv_assembly_syntax_format_t format, out_stream &stream,
    spv_position position, spv_diagnostic *pDiag);

#endif
