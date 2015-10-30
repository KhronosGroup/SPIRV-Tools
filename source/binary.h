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

#ifndef LIBSPIRV_BINARY_H_
#define LIBSPIRV_BINARY_H_

#include "libspirv/libspirv.h"


// TODO(dneto): Move spvBinaryParse and related type definitions to libspirv.h
extern "C" {

// The parser interface.
// This is a C interface because we expect to expose this to clients of the
// SPIR-V Tools C API.

// A pointer to a function that accepts a parsed SPIR-V header.
// The integer arguments are the 32-bit words from the header, as specified
// in SPIR-V 1.0 Section 2.3 Table 1.
// The function should return SPV_SUCCESS if parsing should continue.
typedef spv_result_t (*spv_parsed_header_fn_t)(
    void* user_data, spv_endianness_t endian, uint32_t magic, uint32_t version,
    uint32_t generator, uint32_t id_bound, uint32_t reserved);

// Number kind.  This determines the format at a high level, but not
// the bit width.
// In principle, these could probably be folded into new entries in
// spv_operand_type_t.  But then we'd have some special case differences
// between the assembler and disassembler.
typedef enum spv_number_kind_t {
  SPV_NUMBER_NONE = 0,  // The default for value initialization.
  SPV_NUMBER_UNSIGNED_INT,
  SPV_NUMBER_SIGNED_INT,
  SPV_NUMBER_FLOATING,
} spv_number_kind_t;

// Information about a parsed operand.  Note that the values are not
// included.  You still need access to the binary to extract the values.
typedef struct spv_parsed_operand_t {
  // Location of the operand, in words from the start of the instruction.
  uint16_t offset;
  // Number of words occupied by this operand.
  uint16_t num_words;
  // The "concrete" operand type.  See the definition of spv_operand_type_t
  // for details.
  spv_operand_type_t type;
  // If type is a literal number type, then number_kind says whether it's
  // a signed integer, an unsigned integer, or a floating point number.
  spv_number_kind_t number_kind;
  // The number of bits for a literal number type.
  uint32_t number_bit_width;
} spv_parsed_operand_info_t;

// A parsed instruction.
typedef struct spv_parsed_instruction_t {
  // Location of the instruction, in words from the start of the SPIR-V binary.
  size_t offset;
  SpvOp opcode;
  // The extended instruction type, if opcode is OpExtInst.  Otherwise
  // this is the "none" value.
  spv_ext_inst_type_t ext_inst_type;
  // The type id, or 0 if this instruction doesn't have one.
  uint32_t type_id;
  // The result id, or 0 if this instruction doesn't have one.
  uint32_t result_id;
  // The array of parsed operands.
  const spv_parsed_operand_t* operands;
  uint16_t num_operands;
} spv_parsed_instruction_t;

// A pointer to a function that accepts a parsed SPIR-V instruction.
// The parsed_instruction value is transient: it may be overwritten
// or released immediately after the function has returned.  The function
// should return SPV_SUCCESS if and only if parsing should continue.
typedef spv_result_t (*spv_parsed_instruction_fn_t)(
    void* user_data, const spv_parsed_instruction_t* parsed_instruction);

// Parses a SPIR-V binary, specified as counted sequence of 32-bit words.
// Parsing feedback is provided via two callbacks.  In a valid parse, the
// parsed-header callback is called once, and then the parsed-instruction
// callback once for each instruction in the stream.  The user_data parameter
// is supplied as context to the callbacks.  Returns SPV_SUCCESS on successful
// parse where the callbacks always return SPV_SUCCESS.  For an invalid parse,
// returns SPV_ERROR_INVALID_BINARY and emits a diagnostic.  If a callback
// returns anything other than SPV_SUCCESS, then that error code is returned
// and parsing terminates early.
spv_result_t spvBinaryParse(void* user_data, const uint32_t* const words,
                            const size_t num_words,
                            spv_parsed_header_fn_t parse_header,
                            spv_parsed_instruction_fn_t parse_instruction,
                            spv_diagnostic* diagnostic);

} // extern "C"

// Functions

/// @brief Grab the header from the SPV module
///
/// @param[in] binary the binary module
/// @param[in] endian the endianness of the module
/// @param[out] pHeader the returned header
///
/// @return result code
spv_result_t spvBinaryHeaderGet(const spv_binary binary,
                                const spv_endianness_t endian,
                                spv_header_t* pHeader);

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
                                        spv_operand_desc* pOperandEntry);

#endif  // LIBSPIRV_BINARY_H_
