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
#include "table.h"

// Functions

/// @brief Grab the header from the SPV module
///
/// @param[in] binary the binary module
/// @param[in] endian the endianness of the module
/// @param[out] pHeader the returned header
///
/// @return result code
spv_result_t spvBinaryHeaderGet(const spv_const_binary binary,
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
