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

#ifndef _LIBSPIRV_UTIL_OPCODE_H_
#define _LIBSPIRV_UTIL_OPCODE_H_

#include "instruction.h"
#include <libspirv/libspirv.h>

// Functions

/// @brief Get the name of the SPIR-V generator.
///
/// See the registry at
/// https://www.khronos.org/registry/spir-v/api/spir-v.xml
///
/// @param[in] generator Khronos SPIR-V generator ID
///
/// @return string name
const char *spvGeneratorStr(uint32_t generator);

/// @brief Combine word count and Opcode enumerant in single word
///
/// @param[in] wordCount Opcode consumes
/// @param[in] opcode enumerant value
///
/// @return Opcode word
uint32_t spvOpcodeMake(uint16_t wordCount, Op opcode);

/// @brief Split the binary opcode into its constituent parts
///
/// @param[in] word binary opcode to split
/// @param[out] wordCount the returned number of words (optional)
/// @param[out] opcode the returned opcode enumerant (optional)
void spvOpcodeSplit(const uint32_t word, uint16_t *wordCount, Op *opcode);

/// @brief Find the named Opcode in the table
///
/// @param[in] table to lookup
/// @param[in] name name of Opcode to find
/// @param[out] pEntry returned Opcode table entry
///
/// @return result code
spv_result_t spvOpcodeTableNameLookup(const spv_opcode_table table,
                                      const char *name,
                                      spv_opcode_desc *pEntry);

/// @brief Find the opcode ID in the table
///
/// @param[out] table to lookup
/// @param[in] opcode value of Opcode to fine
/// @param[out] pEntry return Opcode table entry
///
/// @return result code
spv_result_t spvOpcodeTableValueLookup(const spv_opcode_table table,
                                       const Op opcode,
                                       spv_opcode_desc *pEntry);

/// @brief Get the argument index for the <result-id> operand, if any.
///
/// @param[in] entry the Opcode entry
///
/// @return index for the <result-id> operand, or
///         SPV_OPERAND_INVALID_RESULT_ID_INDEX if the given opcode
///         doesn't have a <result-id> operand.
//
/// For example, 0 means <result-id> is the first argument, i.e. right after
/// the wordcount/opcode word.
int16_t spvOpcodeResultIdIndex(spv_opcode_desc entry);

/// @brief Determine if the Opcode has capability requirements.
///
/// This function does not check if @a entry is valid.
///
/// @param[in] entry the Opcode entry
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeRequiresCapabilities(spv_opcode_desc entry);

/// @brief Copy an instructions word and fix the endianness
///
/// @param[in] words the input instruction stream
/// @param[in] opcode the instructions Opcode
/// @param[in] wordCount the number of words to copy
/// @param[in] endian the endianness of the stream
/// @param[out] pInst the returned instruction
void spvInstructionCopy(const uint32_t *words, const Op opcode,
                        const uint16_t wordCount, const spv_endianness_t endian,
                        spv_instruction_t *pInst);

/// @brief Get the string of an OpCode
///
/// @param[in] opcode the opcode
///
/// @return the opcode string
const char *spvOpcodeString(const Op opcode);

/// @brief Determine if the Opcode is a type
///
/// @param[in] opcode the opcode
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeIsType(const Op opcode);

/// @brief Determine if the OpCode is a scalar type
///
/// @param[in] opcode the opcode
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeIsScalarType(const Op opcode);

/// @brief Determine if the Opcode is a constant
///
/// @param[in] opcode the opcode
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeIsConstant(const Op opcode);

/// @brief Determine if the Opcode is a composite type
///
/// @param[in] opcode the opcode
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeIsComposite(const Op opcode);

/// @brief Deep comparison of type declaration instructions
///
/// @param[in] pTypeInst0 type definition zero
/// @param[in] pTypeInst1 type definition one
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeAreTypesEqual(const spv_instruction_t *pTypeInst0,
                               const spv_instruction_t *pTypeInst1);

/// @brief Determine if the Opcode results in a pointer
///
/// @param[in] opcode the opcode
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeIsPointer(const Op opcode);

/// @brief Determine if the Opcode results in a instantation of a non-void type
///
/// @param[in] opcode the opcode
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeIsObject(const Op opcode);

/// @brief Determine if the scalar type Opcode is nullable
///
/// @param[in] opcode the opcode
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeIsBasicTypeNullable(Op opcode);

/// @brief Determine if instruction is in a basic block
///
/// @param[in] pFirstInst first instruction in the stream
/// @param[in] pInst current instruction
///
/// @return zero if false, non-zero otherwise
int32_t spvInstructionIsInBasicBlock(const spv_instruction_t *pFirstInst,
                                     const spv_instruction_t *pInst);

/// @brief Determine if the Opcode contains a value
///
/// @param[in] opcode the opcode
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeIsValue(Op opcode);

/// @brief Determine if the Opcode generates a type
///
/// @param[in] opcode the opcode
///
/// @return zero if false, non-zero otherwise
int32_t spvOpcodeGeneratesType(Op op);

#endif
