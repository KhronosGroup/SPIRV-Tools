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

#ifndef _CODEPLAY_SPIRV_OPERAND_H_
#define _CODEPLAY_SPIRV_OPERAND_H_

#include <deque>

#include <libspirv/libspirv.h>

/// @brief A sequence of operand types.
///
/// A SPIR-V parser uses an operand pattern to describe what is expected
/// next on the input.
///
/// As we parse an instruction in text or binary form from left to right,
/// we pull and push from the front of the pattern.
using spv_operand_pattern_t = std::deque<spv_operand_type_t>;

/// @brief Find the named operand in the table
///
/// @param[in] table to lookup
/// @param[in] type the operand group's type
/// @param[in] name of the operand to find
/// @param[in] nameLength number of bytes of name to compare
/// @param[out] pEntry returned operand table entry
///
/// @return result code
spv_result_t spvOperandTableNameLookup(const spv_operand_table table,
                                       const spv_operand_type_t type,
                                       const char *name,
                                       const size_t nameLength,
                                       spv_operand_desc *pEntry);

/// @brief Find the operand with value in the table
///
/// @param[in] table to lookup
/// @param[in] type the operand group's type
/// @param[in] value of the operand to find
/// @param[out] pEntry return operand table entry
///
/// @return result code
spv_result_t spvOperandTableValueLookup(const spv_operand_table table,
                                        const spv_operand_type_t type,
                                        const uint32_t value,
                                        spv_operand_desc *pEntry);

/// @brief Get the name string of the operand type
///
/// @param type the type of the operand
///
/// @return the string name of the operand
const char *spvOperandTypeStr(spv_operand_type_t type);

/// @brief Returns true if an operand of the given type is optional.
///
/// @param[in] type The operand type
///
/// @return bool
bool spvOperandIsOptional(spv_operand_type_t type);

/// @brief Returns true if an operand type represents zero or more
/// logical operands.
///
/// Note that a single logical operand may still be a variable number
/// of words.  For example, a literal string may be many words, but
/// is just one logical operand.
///
/// @param[in] type The operand type
///
/// @return bool
bool spvOperandIsVariable(spv_operand_type_t type);

/// @brief Inserts a list of operand types into the front of the given pattern.
///
/// @param[in] types source array of types, ending with SPV_OPERAND_TYPE_NONE.
/// @param[in,out] pattern the destination sequence
void spvPrependOperandTypes(const spv_operand_type_t *types,
                            spv_operand_pattern_t *pattern);

/// @brief Inserts the operands expected after the given typed mask onto the
/// front of the given pattern.
///
/// Each set bit in the mask represents zero or more operand types that should
/// be prepended onto the pattern.  Operands for a less significant bit always
/// appear before operands for a more significant bit.
///
/// If the a set bit is unknown, then we assume it has no operands.
///
/// @param[in] operandTable the table of operand type definitions
/// @param[in] type the type of operand
/// @param[in] mask the mask value for the given type
/// @param[in,out] pattern the destination sequence of operand types
void spvPrependOperandTypesForMask(const spv_operand_table operandTable,
                                   const spv_operand_type_t type,
                                   const uint32_t mask,
                                   spv_operand_pattern_t *pattern);

/// @brief Expands an operand type representing zero or more logical operands,
/// exactly once.
///
/// If the given type represents potentially several logical operands,
/// then prepend the given pattern with the first expansion of the logical
/// operands, followed by original type.  Otherwise, don't modify the pattern.
///
/// For example, the SPV_OPERAND_TYPE_VARIABLE_ID represents zero or more
/// IDs.  In that case we would prepend the pattern with SPV_OPERAND_TYPE_ID
/// followed by SPV_OPERAND_TYPE_VARIABLE_ID again.
///
/// This also applies to zero or more tuples of logical operands.  In that case
/// we prepend pattern with for the members of the tuple, followed by the
/// original type argument.  The pattern must encode the fact that if any part
/// of the tuple is present, then all tuple members should be.  So the first
/// member of the tuple must be optional, and the remaining members
/// non-optional.
///
/// @param [in] type an operand type, maybe representing a sequence of operands
/// @param [in,out] pattern the list of operand types
///
/// @return true if we modified the pattern
bool spvExpandOperandSequenceOnce(spv_operand_type_t type,
                                  spv_operand_pattern_t* pattern);

/// Expands the first element in the pattern until it is a matchable operand
/// type, then pops it off the front and returns it.  The pattern must not be
/// empty.
///
/// A matchable operand type is anything other than a zero-or-more-items
/// operand type.
spv_operand_type_t spvTakeFirstMatchableOperand(spv_operand_pattern_t* pattern);

/// Switches *pExpectedOperands to the post-immediate alternate pattern, which
/// allows a limited set of operand types.
void spvSwitchToAlternateParsingAfterImmediate(
    spv_operand_pattern_t *pExpectedOperands);

#endif
