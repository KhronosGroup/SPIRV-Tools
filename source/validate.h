// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef LIBSPIRV_VALIDATE_H_
#define LIBSPIRV_VALIDATE_H_

#include <algorithm>
#include <array>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "assembly_grammar.h"
#include "binary.h"
#include "diagnostic.h"
#include "instruction.h"
#include "spirv-tools/libspirv.h"
#include "spirv_definition.h"
#include "table.h"
#include "val/BasicBlock.h"

// Structures

namespace libspirv {

class ValidationState_t;

/// @brief Calculates dominator edges of a root basic block
///
/// This function calculates the dominator edges form a root BasicBlock. Uses
/// the dominator algorithm by Cooper et al.
///
/// @param[in] first_block the root or entry BasicBlock of a function
///
/// @return a set of dominator edges represented as a pair of blocks
std::vector<std::pair<BasicBlock*, BasicBlock*>> CalculateDominators(
    const BasicBlock& first_block);

/// @brief Performs the Control Flow Graph checks
///
/// @param[in] _ the validation state of the module
///
/// @return SPV_SUCCESS if no errors are found. SPV_ERROR_INVALID_CFG otherwise
spv_result_t PerformCfgChecks(ValidationState_t& _);

/// @brief Updates the immediate dominator for each of the block edges
///
/// Updates the immediate dominator of the blocks for each of the edges
/// provided by the @p dom_edges parameter
///
/// @param[in,out] dom_edges The edges of the dominator tree
void UpdateImmediateDominators(
    std::vector<std::pair<BasicBlock*, BasicBlock*>>& dom_edges);

/// @brief Prints all of the dominators of a BasicBlock
///
/// @param[in] block The dominators of this block will be printed
void printDominatorList(BasicBlock& block);

/// Performs logical layout validation as described in section 2.4 of the SPIR-V
/// spec.
spv_result_t ModuleLayoutPass(ValidationState_t& _,
                              const spv_parsed_instruction_t* inst);

/// Performs Control Flow Graph validation of a module
spv_result_t CfgPass(ValidationState_t& _,
                     const spv_parsed_instruction_t* inst);

/// Performs SSA validation of a module
spv_result_t SsaPass(ValidationState_t& _,
                     const spv_parsed_instruction_t* inst);

/// Performs instruction validation.
spv_result_t InstructionPass(ValidationState_t& _,
                             const spv_parsed_instruction_t* inst);

}  // namespace libspirv

/// @brief Validate the ID usage of the instruction stream
///
/// @param[in] pInsts stream of instructions
/// @param[in] instCount number of instructions
/// @param[in] opcodeTable table of specified Opcodes
/// @param[in] operandTable table of specified operands
/// @param[in] usedefs use-def info from module parsing
/// @param[in,out] position current position in the stream
/// @param[out] pDiag contains diagnostic on failure
///
/// @return result code
spv_result_t spvValidateInstructionIDs(const spv_instruction_t* pInsts,
                                       const uint64_t instCount,
                                       const spv_opcode_table opcodeTable,
                                       const spv_operand_table operandTable,
                                       const spv_ext_inst_table extInstTable,
                                       const libspirv::ValidationState_t& state,
                                       spv_position position,
                                       spv_diagnostic* pDiag);

/// @brief Validate the ID's within a SPIR-V binary
///
/// @param[in] pInstructions array of instructions
/// @param[in] count number of elements in instruction array
/// @param[in] bound the binary header
/// @param[in] opcodeTable table of specified Opcodes
/// @param[in] operandTable table of specified operands
/// @param[in,out] position current word in the binary
/// @param[out] pDiagnostic contains diagnostic on failure
///
/// @return result code
spv_result_t spvValidateIDs(const spv_instruction_t* pInstructions,
                            const uint64_t count, const uint32_t bound,
                            const spv_opcode_table opcodeTable,
                            const spv_operand_table operandTable,
                            const spv_ext_inst_table extInstTable,
                            spv_position position, spv_diagnostic* pDiagnostic);

#define spvCheckReturn(expression) \
  if (spv_result_t error = (expression)) return error;

#endif  // LIBSPIRV_VALIDATE_H_
