// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "validate.h"

#include <cassert>
#include <cstdio>

#include <algorithm>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "binary.h"
#include "diagnostic.h"
#include "instruction.h"
#include "opcode.h"
#include "operand.h"
#include "spirv-tools/libspirv.h"
#include "spirv_constant.h"
#include "spirv_endian.h"
#include "val/Construct.h"
#include "val/Function.h"
#include "val/ValidationState.h"

using std::function;
using std::ostream_iterator;
using std::placeholders::_1;
using std::string;
using std::stringstream;
using std::transform;
using std::vector;

using libspirv::CfgPass;
using libspirv::InstructionPass;
using libspirv::ModuleLayoutPass;
using libspirv::IdPass;
using libspirv::ValidationState_t;

spv_result_t spvValidateIDs(const spv_instruction_t* pInsts,
                            const uint64_t count,
                            const spv_opcode_table opcodeTable,
                            const spv_operand_table operandTable,
                            const spv_ext_inst_table extInstTable,
                            const ValidationState_t& state,
                            spv_position position) {
  position->index = SPV_INDEX_INSTRUCTION;
  if (auto error =
          spvValidateInstructionIDs(pInsts, count, opcodeTable, operandTable,
                                    extInstTable, state, position))
    return error;
  return SPV_SUCCESS;
}

namespace {

// TODO(umar): Validate header
// TODO(umar): The Id bound should be validated also. But you can only do that
// after you've seen all the instructions in the module.
// TODO(umar): The binary parser validates the magic word, and the length of the
// header, but nothing else.
spv_result_t setHeader(void* user_data, spv_endianness_t endian, uint32_t magic,
                       uint32_t version, uint32_t generator, uint32_t id_bound,
                       uint32_t reserved) {
  (void)user_data;
  (void)endian;
  (void)magic;
  (void)version;
  (void)generator;
  (void)id_bound;
  (void)reserved;
  return SPV_SUCCESS;
}

// Improves diagnostic messages by collecting names of IDs
// NOTE: This function returns void and is not involved in validation
void DebugInstructionPass(ValidationState_t& _,
                          const spv_parsed_instruction_t* inst) {
  switch (inst->opcode) {
    case SpvOpName: {
      const uint32_t target = *(inst->words + inst->operands[0].offset);
      const char* str =
          reinterpret_cast<const char*>(inst->words + inst->operands[1].offset);
      _.AssignNameToId(target, str);
    } break;
    case SpvOpMemberName: {
      const uint32_t target = *(inst->words + inst->operands[0].offset);
      const char* str =
          reinterpret_cast<const char*>(inst->words + inst->operands[2].offset);
      _.AssignNameToId(target, str);
    } break;
    case SpvOpSourceContinued:
    case SpvOpSource:
    case SpvOpSourceExtension:
    case SpvOpString:
    case SpvOpLine:
    case SpvOpNoLine:

    default:
      break;
  }
}

spv_result_t ProcessInstruction(void* user_data,
                                const spv_parsed_instruction_t* inst) {
  ValidationState_t& _ = *(reinterpret_cast<ValidationState_t*>(user_data));
  _.increment_instruction_count();
  if (static_cast<SpvOp>(inst->opcode) == SpvOpEntryPoint)
    _.entry_points().push_back(inst->words[2]);

  DebugInstructionPass(_, inst);
  // TODO(umar): Perform data rules pass
  if (auto error = IdPass(_, inst)) return error;
  if (auto error = ModuleLayoutPass(_, inst)) return error;
  if (auto error = CfgPass(_, inst)) return error;
  if (auto error = InstructionPass(_, inst)) return error;

  return SPV_SUCCESS;
}

void printDot(const ValidationState_t& _, const libspirv::BasicBlock& other) {
  string block_string;
  if (other.successors()->empty()) {
    block_string += "end ";
  } else {
    for (auto block : *other.successors()) {
      block_string += _.getIdOrName(block->id()) + " ";
    }
  }
  printf("%10s -> {%s\b}\n", _.getIdOrName(other.id()).c_str(),
         block_string.c_str());
}

void PrintBlocks(ValidationState_t& _, libspirv::Function func) {
  assert(func.first_block());

  printf("%10s -> %s\n", _.getIdOrName(func.id()).c_str(),
         _.getIdOrName(func.first_block()->id()).c_str());
  for (const auto& block : func.ordered_blocks()) {
    printDot(_, *block);
  }
}

#ifdef __clang__
#define UNUSED(func) [[gnu::unused]] func
#elif defined(__GNUC__)
#define UNUSED(func)            \
  func __attribute__((unused)); \
  func
#elif defined(_MSC_VER)
#define UNUSED(func) func
#endif

UNUSED(void PrintDotGraph(ValidationState_t& _, libspirv::Function func)) {
  if (func.first_block()) {
    string func_name(_.getIdOrName(func.id()));
    printf("digraph %s {\n", func_name.c_str());
    PrintBlocks(_, func);
    printf("}\n");
  }
}
}  // anonymous namespace

spv_result_t spvValidate(const spv_const_context context,
                         const spv_const_binary binary,
                         spv_diagnostic* pDiagnostic) {
  return spvValidateBinary(context, binary->code, binary->wordCount,
                           pDiagnostic);
}
spv_result_t spvValidateBinary(const spv_const_context context,
                               const uint32_t* words, const size_t num_words,
                               spv_diagnostic* pDiagnostic) {
  spv_context_t hijack_context = *context;

  spv_const_binary binary = new spv_const_binary_t{words, num_words};
  if (pDiagnostic) {
    *pDiagnostic = nullptr;
    libspirv::UseDiagnosticAsMessageConsumer(&hijack_context, pDiagnostic);
  }

  spv_endianness_t endian;
  spv_position_t position = {};
  if (spvBinaryEndianness(binary, &endian)) {
    return libspirv::DiagnosticStream(position, hijack_context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
           << "Invalid SPIR-V magic number.";
  }

  spv_header_t header;
  if (spvBinaryHeaderGet(binary, endian, &header)) {
    return libspirv::DiagnosticStream(position, hijack_context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
           << "Invalid SPIR-V header.";
  }

  // NOTE: Parse the module and perform inline validation checks. These
  // checks do not require the the knowledge of the whole module.
  ValidationState_t vstate(&hijack_context);
  if (auto error = spvBinaryParse(&hijack_context, &vstate, words, num_words,
                                  setHeader, ProcessInstruction, pDiagnostic))
    return error;

  if (vstate.in_function_body())
    return vstate.diag(SPV_ERROR_INVALID_LAYOUT)
           << "Missing OpFunctionEnd at end of module.";

  // TODO(umar): Add validation checks which require the parsing of the entire
  // module. Use the information from the ProcessInstruction pass to make the
  // checks.
  if (vstate.unresolved_forward_id_count() > 0) {
    stringstream ss;
    vector<uint32_t> ids = vstate.UnresolvedForwardIds();

    transform(begin(ids), end(ids), ostream_iterator<string>(ss, " "),
              bind(&ValidationState_t::getIdName, std::ref(vstate), _1));

    auto id_str = ss.str();
    return vstate.diag(SPV_ERROR_INVALID_ID)
           << "The following forward referenced IDs have not be defined:\n"
           << id_str.substr(0, id_str.size() - 1);
  }

  // CFG checks are performed after the binary has been parsed
  // and the CFGPass has collected information about the control flow
  if (auto error = PerformCfgChecks(vstate)) return error;
  if (auto error = UpdateIdUse(vstate)) return error;
  if (auto error = CheckIdDefinitionDominateUse(vstate)) return error;

  // NOTE: Copy each instruction for easier processing
  std::vector<spv_instruction_t> instructions;
  uint64_t index = SPV_INDEX_INSTRUCTION;
  while (index < binary->wordCount) {
    uint16_t wordCount;
    uint16_t opcode;
    spvOpcodeSplit(spvFixWord(binary->code[index], endian), &wordCount,
                   &opcode);
    spv_instruction_t inst;
    spvInstructionCopy(&binary->code[index], static_cast<SpvOp>(opcode),
                       wordCount, endian, &inst);
    instructions.push_back(inst);
    index += wordCount;
  }

  position.index = SPV_INDEX_INSTRUCTION;
  return spvValidateIDs(instructions.data(), instructions.size(),
                        hijack_context.opcode_table,
                        hijack_context.operand_table,
                        hijack_context.ext_inst_table, vstate, &position);
}
