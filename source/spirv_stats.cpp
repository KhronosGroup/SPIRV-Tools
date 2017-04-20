// Copyright (c) 2017 Google Inc.
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

#include "spirv_stats.h"

#include <cassert>

#include <algorithm>
#include <string>
#include <vector>

#include "binary.h"
#include "diagnostic.h"
#include "enum_string_mapping.h"
#include "extensions.h"
#include "instruction.h"
#include "opcode.h"
#include "operand.h"
#include "spirv-tools/libspirv.h"
#include "spirv_endian.h"

using libspirv::SpirvStats;

namespace {

struct StatsContext {
  SpirvStats* stats;

  // Opcodes of already processed instructions in the order as they appear in
  // the module.
  std::vector<uint32_t> opcodes;
};

// Collects statistics from SPIR-V header (version, generator).
spv_result_t ProcessHeader(
    void* user_data, spv_endianness_t /* endian */, uint32_t /* magic */,
    uint32_t version, uint32_t generator, uint32_t /* id_bound */,
    uint32_t /* schema */) {
  StatsContext* ctx = reinterpret_cast<StatsContext*>(user_data);
  ++ctx->stats->version_hist[version];
  ++ctx->stats->generator_hist[generator];
  return SPV_SUCCESS;
}

// Collects OpCapability statistics.
void ProcessCapability(StatsContext* ctx,
                       const spv_parsed_instruction_t* inst) {
  if (static_cast<SpvOp>(inst->opcode) != SpvOpCapability) return;
  assert(inst->num_operands == 1);
  const spv_parsed_operand_t& operand = inst->operands[0];
  assert(operand.num_words == 1);
  assert(operand.offset < inst->num_words);
  const uint32_t capability = inst->words[operand.offset];
  ++ctx->stats->capability_hist[capability];
}

// Collects OpExtension statistics.
void ProcessExtension(StatsContext* ctx,
                      const spv_parsed_instruction_t* inst) {
  if (static_cast<SpvOp>(inst->opcode) != SpvOpExtension) return;
  const std::string extension = libspirv::GetExtensionString(inst);
  ++ctx->stats->extension_hist[extension];
}

// Collects OpCode statistics.
void ProcessOpcode(StatsContext* ctx,
                   const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  ++ctx->stats->opcode_hist[opcode];

  auto opcode_it = ctx->opcodes.rbegin();
  auto step_it = ctx->stats->opcode_markov_hist.begin();
  for (; opcode_it != ctx->opcodes.rend() &&
       step_it != ctx->stats->opcode_markov_hist.end();
       ++opcode_it, ++step_it) {
    auto& hist = (*step_it)[*opcode_it];
    ++hist[opcode];
  }
}

// Collects opcode usage statistics and calls other collectors.
spv_result_t ProcessInstruction(
    void* user_data, const spv_parsed_instruction_t* inst) {
  StatsContext* ctx = reinterpret_cast<StatsContext*>(user_data);

  ProcessOpcode(ctx, inst);
  ProcessCapability(ctx, inst);
  ProcessExtension(ctx, inst);

  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  ctx->opcodes.push_back(opcode);

  return SPV_SUCCESS;
}

}  // namespace

namespace libspirv {

spv_result_t AggregateStats(
    const spv_context_t& spv_context, const uint32_t* words, const size_t num_words,
    spv_diagnostic* pDiagnostic, SpirvStats* stats) {
  spv_const_binary_t binary = {words, num_words};

  spv_endianness_t endian;
  spv_position_t position = {};
  if (spvBinaryEndianness(&binary, &endian)) {
    return libspirv::DiagnosticStream(position, spv_context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
        << "Invalid SPIR-V magic number.";
  }

  spv_header_t header;
  if (spvBinaryHeaderGet(&binary, endian, &header)) {
    return libspirv::DiagnosticStream(position, spv_context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
        << "Invalid SPIR-V header.";
  }

  StatsContext stats_context;
  stats_context.stats = stats;

  return spvBinaryParse(&spv_context, &stats_context, words, num_words,
                        ProcessHeader, ProcessInstruction, pDiagnostic);
}

}  // namespace libspirv
