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
#include <memory>
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
#include "spirv_validator_options.h"
#include "validate.h"
#include "val/instruction.h"
#include "val/validation_state.h"

using libspirv::Instruction;
using libspirv::SpirvStats;
using libspirv::ValidationState_t;

namespace {

// Helper class for stats aggregation. Receives as in/out parameter.
// Constructs ValidationState and updates it by running validator for each
// instruction.
class StatsAggregator {
 public:
  StatsAggregator(SpirvStats* in_out_stats, const spv_const_context context) {
    stats_ = in_out_stats;
    vstate_.reset(new ValidationState_t(context, &validator_options_));
  }

  // Collects header statistics and sets correct id_bound.
  spv_result_t ProcessHeader(
      spv_endianness_t /* endian */, uint32_t /* magic */,
      uint32_t version, uint32_t generator, uint32_t id_bound,
      uint32_t /* schema */) {
    vstate_->setIdBound(id_bound);
    ++stats_->version_hist[version];
    ++stats_->generator_hist[generator];
    return SPV_SUCCESS;
  }

  // Runs validator to validate the instruction and update vstate_,
  // then procession the instruction to collect stats.
  spv_result_t ProcessInstruction(const spv_parsed_instruction_t* inst) {
    const spv_result_t validation_result =
        spvtools::ValidateInstructionAndUpdateValidationState(vstate_.get(), inst);
    if (validation_result != SPV_SUCCESS)
      return validation_result;

    ProcessOpcode();
    ProcessCapability();
    ProcessExtension();

    return SPV_SUCCESS;
  }

  // Collects OpCapability statistics.
  void ProcessCapability() {
    const Instruction& inst = GetCurrentInstruction();
    if (inst.opcode() != SpvOpCapability) return;
    const uint32_t capability = inst.word(inst.operands()[0].offset);
    ++stats_->capability_hist[capability];
  }

  // Collects OpExtension statistics.
  void ProcessExtension() {
    const Instruction& inst = GetCurrentInstruction();
    if (inst.opcode() != SpvOpExtension) return;
    const std::string extension = libspirv::GetExtensionString(&inst.c_inst());
    ++stats_->extension_hist[extension];
  }

  // Collects OpCode statistics.
  void ProcessOpcode() {
    auto inst_it = vstate_->ordered_instructions().rbegin();
    const SpvOp opcode = inst_it->opcode();
    ++stats_->opcode_hist[opcode];

    ++inst_it;
    auto step_it = stats_->opcode_markov_hist.begin();
    for (; inst_it != vstate_->ordered_instructions().rend() &&
         step_it != stats_->opcode_markov_hist.end(); ++inst_it, ++step_it) {
      auto& hist = (*step_it)[inst_it->opcode()];
      ++hist[opcode];
    }
  }

  SpirvStats* stats() {
    return stats_;
  }

 private:
  // Returns the current instruction (the one last processed by the validator).
  const Instruction& GetCurrentInstruction() const {
    return vstate_->ordered_instructions().back();
  }

  SpirvStats* stats_;
  spv_validator_options_t validator_options_;
  std::unique_ptr<ValidationState_t> vstate_;
};

spv_result_t ProcessHeader(
    void* user_data, spv_endianness_t endian, uint32_t magic,
    uint32_t version, uint32_t generator, uint32_t id_bound,
    uint32_t schema) {
  StatsAggregator* stats_aggregator =
      reinterpret_cast<StatsAggregator*>(user_data);
  return stats_aggregator->ProcessHeader(
      endian, magic, version, generator, id_bound, schema);
}

spv_result_t ProcessInstruction(
    void* user_data, const spv_parsed_instruction_t* inst) {
  StatsAggregator* stats_aggregator =
      reinterpret_cast<StatsAggregator*>(user_data);
  return stats_aggregator->ProcessInstruction(inst);
}

}  // namespace

namespace libspirv {

spv_result_t AggregateStats(
    const spv_context_t& context, const uint32_t* words, const size_t num_words,
    spv_diagnostic* pDiagnostic, SpirvStats* stats) {
  spv_const_binary_t binary = {words, num_words};

  spv_endianness_t endian;
  spv_position_t position = {};
  if (spvBinaryEndianness(&binary, &endian)) {
    return libspirv::DiagnosticStream(position, context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
        << "Invalid SPIR-V magic number.";
  }

  spv_header_t header;
  if (spvBinaryHeaderGet(&binary, endian, &header)) {
    return libspirv::DiagnosticStream(position, context.consumer,
                                      SPV_ERROR_INVALID_BINARY)
        << "Invalid SPIR-V header.";
  }

  StatsAggregator stats_aggregator(stats, &context);

  return spvBinaryParse(&context, &stats_aggregator, words, num_words,
                        ProcessHeader, ProcessInstruction, pDiagnostic);
}

}  // namespace libspirv
