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

#ifndef LIBSPIRV_SPIRV_STATS_H_
#define LIBSPIRV_SPIRV_STATS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "spirv-tools/libspirv.hpp"

namespace libspirv {

struct SpirvStats {
  // Version histogram, version_word -> count.
  std::unordered_map<uint32_t, uint32_t> version_hist;

  // Generator histogram, generator_word -> count.
  std::unordered_map<uint32_t, uint32_t> generator_hist;

  // Capability histogram, SpvCapabilityXXX -> count.
  std::unordered_map<uint32_t, uint32_t> capability_hist;

  // Extension histogram, extension_string -> count.
  std::unordered_map<std::string, uint32_t> extension_hist;

  // Opcode histogram, SpvOpXXX -> count.
  std::unordered_map<uint32_t, uint32_t> opcode_hist;
};

// Aggregates existing |stats| with new stats extracted from |binary|.
spv_result_t AggregateStats(
    const spv_context_t& context, const uint32_t* words, const size_t num_words,
    spv_diagnostic* pDiagnostic, SpirvStats* stats);

}  // namespace libspirv

#endif  // LIBSPIRV_SPIRV_STATS_H_
