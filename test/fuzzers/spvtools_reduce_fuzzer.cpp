// Copyright (c) 2021 Google LLC
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

#include <cstdint>
#include <vector>

#include "source/reduce/reducer.h"
#include "spirv-tools/libspirv.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size == 0 || (size % sizeof(uint32_t)) != 0) {
    // An empty binary, or a binary whose size is not a multiple of word-size,
    // cannot be valid, so can be rejected immediately.
    return 0;
  }

  std::vector<uint32_t> initial_binary(size / sizeof(uint32_t));
  memcpy(initial_binary.data(), data, size);

  spvtools::reduce::Reducer reducer(SPV_ENV_UNIVERSAL_1_3);
  reducer.SetMessageConsumer([](spv_message_level_t, const char*,
                                const spv_position_t&, const char*) {});
  reducer.AddDefaultReductionPasses();
  reducer.SetInterestingnessFunction(
      [](const std::vector<uint32_t>& binary, uint32_t counter) -> bool {
        if (counter == 0) {
          // This ensures that the SPIR-V binary is always regarded as
          // interesting, initially.
          return true;
        }
        if (binary.empty()) {
          return false;
        }
        // Decide whether the binary is interesting based on a function of the
        // integer in the middle of the binary, and the counter passed into the
        // interestingness function. The intent is that "true" and "false" will
        // be returned with roughly equal probability, but that the sequence of
        // "true" and "false" values will vary depending on the binary being
        // reduced.
        return ((binary[binary.size() / 2] + counter) % 2) == 0;
      });

  std::vector<uint32_t> binary_out;
  spvtools::ReducerOptions reducer_options;
  spvtools::ValidatorOptions validator_options;
  reducer.Run(initial_binary, &binary_out, reducer_options, validator_options);
  return 0;
}
