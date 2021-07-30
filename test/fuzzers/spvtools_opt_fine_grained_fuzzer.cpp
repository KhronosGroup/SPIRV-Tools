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
#include <string>
#include <vector>

#include "spirv-tools/optimizer.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size == 0 || (size % sizeof(uint32_t)) != 0) {
    // An empty binary, or a binary whose size is not a multiple of word-size,
    // cannot be valid, so can be rejected immediately.
    return 0;
  }

  std::vector<uint32_t> initial_binary(size / sizeof(uint32_t));
  memcpy(initial_binary.data(), data, size);

  // A combination of the following optimizer passes will be selected.
  std::vector<std::string> opt_passes({"--combine-access-chains",
                                       "--loop-unroll",
                                       "--merge-blocks",
                                       "--cfg-cleanup",
                                       "--eliminate-dead-functions",
                                       "--merge-return",
                                       "--wrap-opkill",
                                       "--eliminate-dead-code-aggressive",
                                       "--if-conversion",
                                       "--eliminate-local-single-store",
                                       "--eliminate-local-single-block",
                                       "--eliminate-dead-branches",
                                       "--scalar-replacement=0",
                                       "--eliminate-dead-inserts",
                                       "--eliminate-dead-members",
                                       "--simplify-instructions",
                                       "--private-to-local",
                                       "--ssa-rewrite",
                                       "--ccp",
                                       "--reduce-load-size",
                                       "--vector-dce",
                                       "--scalar-replacement=100",
                                       "--inline-entry-points-exhaustive",
                                       "--redundancy-elimination",
                                       "--convert-local-access-chains",
                                       "--copy-propagate-arrays",
                                       "--fix-storage-class"});

  const uint32_t kNumPasses = 5;

  // A series of passes are selected in a deterministic fashion, as a function
  // of the contents of the binary. The aim is to achieve diverse coverage of
  // optimization passes across different binaries, while avoiding
  // randomization, which would make bugs found by the fuzzer hard to reproduce.
  std::vector<std::string> selected_passes;
  for (uint32_t count = 0; count < kNumPasses; count++) {
    // Choose an index into the binary, starting from its midpoint, adding the
    // loop counter, and using modular arithmetic to ensure it is in-bounds.
    const uint32_t index_into_binary =
        (initial_binary.size() / 2 + count) % initial_binary.size();
    // Choose an index into the set of available passes by reading from the
    // binary at the chosen index, adding the loop counter, and using modular
    // arithmetic to ensure that the pass index is in-bounds.
    const uint32_t index_into_opt_passes =
        (initial_binary[index_into_binary] + count) % opt_passes.size();
    // Select a pass based on this index.
    selected_passes.push_back(opt_passes[index_into_opt_passes]);
  }
  std::vector<uint32_t> optimized_binary;
  spvtools::Optimizer optimizer(SPV_ENV_UNIVERSAL_1_3);
  optimizer.SetMessageConsumer([](spv_message_level_t, const char*,
                                  const spv_position_t&, const char*) {});
  optimizer.SetValidateAfterAll(true);
  optimizer.RegisterPassesFromFlags(selected_passes);
  optimizer.Run(initial_binary.data(), initial_binary.size(),
                &optimized_binary);
  return 0;
}
