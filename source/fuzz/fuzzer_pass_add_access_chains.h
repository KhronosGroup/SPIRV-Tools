// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_ACCESS_CHAINS_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_ACCESS_CHAINS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Fuzzer pass that randomly adds access chains based on pointers available in
// the module.  Other passes can use these access chains, e.g. by loading from
// them.
class FuzzerPassAddAccessChains : public FuzzerPass {
 public:
  FuzzerPassAddAccessChains(opt::IRContext* ir_context,
                            TransformationContext* transformation_context,
                            FuzzerContext* fuzzer_context,
                            protobufs::TransformationSequence* transformations);

  ~FuzzerPassAddAccessChains();

  void Apply() override;

 private:
  // Returns the id of a 32-bit integer constant of value |value|, which is
  // randomly chosen to be signed or unsigned. If such constant already exists
  // in the module, it is found, otherwise a new one is created.
  // Adds instructions to the module so that clamping can be correctly
  // performed. In particular, it makes sure (by adding them if not present)
  // that the module has:
  // - An OpTypeBool instruction
  // - An OpConstant of the same type as the one whose id is returned, and value
  //   |bound| - 1
  // Adds a pair of fresh ids to |fresh_ids_for_clamping|.
  uint32_t FindOrCreateIntegerConstantReadyForClamping(
      uint32_t value, uint32_t bound,
      std::vector<std::pair<uint32_t, uint32_t>>* fresh_ids_for_clamping);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_ACCESS_CHAINS_H_
