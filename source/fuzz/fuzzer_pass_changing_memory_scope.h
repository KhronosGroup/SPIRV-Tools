// Copyright (c) 2021 Mostafa Ashraf
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_CHANGING_MEMORY_SCOPE_H_
#define SOURCE_FUZZ_FUZZER_PASS_CHANGING_MEMORY_SCOPE_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// This fuzzer pass changes the memory scope for atomic or barrier instruction
// with a new one.
class FuzzerPassChangingMemoryScope : public FuzzerPass {
 public:
  FuzzerPassChangingMemoryScope(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  std::vector<SpvScope> GetSuitableNewMemoryScope(
      SpvScope old_memory_scope_value);

  void Apply() override;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_CHANGING_MEMORY_SCOPE_H_
