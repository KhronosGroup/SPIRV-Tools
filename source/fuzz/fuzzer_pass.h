// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_H_
#define SOURCE_FUZZ_FUZZER_PASS_H_

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

// Interface for applying a pass of transformations to a module.
class FuzzerPass {
 public:
  FuzzerPass() = default;

  virtual ~FuzzerPass() = default;

  // Applies the pass to the given module, |ir_context|, assuming and updating
  // facts from |fact_manager|, and using |fuzzer_context| to guide the process.
  // Appends to |transformations| all transformations that were applied during
  // the pass.
  virtual void Apply(
      opt::IRContext* ir_context, FactManager* fact_manager,
      FuzzerContext* fuzzer_context,
      std::vector<std::unique_ptr<Transformation>>* transformations) = 0;

 private:
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // #define SOURCE_FUZZ_FUZZER_PASS_H_
