// Copyright (c) 2020 Vasyl Teliman
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_PARAMETERS_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_PARAMETERS_H_

#include <vector>

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Randomly decides for each non-entry-point function in the module whether to
// add one new parameter to it. If so, randomly determines parameter's type and
// creates a constant to initialize it.
class FuzzerPassAddParameters : public FuzzerPass {
 public:
  FuzzerPassAddParameters(opt::IRContext* ir_context,
                          TransformationContext* transformation_context,
                          FuzzerContext* fuzzer_context,
                          protobufs::TransformationSequence* transformations);

  ~FuzzerPassAddParameters() override;

  void Apply() override;

 private:
  // Uses types, defined in the module, to compute a vector of their ids, which
  // will be used as type ids of new parameters.
  std::vector<uint32_t> ComputeTypeCandidates() const;

  // Returns number of parameters of |function|.
  uint32_t GetNumberOfParameters(const opt::Function& function) const;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_PARAMETERS_H_
