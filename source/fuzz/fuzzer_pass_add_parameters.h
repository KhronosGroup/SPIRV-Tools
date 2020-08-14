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
// add new parameters to it. If so, randomly determines the number of parameters
// to add, their type and creates constants used to initialize them.
class FuzzerPassAddParameters : public FuzzerPass {
 public:
  FuzzerPassAddParameters(opt::IRContext* ir_context,
                          TransformationContext* transformation_context,
                          FuzzerContext* fuzzer_context,
                          protobufs::TransformationSequence* transformations);

  ~FuzzerPassAddParameters() override;

  void Apply() override;

 private:
  // Returns number of parameters of |function|.
  uint32_t GetNumberOfParameters(const opt::Function& function) const;

  // Returns the id of an available local variable with the fact
  // PointeeValueIsIrrelevant. If there is no such variable, it creates one in
  // the |function| adding a zero initializer constant that is irrelevant. The
  // variable may be irrelevant itself according to
  // |pointee_value_is_irrelevant|. The function returns the id of the created
  // variable.
  uint32_t FindOrCreateLocalVariable(uint32_t pointer_type_id,
                                     uint32_t function_id,
                                     bool pointee_value_is_irrelevant);

  // Returns the id of an available global variable with the fact
  // PointeeValueIsIrrelevant. If there is no such variable, it creates one in
  // adding a zero initializer constant that is irrelevant. The variable may be
  // irrelevant itself according to |pointee_value_is_irrelevant|. The function
  // returns the id of the created variable.
  uint32_t FindOrCreateGlobalVariable(uint32_t pointer_type_id,
                                      bool pointee_value_is_irrelevant);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_PARAMETERS_H_
