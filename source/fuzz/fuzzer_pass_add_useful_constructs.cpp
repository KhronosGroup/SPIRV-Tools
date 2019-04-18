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

#include "source/fuzz/fuzzer_pass_add_useful_constructs.h"
#include "source/fuzz/transformation_add_boolean_constant.h"

namespace spvtools {
namespace fuzz {

using opt::IRContext;

void FuzzerPassAddUsefulConstructs::Apply(
    IRContext* ir_context, FuzzerContext* fuzzer_context,
    std::vector<std::unique_ptr<Transformation>>* transformations) {
  // Add OpConstantTrue if it is not already there.
  auto make_true = MakeUnique<TransformationAddBooleanConstant>(
      fuzzer_context->FreshId(), true);
  if (make_true->IsApplicable(ir_context)) {
    make_true->Apply(ir_context);
    transformations->push_back(std::move(make_true));
  }

  // Add OpConstantFalse if it is not already there.
  auto make_false = MakeUnique<TransformationAddBooleanConstant>(
      fuzzer_context->FreshId(), false);
  if (make_false->IsApplicable(ir_context)) {
    make_false->Apply(ir_context);
    transformations->push_back(std::move(make_false));
  }
}

}  // namespace fuzz
}  // namespace spvtools
