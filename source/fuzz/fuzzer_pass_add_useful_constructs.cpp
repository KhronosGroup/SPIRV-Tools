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
#include "source/fuzz/transformation_add_constant_boolean.h"

namespace spvtools {
namespace fuzz {

using opt::IRContext;

void FuzzerPassAddUsefulConstructs::Apply(
    IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations) {
  // Add OpConstantTrue if it is not already there.
  protobufs::TransformationAddConstantBoolean make_true;
  make_true.set_fresh_id(fuzzer_context->FreshId());
  make_true.set_is_true(true);
  if (transformation::IsApplicable(make_true, ir_context, *fact_manager)) {
    transformation::Apply(make_true, ir_context, fact_manager);
    *transformations->add_transformations()->mutable_add_boolean_constant() =
        make_true;
  }

  // Add OpConstantFalse if it is not already there.
  protobufs::TransformationAddConstantBoolean make_false;
  make_false.set_fresh_id(fuzzer_context->FreshId());
  make_false.set_is_true(false);
  if (transformation::IsApplicable(make_false, ir_context, *fact_manager)) {
    transformation::Apply(make_false, ir_context, fact_manager);
    *transformations->add_transformations()->mutable_add_boolean_constant() =
        make_false;
  }
}

}  // namespace fuzz
}  // namespace spvtools
