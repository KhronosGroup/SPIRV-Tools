// Copyright (c) 2020 Stefano Milizia
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

#include "source/fuzz/fuzzer_pass_toggle_constant_null.h"

#include "source/fuzz/transformation_toggle_constant_null.h"

namespace spvtools {
namespace fuzz {

FuzzerPassToggleConstantNull::FuzzerPassToggleConstantNull(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassToggleConstantNull::~FuzzerPassToggleConstantNull() = default;

void FuzzerPassToggleConstantNull::Apply() {
  // Consider every constant declaration
  for (auto& declaration : GetIRContext()->GetConstants()) {
    // Ignore non-scalar and non-null constants
    if (declaration->opcode() != SpvOpConstant &&
        declaration->opcode() != SpvOpConstantFalse &&
        declaration->opcode() != SpvOpConstantNull) {
      continue;
    }

    auto transformation =
        TransformationToggleConstantNull(declaration->result_id());
    // Check the other conditions
    if (!transformation.IsApplicable(GetIRContext(),
                                     *GetTransformationContext())) {
      continue;
    }

    // Choose randomly whether to apply the transformation
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfTogglingConstantNull())) {
      continue;
    }

    ApplyTransformation(transformation);
  }
}
}  // namespace fuzz
}  // namespace spvtools
