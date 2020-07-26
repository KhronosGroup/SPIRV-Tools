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

#include "source/fuzz/fuzzer_pass_replace_adds_subs_muls_with_carrying_extended.h"

#include "source/fuzz/transformation_replace_add_sub_mul_with_carrying_extended.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceAddsSubsMulsWithCarryingExtended::
    FuzzerPassReplaceAddsSubsMulsWithCarryingExtended(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceAddsSubsMulsWithCarryingExtended::
    ~FuzzerPassReplaceAddsSubsMulsWithCarryingExtended() = default;

void FuzzerPassReplaceAddsSubsMulsWithCarryingExtended::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& inst : block) {
        (void)inst;
        if (GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()
                    ->GetChanceOfReplacingAddSubMulWithCarryingExtended())) {
        }
      }
    }
  }
}
}  // namespace fuzz
}  // namespace spvtools
