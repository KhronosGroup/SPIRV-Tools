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

#include "source/fuzz/fuzzer_replace_copyobject_with_store_load.h"

#include "source/fuzz/transformation_replace_copyobject_with_store_load.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceCopyObjectWithStoreLoad::
    FuzzerPassReplaceCopyObjectWithStoreLoad(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceCopyObjectWithStoreLoad::
    ~FuzzerPassReplaceCopyObjectWithStoreLoad() = default;

// TODO: Check/Write
void FuzzerPassReplaceCopyObjectWithStoreLoad::Apply() {
  // Consider every instruction in every block in every function.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& inst : block) {
        if (GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingRelaxedDecoration())) {
          TransformationAddRelaxedDecoration transformation(inst.result_id());
          // Restrict attention to applicable instructions.
          if (transformation.IsApplicable(GetIRContext(),
                                          *GetTransformationContext())) {
            transformation.Apply(GetIRContext(), GetTransformationContext());
            *GetTransformations()->add_transformation() =
                transformation.ToMessage();
          }
        }
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
