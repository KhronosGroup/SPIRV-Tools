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

#include "source/fuzz/fuzzer_pass_add_no_contraction_decorations.h"

#include "source/fuzz/transformation_add_no_contraction_decoration.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddNoContractionDecorations::FuzzerPassAddNoContractionDecorations(
        opt::IRContext* ir_context, FactManager* fact_manager,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
        : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations){};

FuzzerPassAddNoContractionDecorations::~FuzzerPassAddNoContractionDecorations() = default;

void FuzzerPassAddNoContractionDecorations::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& inst : block) {
        if (TransformationAddNoContractionDecoration::IsArithmetic(inst.opcode())) {
          if (GetFuzzerContext()->ChoosePercentage(GetFuzzerContext()->GetChanceOfAddingNoContractionDecoration())) {
            TransformationAddNoContractionDecoration transformation(inst.result_id());
            assert(transformation.IsApplicable(GetIRContext(), *GetFactManager()) && "Transformation should be applicable by construction.");
            transformation.Apply(GetIRContext(), GetFactManager());
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
