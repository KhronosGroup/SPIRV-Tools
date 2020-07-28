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

#include "source/fuzz/fuzzer_pass_propagate_instructions.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_propagate_instruction_up.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPropagateInstructions::FuzzerPassPropagateInstructions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassPropagateInstructions::~FuzzerPassPropagateInstructions() = default;

void FuzzerPassPropagateInstructions::Apply() {
  for (const auto& function : *GetIRContext()->module()) {
    for (const auto& block : function) {
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfPropagatingInstructions())) {
        continue;
      }

      if (TransformationPropagateInstructionUp::IsApplicableToTheBlock(
              GetIRContext(), block.id())) {
        std::unordered_map<uint32_t, uint32_t> fresh_ids;
        for (auto id : GetIRContext()->cfg()->preds(block.id())) {
          fresh_ids[id] = GetFuzzerContext()->GetFreshId();
        }

        ApplyTransformation(
            TransformationPropagateInstructionUp(block.id(), fresh_ids));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
