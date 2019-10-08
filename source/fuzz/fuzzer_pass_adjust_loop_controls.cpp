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

#include "source/fuzz/fuzzer_pass_adjust_loop_controls.h"

#include "source/fuzz/transformation_set_loop_control.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAdjustLoopControls::FuzzerPassAdjustLoopControls(
        opt::IRContext* ir_context, FactManager* fact_manager,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
        : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations){};

FuzzerPassAdjustLoopControls::~FuzzerPassAdjustLoopControls() =
default;

void FuzzerPassAdjustLoopControls::Apply() {
  // Consider every merge instruction in the module (via looking through all
  // functions and blocks).
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      if (auto merge_inst = block.GetMergeInst()) {
        // Ignore the instruction if it is not a loop merge.
        if (merge_inst->opcode() != SpvOpLoopMerge) {
          continue;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAdjustingLoopControl())) {
          continue;
        }

        uint32_t existing_mask = merge_inst->GetSingleWordOperand(TransformationSetLoopControl::kLoopControlMaskInOperandIndex);
        std::vector<uint32_t> basic_masks = { SpvLoopControlMaskNone, SpvLoopControlUnrollMask, SpvLoopControlDontUnrollMask };
        uint32_t new_mask = basic_masks[GetFuzzerContext()->RandomIndex(basic_masks)];
        for (auto mask : {
                SpvLoopControlDependencyInfiniteMask,
                SpvLoopControlDependencyLengthMask,
                SpvLoopControlMinIterationsMask,
                SpvLoopControlMaxIterationsMask,
                SpvLoopControlIterationMultipleMask}) {
          if ((existing_mask & mask) && GetFuzzerContext()->ChooseEven()) {
            new_mask |= mask;
          }
        }

        uint32_t peel_count = 0;
        uint32_t partial_count = 0;

        if (!(new_mask & SpvLoopControlDontUnrollMask)) {
          if (TransformationSetLoopControl::PeelCountIsSupported(GetIRContext()) && GetFuzzerContext()->ChooseEven()) {
            new_mask |= SpvLoopControlPeelCountMask;
            peel_count = GetFuzzerContext()->GetRandomLoopControlPeelCount();
          }
          if (TransformationSetLoopControl::PartialCountIsSupported(GetIRContext()) &&
              GetFuzzerContext()->ChooseEven()) {
            new_mask |= SpvLoopControlPartialCountMask;
            partial_count = GetFuzzerContext()->GetRandomLoopControlPartialCount();
          }
        }

        // Apply the transformation and add it to the output transformation
        // sequence.
        TransformationSetLoopControl transformation(
                block.id(), new_mask, peel_count, partial_count);
        assert(transformation.IsApplicable(GetIRContext(), *GetFactManager()) &&
               "Transformation should be applicable by construction.");
        transformation.Apply(GetIRContext(), GetFactManager());
        *GetTransformations()->add_transformation() =
                transformation.ToMessage();
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
