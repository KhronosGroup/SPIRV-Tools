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

#include "source/fuzz/fuzzer_pass_replace_opselects_with_conditional_branches.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_replace_opselect_with_conditional_branch.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceOpSelectsWithConditionalBranches::
    FuzzerPassReplaceOpSelectsWithConditionalBranches(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceOpSelectsWithConditionalBranches::
    ~FuzzerPassReplaceOpSelectsWithConditionalBranches() = default;

void FuzzerPassReplaceOpSelectsWithConditionalBranches::Apply() {
  // Keep track of the instructions that we want to replace. We need to collect
  // them in a vector, since it's not safe to modify the module while iterating
  // over it.
  std::vector<uint32_t> replaceable_opselect_instruction_ids;

  // Loop over all the instructions in the module.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // We cannot split loop headers, so we don't need to consider instructions
      // in loop headers.
      if (block.IsLoopHeader()) {
        continue;
      }

      for (auto& instruction : block) {
        // We only care about OpSelect instructions.
        if (instruction.opcode() != SpvOpSelect) {
          continue;
        }

        // Randomly choose whether to consider this instruction for replacement.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()
                    ->GetChanceOfReplacingOpselectWithConditionalBranch())) {
          continue;
        }

        // If the instruction separates an OpSampledImage from its use, the
        // block cannot be split around it and the instruction cannot be
        // replaced.
        if (fuzzerutil::
                SplitBeforeInstructionSeparatesOpSampledImageDefinitionFromUse(
                    &block, &instruction)) {
          continue;
        }

        // We can apply the transformation to this instruction.
        replaceable_opselect_instruction_ids.push_back(instruction.result_id());
      }
    }
  }

  // Apply the transformations.
  for (uint32_t instruction_id : replaceable_opselect_instruction_ids) {
    ApplyTransformation(TransformationReplaceOpSelectWithConditionalBranch(
        instruction_id,
        {GetFuzzerContext()->GetFreshId(), GetFuzzerContext()->GetFreshId()}));
  }
}

}  // namespace fuzz
}  // namespace spvtools
