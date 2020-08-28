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

#include "source/fuzz/fuzzer_pass_flatten_conditional_branches.h"

#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_flatten_conditional_branch.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass that randomly selects conditional branches to flatten and
// flattens them, if possible.
FuzzerPassFlattenConditionalBranches::FuzzerPassFlattenConditionalBranches(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassFlattenConditionalBranches::~FuzzerPassFlattenConditionalBranches() =
    default;

void FuzzerPassFlattenConditionalBranches::Apply() {
  // Get all the selection headers that we want to flatten. We need to collect
  // all of them first, because, since we are changing the structure of the
  // module, it's not safe to modify them while iterating.
  std::vector<opt::BasicBlock*> selection_headers;
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // Randomly decide whether to consider this block.
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfFlatteningConditionalBranch())) {
        continue;
      }

      // Only consider this block if it is the header of a conditional.
      if (block.GetMergeInst() &&
          block.GetMergeInst()->opcode() == SpvOpSelectionMerge &&
          block.terminator()->opcode() == SpvOpBranchConditional) {
        selection_headers.emplace_back(&block);
      }
    }
  }

  // Apply the transformation to the headers which can be flattened.
  for (auto header : selection_headers) {
    // Make a set to keep track of the instructions that need fresh ids.
    std::set<opt::Instruction*> instructions_that_need_ids;

    // Do not consider this header if the conditional cannot be flattened.
    if (!TransformationFlattenConditionalBranch::ConditionalCanBeFlattened(
            GetIRContext(), header, &instructions_that_need_ids)) {
      continue;
    }

    // Some instructions will require to be enclosed inside conditionals because
    // they have side effects (for example, loads and stores). Some of this have
    // no result id, so we require instruction descriptors to identify them.
    // Each of them is associated with an IdsForEnclosingInst struct, containing
    // all the necessary fresh ids for it.
    std::vector<
        std::pair<protobufs::InstructionDescriptor, IdsForEnclosingInst>>
        instructions_to_fresh_ids;

    for (auto instruction : instructions_that_need_ids) {
      IdsForEnclosingInst info = {GetFuzzerContext()->GetFreshId(),
                                  GetFuzzerContext()->GetFreshId(), 0, 0, 0};

      if (TransformationFlattenConditionalBranch::InstructionNeedsPlaceholder(
              GetIRContext(), *instruction)) {
        info.actual_result_id = GetFuzzerContext()->GetFreshId();
        info.alternative_block_id = GetFuzzerContext()->GetFreshId();
        info.placeholder_result_id = GetFuzzerContext()->GetFreshId();
      }

      instructions_to_fresh_ids.push_back(
          {MakeInstructionDescriptor(GetIRContext(), instruction), info});
    }

    // Apply the transformation.
    ApplyTransformation(TransformationFlattenConditionalBranch(
        header->id(), std::move(instructions_to_fresh_ids)));
  }
}
}  // namespace fuzz
}  // namespace spvtools
