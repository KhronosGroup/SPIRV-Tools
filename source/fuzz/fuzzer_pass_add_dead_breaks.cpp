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

#include "source/fuzz/fuzzer_pass_add_dead_breaks.h"
#include "source/fuzz/transformation_add_dead_break.h"

namespace spvtools {
namespace fuzz {

using opt::IRContext;

void FuzzerPassAddDeadBreaks::Apply(
    IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations) {
  // We first collect up lots of possibly-applicable transformations.
  std::vector<protobufs::TransformationAddDeadBreak> candidate_transformations;
  // We consider each function separately.
  for (auto& function : *ir_context->module()) {
    // For a given function, we find all the merge blocks in that function.
    std::vector<uint32_t> merge_block_ids;
    for (auto& block : function) {
      auto maybe_merge_id = block.MergeBlockIdIfAny();
      if (maybe_merge_id) {
        merge_block_ids.push_back(maybe_merge_id);
      }
    }
    // We rather aggressively consider the possibility of adding a break from
    // every block in the function to every merge block.  Many of these will be
    // inapplicable as they would be illegal.  That's OK - we discard the ones
    // that turn out to be no good.
    for (auto& block : function) {
      for (auto merge_block_id : merge_block_ids) {
        // TODO: right now we completely ignore OpPhi instructions at merge
        // blocks.  This will lead to interesting opportunities being missed.
        std::vector<uint32_t> phi_ids;
        auto candidate_transformation =
            transformation::MakeTransformationAddDeadBreak(
                block.id(), merge_block_id,
                fuzzer_context->GetRandomGenerator()->RandomBool(),
                std::move(phi_ids));
        if (transformation::IsApplicable(candidate_transformation, ir_context,
                                         *fact_manager)) {
          // Only consider a transformation as a candidate if it is applicable.
          candidate_transformations.push_back(
              std::move(candidate_transformation));
        }
      }
    }
  }

  // Go through the candidate transformations that were accumulated,
  // probabilistically deciding whether to consider each one further and
  // applying the still-applicable ones that are considered further.
  while (!candidate_transformations.empty()) {
    auto index = fuzzer_context->GetRandomGenerator()->RandomUint32(
        (uint32_t)candidate_transformations.size());
    auto message = std::move(candidate_transformations[index]);
    candidate_transformations.erase(candidate_transformations.begin() + index);
    if (fuzzer_context->GetRandomGenerator()->RandomPercentage() >
        fuzzer_context->GetChanceOfAddingDeadBreak()) {
      continue;
    }
    if (transformation::IsApplicable(message, ir_context, *fact_manager)) {
      *transformations->add_transformation()->mutable_add_dead_break() =
          message;
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
