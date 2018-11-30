// Copyright (c) 2018 Google LLC
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

#include "structured_loop_to_selection_reduction_pass.h"
#include "structured_loop_to_selection_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

using namespace opt;

namespace {
const uint32_t kMergeNodeIndex = 0;
}

std::vector<std::unique_ptr<ReductionOpportunity>>
StructuredLoopToSelectionReductionPass::GetAvailableOpportunities(
    opt::IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  // Consider each loop construct header in the module.
  for (auto& function : *context->module()) {
    for (auto block_iterator = function.begin();
         block_iterator != function.end(); ++block_iterator) {
      auto loop_merge_inst = block_iterator->GetLoopMergeInst();
      if (!loop_merge_inst) {
        // This is not a loop construct header.
        continue;
      }
      // Check whether the loop construct header dominates its merge block.
      // If not, the merge block must be unreachable in the control flow graph
      // so we cautiously do not consider applying a transformation.
      auto merge_block_id =
          loop_merge_inst->GetSingleWordInOperand(kMergeNodeIndex);
      if (!context->GetDominatorAnalysis(&function)->Dominates(
              block_iterator->id(), merge_block_id)) {
        continue;
      }
      if (!context->GetPostDominatorAnalysis(&function)->Dominates(
              merge_block_id, block_iterator->id())) {
        continue;
      }

      result.push_back(
          MakeUnique<StructuredLoopToSelectionReductionOpportunity>(
              block_iterator, &function));
    }
  }
  return result;
}

std::string StructuredLoopToSelectionReductionPass::GetName() const {
  return "StructuredLoopToSelectionReductionPass";
}

}  // namespace reduce
}  // namespace spvtools
