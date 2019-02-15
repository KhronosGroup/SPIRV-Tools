// Copyright (c) 2019 Google Inc.
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

#include "remove_selection_reduction_opportunity_finder.h"
#include "remove_selection_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

using namespace opt;

std::string RemoveSelectionReductionOpportunityFinder::GetName() const {
  return "RemoveSelectionReductionOpportunityFinder";
}

std::vector<std::unique_ptr<ReductionOpportunity>>
RemoveSelectionReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  // Find all opportunities for removing a selection construct by branching to
  // the LHS of the construct before finding corresponding opportunities for
  // branching to the RHS of the construct, to reduce the extent to which
  // reduction opportunities that can disable one another are adjacent.
  for (auto choose_lhs : {true, false}) {
    // Consider every function.
    for (auto& function : *context->module()) {
      // Consider every block in the function.
      for (auto& block : function) {
        // Check whether the block heads a selection construct.
        auto merge_inst = block.GetMergeInst();
        if (!merge_inst || merge_inst->opcode() != SpvOpSelectionMerge) {
          continue;
        }
        // Check that the construct is a conditional, not a switch.
        if (block.terminator()->opcode() != SpvOpBranchConditional) {
          continue;
        }
        // If the targets of the branch are the same, only add a reduction
        // opportunity to choose the LHS (as also adding an opportunity to
        // choose the RHS would be redundant).
        if (!choose_lhs && block.terminator()->GetSingleWordInOperand(1) ==
                               block.terminator()->GetSingleWordInOperand(2)) {
          continue;
        }
        // Add the opportunity to remove this selection construct, branching to
        // its former LHS or RHS.
        result.push_back(MakeUnique<RemoveSelectionReductionOpportunity>(
            &function, &block, choose_lhs));
      }
    }
  }
  return result;
}

}  // namespace reduce
}  // namespace spvtools
