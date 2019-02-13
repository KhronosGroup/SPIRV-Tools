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

#include "simplify_selection_reduction_opportunity_finder.h"
#include "simplify_selection_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

using namespace opt;

std::string SimplifySelectionReductionOpportunityFinder::GetName() const {
  return "SimplifySelectionReductionOpportunityFinder";
}

std::vector<std::unique_ptr<ReductionOpportunity>>
SimplifySelectionReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  // Find all opportunities for redirecting branches of a selection to the LHS
  // of the selection before corresponding opportunities for redirecting
  // branches to the RHS of the selection, because redirecting to the LHS
  // disables redirecting to the RHS, and efficiency of the reducer is improved
  // by avoiding contiguous opportunities from disabling one another.
  for (auto redirect_to_lhs : {true, false}) {
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
        // Check that the targets for the conditional are different.  If they
        // are not, the selection construct is already in simplified form.
        if (block.terminator()->GetSingleWordInOperand(1) ==
            block.terminator()->GetSingleWordInOperand(2)) {
          continue;
        }
        // Add the opportunity to simplify this construct so that both targets
        // become the same.
        result.push_back(MakeUnique<SimplifySelectionReductionOpportunity>(
            block.terminator(), redirect_to_lhs));
      }
    }
  }

  return result;
}

}  // namespace reduce
}  // namespace spvtools
