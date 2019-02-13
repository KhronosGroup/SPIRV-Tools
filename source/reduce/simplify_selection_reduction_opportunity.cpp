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

#include "source/opt/instruction.h"

#include "simplify_selection_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

bool SimplifySelectionReductionOpportunity::PreconditionHolds() {
  // The reduction opportunity is enabled if the two targets of the branch
  // instruction are different.  If another reduction opportunity redirects
  // them, this may become false.
  return branch_instruction_->GetSingleWordInOperand(1) !=
         branch_instruction_->GetSingleWordInOperand(2);
}

void SimplifySelectionReductionOpportunity::Apply() {
  // If we are redirecting both branches to the LHS branch then we want to copy
  // operand 1 into operand 2; otherwise the opposite.
  uint32_t operand_to_modify = redirect_to_lhs_ ? 2 : 1;
  uint32_t operand_to_copy = redirect_to_lhs_ ? 1 : 2;

  // Do the branch redirection.
  branch_instruction_->SetInOperand(
      operand_to_modify,
      {branch_instruction_->GetSingleWordInOperand(operand_to_copy)});
}

}  // namespace reduce
}  // namespace spvtools
