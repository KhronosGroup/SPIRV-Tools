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

#include "source/opt/basic_block.h"
#include "source/opt/ir_context.h"

#include "remove_selection_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

bool RemoveSelectionReductionOpportunity::PreconditionHolds() {
  // It is possible that another opportunity may have changed the block so that
  // it no longer heads a selection construct.
  return block_->GetMergeInst() &&
         block_->GetMergeInst()->opcode() == SpvOpSelectionMerge &&
         block_->terminator()->opcode() == SpvOpBranchConditional;
}

void RemoveSelectionReductionOpportunity::Apply() {
  block_->GetMergeInst()->context()->KillInst(block_->GetMergeInst());
  auto terminator = block_->terminator();
  terminator->SetOpcode(SpvOpBranch);
  // Pick one of the former conditional branch operands depending on whether we
  // are choosing the LHS or RHS.
  terminator->SetInOperands({choose_lhs_ ? terminator->GetInOperand(1)
                                         : terminator->GetInOperand(2)});
}

}  // namespace reduce
}  // namespace spvtools
