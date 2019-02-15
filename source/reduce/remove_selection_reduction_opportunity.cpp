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
  auto merge_instruction = block_->GetMergeInst();
  auto merge_block_id = merge_instruction->GetSingleWordInOperand(0);
  auto terminator = block_->terminator();

  terminator->SetOpcode(SpvOpBranch);
  // Pick one of the former conditional branch operands depending on whether we
  // are choosing the LHS or RHS.
  terminator->SetInOperands({choose_lhs_ ? terminator->GetInOperand(1)
                                         : terminator->GetInOperand(2)});

  // Go through the function looking for breaks out of the selection, i.e.
  // conditional branches that can jump to the selection's merge block.  Rewrite
  // these as unconditional branches.
  for (auto& block : *function_) {
    for (auto& inst : block) {
      if (inst.opcode() != SpvOpBranchConditional) {
        // This is not a conditional branch; move on.
        continue;
      }
      const auto true_target = inst.GetSingleWordInOperand(1);
      const auto false_target = inst.GetSingleWordInOperand(2);
      if (true_target != merge_block_id && false_target != merge_block_id) {
        // This conditional branch does not target the merge block for the
        // selection; move on.
        continue;
      }
      // Change the conditional branch to an unconditional branch; if one of the
      // targets of the branch is not the merge block of the selection then
      // favour that target.
      inst.SetOpcode(SpvOpBranch);
      uint32_t new_target =
          true_target == merge_block_id ? false_target : true_target;
      inst.SetInOperands({{SPV_OPERAND_TYPE_ID, {new_target}}});
    }
  }

  // Remove the merge instruction.
  merge_instruction->context()->KillInst(merge_instruction);
}

}  // namespace reduce
}  // namespace spvtools
