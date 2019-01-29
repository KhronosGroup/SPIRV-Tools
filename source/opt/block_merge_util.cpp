// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "block_merge_util.h"

namespace spvtools {
namespace opt {
namespace blockmergeutil {

namespace {

// Returns true if |block| contains a merge instruction.
bool IsHeader(BasicBlock* block) { return block->GetMergeInst() != nullptr; }

// Returns true if |id| contains a merge instruction.
bool IsHeader(IRContext* context, uint32_t id) {
  return IsHeader(
      context->get_instr_block(context->get_def_use_mgr()->GetDef(id)));
}

// Returns true if |id| is the merge target of a merge instruction.
bool IsMerge(IRContext* context, uint32_t id) {
  return !context->get_def_use_mgr()->WhileEachUse(id, [](Instruction* user,
                                                          uint32_t index) {
    SpvOp op = user->opcode();
    if ((op == SpvOpLoopMerge || op == SpvOpSelectionMerge) && index == 0u) {
      return false;
    }
    return true;
  });
}

// Returns true if |block| is the merge target of a merge instruction.
bool IsMerge(IRContext* context, BasicBlock* block) {
  return IsMerge(context, block->id());
}

// Requires that any OpPhi in |successor| has |predecessor| as its sole parent.
// Replaces all uses of such OpPhis with the id associated with |predecessor|,
// and removes the OpPhi instructions.
void EliminateOpPhiInstructions(IRContext* context, BasicBlock* successor,
                                BasicBlock* predecessor) {
  successor->ForEachPhiInst([context, predecessor](Instruction* phi) {
    assert(2 == phi->NumInOperands() &&
           phi->GetSingleWordInOperand(1) == predecessor->id() &&
           "Blocks 'successor' and 'predecessor' are being merged, so "
           "'predecessor' must be the sole predecessor of 'successor', and "
           "thus any parent referenced in OpPhi must be 'predecessor'.");
    context->ReplaceAllUsesWith(phi->result_id(),
                                phi->GetSingleWordInOperand(0));
    context->KillInst(phi);
  });
}

}  // Anonymous namespace

bool CanMergeWithSuccessor(IRContext* context, BasicBlock* block) {
  // Find block with single successor which has no other predecessors.
  auto ii = block->end();
  --ii;
  Instruction* br = &*ii;
  if (br->opcode() != SpvOpBranch) {
    return false;
  }

  const uint32_t lab_id = br->GetSingleWordInOperand(0);
  if (context->cfg()->preds(lab_id).size() != 1) {
    return false;
  }

  bool pred_is_merge = IsMerge(context, block);
  bool succ_is_merge = IsMerge(context, lab_id);
  if (pred_is_merge && succ_is_merge) {
    // Cannot merge two merges together.
    return false;
  }

  Instruction* merge_inst = block->GetMergeInst();
  const bool pred_is_header = IsHeader(block);
  if (pred_is_header && lab_id != merge_inst->GetSingleWordInOperand(0u)) {
    bool succ_is_header = IsHeader(context, lab_id);
    if (pred_is_header && succ_is_header) {
      // Cannot merge two headers together when the successor is not the merge
      // block of the predecessor.
      return false;
    }

    // If this is a header block and the successor is not its merge, we must
    // be careful about which blocks we are willing to merge together.
    // OpLoopMerge must be followed by a conditional or unconditional branch.
    // The merge must be a loop merge because a selection merge cannot be
    // followed by an unconditional branch.
    BasicBlock* succ_block = context->get_instr_block(lab_id);
    SpvOp succ_term_op = succ_block->terminator()->opcode();
    assert(merge_inst->opcode() == SpvOpLoopMerge);
    if (succ_term_op != SpvOpBranch && succ_term_op != SpvOpBranchConditional) {
      return false;
    }
  }
  return true;
}

void MergeWithSuccessor(IRContext* context, Function* func,
                        Function::iterator bi) {
  assert(CanMergeWithSuccessor(context, &*bi) &&
         "Precondition failure for MergeWithSuccessor: it must be legal to "
         "merge the block and its successor.");

  auto ii = bi->end();
  --ii;
  Instruction* br = &*ii;
  const uint32_t lab_id = br->GetSingleWordInOperand(0);
  Instruction* merge_inst = bi->GetMergeInst();
  bool pred_is_header = IsHeader(&*bi);

  // Merge blocks.
  context->KillInst(br);
  auto sbi = bi;
  for (; sbi != func->end(); ++sbi)
    if (sbi->id() == lab_id) break;
  // If bi is sbi's only predecessor, it dominates sbi and thus
  // sbi must follow bi in func's ordering.
  assert(sbi != func->end());

  // Update the inst-to-block mapping for the instructions in sbi.
  for (auto& inst : *sbi) {
    context->set_instr_block(&inst, &*bi);
  }

  EliminateOpPhiInstructions(context, &*sbi, &*bi);

  // Now actually move the instructions.
  bi->AddInstructions(&*sbi);

  if (merge_inst) {
    if (pred_is_header && lab_id == merge_inst->GetSingleWordInOperand(0u)) {
      // Merging the header and merge blocks, so remove the structured control
      // flow declaration.
      context->KillInst(merge_inst);
    } else {
      // Move the merge instruction to just before the terminator.
      merge_inst->InsertBefore(bi->terminator());
    }
  }
  context->ReplaceAllUsesWith(lab_id, bi->id());
  context->KillInst(sbi->GetLabelInst());
  (void)sbi.Erase();
}

}  // namespace blockmergeutil
}  // namespace opt
}  // namespace spvtools
