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

#include "cut_loop_reduction_opportunity.h"
#include "source/opt/aggressive_dead_code_elim_pass.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace reduce {

namespace {
const uint32_t kMergeNodeIndex = 0;
const uint32_t kContinueNodeIndex = 1;
}  // namespace

bool CutLoopReductionOpportunity::PreconditionHolds() {
  return loop_construct_header_->GetLabel()
      ->context()
      ->GetDominatorAnalysis(enclosing_function_)
      ->IsReachable(loop_construct_header_->id());
}

void CutLoopReductionOpportunity::Apply() {
  auto loop_merge_inst = loop_construct_header_->GetLoopMergeInst();

  // Compute dominator analysis and CFG before we start to mess with edges in
  // the function.
  auto context = loop_merge_inst->context();
  auto dominator_analysis = context->GetDominatorAnalysis(enclosing_function_);
  auto cfg = context->cfg();

  // (1) Redirect edges that pointed to the loop's continue target.
  auto const loop_continue_target_id =
      loop_merge_inst->GetSingleWordOperand(kContinueNodeIndex);
  for (auto pred : cfg->preds(loop_continue_target_id)) {
    if (dominator_analysis->IsReachable(pred)) {
      ReplaceSelectionTargetWithClosestMerge(context, *cfg, *dominator_analysis,
                                             loop_continue_target_id, pred);
    }
  }

  // (2) Redirect edges that pointed to the loop's merge block.
  auto const loop_merge_block_id =
      loop_merge_inst->GetSingleWordOperand(kMergeNodeIndex);
  for (auto pred : cfg->preds(loop_merge_block_id)) {
    if (dominator_analysis->IsReachable(pred)) {
      ReplaceSelectionTargetWithClosestMerge(context, *cfg, *dominator_analysis,
                                             loop_merge_block_id, pred);
    }
  }

  // (3) Change the loop construct header to turn it into a selection
  ChangeLoopToSelection(context, *cfg);

  // (4) The continue target is now unreachable.  If it begins with phi nodes,
  // these must be removed.  We deal with this by changing them to Undefs.
  auto continue_block = cfg->block(loop_continue_target_id);
  for (auto& inst : *continue_block) {
    if (inst.opcode() != SpvOpPhi) {
      // Phi nodes must appear first in a block; break if we've seen the last
      // of them.
      break;
    }
    inst.SetOpcode(SpvOpUndef);
    inst.SetInOperands({});
  }

  // At this point we could consider removing the continue target, but currently
  // we do not.

  context->InvalidateAnalyses(IRContext::Analysis::kAnalysisCFG |
                              IRContext::Analysis::kAnalysisDominatorAnalysis);
}

void CutLoopReductionOpportunity::ReplaceSelectionTargetWithClosestMerge(
    IRContext* context, const CFG& cfg,
    const DominatorAnalysis& dominator_analysis,
    uint32_t original_target_block_id, uint32_t predecessor_block_id) {
  auto new_merge_target =
      FindClosestMerge(cfg, dominator_analysis, predecessor_block_id);
  assert(new_merge_target != predecessor_block_id);
  if (new_merge_target && new_merge_target != original_target_block_id) {
    RedirectEdge(predecessor_block_id, original_target_block_id,
                 new_merge_target, context, cfg);
  }
}

bool CutLoopReductionOpportunity::ContainedInStructuredControlFlowConstruct(
    uint32_t block_id, BasicBlock* selection_construct_header,
    const DominatorAnalysis& dominator_analysis) {
  assert(
      dominator_analysis.Dominates(selection_construct_header->id(), block_id));
  auto merge_inst = selection_construct_header->GetMergeInst();
  assert(merge_inst);
  auto merge_block_id = merge_inst->GetSingleWordOperand(kMergeNodeIndex);
  if (dominator_analysis.Dominates(merge_block_id, block_id)) {
    // The block is dominated by the construct's merge block.  Whether a loop or
    // selection, the block is not part of the construct.
    return false;
  }
  if (merge_inst->opcode() == SpvOpLoopMerge) {
    // In the case of a loop...
    if (dominator_analysis.Dominates(
            merge_inst->GetSingleWordOperand(kContinueNodeIndex), block_id)) {
      // The block is dominated by the loop's continue target, i.e. it is part
      // of the loop's continue construct and thus not part of the loop
      // construct.
      return false;
    }
  }

  // As a hack for now, due to some possible issues with how structured control
  // flow is defined, we also ask whether the merge comes later in the module
  // than the block.
  for (auto& bb : *enclosing_function_) {
    if (bb.id() == block_id) {
      return true;
    }
    if (bb.id() == merge_block_id) {
      // We saw the merge block first: this suggests that the block isn't part
      // of this selection construct.
      return false;
    }
  }
  return true;
}

uint32_t CutLoopReductionOpportunity::FindClosestMerge(
    const CFG& cfg, const DominatorAnalysis& dominator_analysis,
    uint32_t block_id) {
  assert(dominator_analysis.IsReachable(block_id));
  for (auto current_dominator = cfg.block(block_id);;
       current_dominator =
           dominator_analysis.ImmediateDominator(current_dominator)) {
    assert(current_dominator);
    if (current_dominator->GetMergeInst()) {
      if (ContainedInStructuredControlFlowConstruct(block_id, current_dominator,
                                                    dominator_analysis)) {
        return current_dominator->MergeBlockIdIfAny();
      }
    }
    if (current_dominator->id() == enclosing_function_->entry()->id()) {
      break;
    }
  }
  return 0;
}

void CutLoopReductionOpportunity::RedirectEdge(uint32_t source_id,
                                               uint32_t original_target_id,
                                               uint32_t new_target_id,
                                               IRContext* context,
                                               const CFG& cfg) {
  assert(source_id != original_target_id);
  assert(source_id != new_target_id);
  assert(original_target_id != new_target_id);

  // Redirect the edge; depends on what kind of branch instruction is involved.
  auto terminator = cfg.block(source_id)->terminator();
  if (terminator->opcode() == SpvOpBranch) {
    assert(terminator->GetSingleWordOperand(0) == original_target_id);
    terminator->SetOperand(0, {new_target_id});
  } else {
    assert(terminator->opcode() == SpvOpBranchConditional);
    assert(terminator->GetSingleWordOperand(1) == original_target_id ||
           terminator->GetSingleWordOperand(2) == original_target_id);
    for (auto index : {1u, 2u}) {
      if (terminator->GetSingleWordOperand(index) == original_target_id) {
        terminator->SetOperand(index, {new_target_id});
      }
    }
  }

  // The old and new targets may have phi nodes; these will need to respect the
  // change in edges.
  AdaptPhiNodesForRemovedEdge(source_id, cfg.block(original_target_id));
  AdaptPhiNodesForAddedEdge(source_id, cfg.block(new_target_id), context);
}

void CutLoopReductionOpportunity::ChangeLoopToSelection(IRContext* context,
                                                        const CFG& cfg) {
  auto loop_merge_inst = loop_construct_header_->GetLoopMergeInst();
  auto const loop_merge_block_id =
      loop_merge_inst->GetSingleWordOperand(kMergeNodeIndex);
  loop_merge_inst->SetOpcode(SpvOpSelectionMerge);
  loop_merge_inst->ReplaceOperands(
      {{loop_merge_inst->GetOperand(kMergeNodeIndex).type,
        {loop_merge_block_id}},
       {SPV_OPERAND_TYPE_SELECTION_CONTROL, {SpvSelectionControlMaskNone}}});
  auto terminator = loop_construct_header_->terminator();
  if (terminator->opcode() == SpvOpBranch) {
    analysis::Bool temp;
    const analysis::Bool* bool_type =
        context->get_type_mgr()->GetRegisteredType(&temp)->AsBool();
    auto const_mgr = context->get_constant_mgr();
    auto true_const = const_mgr->GetConstant(bool_type, {true});
    auto true_const_result_id =
        const_mgr->GetDefiningInstruction(true_const)->result_id();
    auto original_branch_id = terminator->GetSingleWordOperand(0);
    terminator->SetOpcode(SpvOpBranchConditional);
    terminator->ReplaceOperands({{SPV_OPERAND_TYPE_ID, {true_const_result_id}},
                                 {SPV_OPERAND_TYPE_ID, {original_branch_id}},
                                 {SPV_OPERAND_TYPE_ID, {loop_merge_block_id}}});
    if (original_branch_id != loop_merge_block_id) {
      // TODO: add a test for the case when they are equal.
      AdaptPhiNodesForAddedEdge(loop_construct_header_->id(),
                                cfg.block(loop_merge_block_id), context);
    }
  }
}

void CutLoopReductionOpportunity::AdaptPhiNodesForRemovedEdge(
    uint32_t from_id, BasicBlock* to_block) {
  for (auto& inst : *to_block) {
    if (inst.opcode() != SpvOpPhi) {
      break;
    }
    Instruction::OperandList new_in_operands;
    for (uint32_t index = 2; index < inst.NumOperands(); index += 2) {
      if (inst.GetOperand(index + 1).words[0] != from_id) {
        new_in_operands.push_back(inst.GetOperand(index));
        new_in_operands.push_back(inst.GetOperand(index + 1));
      }
    }
    inst.SetInOperands(std::move(new_in_operands));
  }
}

void CutLoopReductionOpportunity::AdaptPhiNodesForAddedEdge(
    uint32_t from_id, BasicBlock* to_block, IRContext* context) {
  for (auto& inst : *to_block) {
    if (inst.opcode() != SpvOpPhi) {
      break;
    }
    auto undef_id = FindOrCreateGlobalUndef(context, inst.type_id());
    inst.AddOperand(Operand(SPV_OPERAND_TYPE_ID, {undef_id}));
    inst.AddOperand(Operand(SPV_OPERAND_TYPE_ID, {from_id}));
  }
}

uint32_t CutLoopReductionOpportunity::FindOrCreateGlobalUndef(
    IRContext* context, uint32_t type_id) {
  for (auto& inst : context->module()->types_values()) {
    if (inst.opcode() != SpvOpUndef) {
      continue;
    }
    if (inst.type_id() == type_id) {
      return inst.result_id();
    }
  }
  // TODO: this is adapted from MemPass::Type2Undef.  In due course it would
  // be good to factor out this duplication.
  const uint32_t undef_id = context->TakeNextId();
  std::unique_ptr<Instruction> undef_inst(
      new Instruction(context, SpvOpUndef, type_id, undef_id, {}));
  assert(undef_id == undef_inst->result_id());
  context->module()->AddGlobalValue(std::move(undef_inst));
  return undef_id;
}

}  // namespace reduce
}  // namespace spvtools
