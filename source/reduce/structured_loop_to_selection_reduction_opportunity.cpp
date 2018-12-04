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

#include "structured_loop_to_selection_reduction_opportunity.h"
#include "source/opt/aggressive_dead_code_elim_pass.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace reduce {

namespace {
const uint32_t kMergeNodeIndex = 0;
const uint32_t kContinueNodeIndex = 1;
}  // namespace

bool StructuredLoopToSelectionReductionOpportunity::PreconditionHolds() {
  // Is the loop header reachable?
  return loop_construct_header_->GetLabel()
      ->context()
      ->GetDominatorAnalysis(enclosing_function_)
      ->IsReachable(loop_construct_header_);
}

void StructuredLoopToSelectionReductionOpportunity::Apply() {
  auto loop_merge_inst = loop_construct_header_->GetLoopMergeInst();

  // Compute dominator analysis and CFG before we start to mess with edges in
  // the function.
  auto context = loop_merge_inst->context();
  auto dominator_analysis = context->GetDominatorAnalysis(enclosing_function_);
  auto cfg = context->cfg();

  // (1) Redirect edges that point to the loop's continue target to their
  // closest merge block.
  RedirectToClosestMergeBlock(
      loop_merge_inst->GetSingleWordOperand(kContinueNodeIndex), context,
      *dominator_analysis, *cfg);

  // (2) Redirect edges that point to the loop's merge block to their closest
  // merge block (which might be that of an enclosing selection, for instance).
  RedirectToClosestMergeBlock(
      loop_merge_inst->GetSingleWordOperand(kMergeNodeIndex), context,
      *dominator_analysis, *cfg);

  // (3) Turn the loop construct header into a selection.
  ChangeLoopToSelection(context, *cfg);

  // We have made control flow changes, so invalidate the analyses that were
  // calculated.
  context->InvalidateAnalyses(IRContext::Analysis::kAnalysisCFG |
                              IRContext::Analysis::kAnalysisDominatorAnalysis);

  // (4) By changing CFG edges we may have created scenarios where ids are used
  // without being dominated.  Address this.
  dominator_analysis = context->GetDominatorAnalysis(enclosing_function_);
  auto def_use_mgr = context->get_def_use_mgr();
  FixNonDominatedIdUses(*dominator_analysis, *def_use_mgr);

  // Invalidate the analyses we just used to reflect changes.
  context->InvalidateAnalyses(IRContext::Analysis::kAnalysisDefUse |
                              IRContext::Analysis::kAnalysisDominatorAnalysis);
}

void StructuredLoopToSelectionReductionOpportunity::RedirectToClosestMergeBlock(
    uint32_t original_target_id, IRContext* context,
    const DominatorAnalysis& dominator_analysis, const CFG& cfg) {
  // Consider every predecessor of the node with respect to which edges should
  // be redirected.
  for (auto pred : cfg.preds(original_target_id)) {
    if (!dominator_analysis.IsReachable(pred)) {
      // We do not care about unreachable predecessors (and dominance
      // information, and thus the notion of structured control flow, makes
      // little sense for unreachable blocks).
      continue;
    }
    // Find the merge block of the structured control construct that most
    // tightly encloses the predecessor.
    auto new_merge_target = FindClosestMerge(cfg, dominator_analysis, pred);
    assert(new_merge_target != pred);

    if (!new_merge_target) {
      // If the loop being transformed is outermost, and the predecessor is
      // part of that loop's continue construct, there will be no such
      // enclosing control construct.  In this case, the continue construct
      // will become unreachable anyway, so it is fine not to redirect the
      // edge.
      continue;
    }

    if (new_merge_target != original_target_id) {
      // Redirect the edge if it doesn't already point to the desired block.
      RedirectEdge(pred, original_target_id, new_merge_target, context, cfg);
    }
  }
}

uint32_t StructuredLoopToSelectionReductionOpportunity::FindClosestMerge(
    const CFG& cfg, const DominatorAnalysis& dominator_analysis,
    uint32_t block_id) {
  // We want to find the merge block associated with the structured control flow
  // construct that most tightly contains block_id.
  //
  // For example if our SPIR-V had come from some structured code like:
  //
  // while (...) {
  //   if (...) {
  //     if (...) {
  //       straight-line code including block_id
  //     }
  //   }
  // }
  //
  // then we are looking to find the merge block for the inner-most "if".

  assert(dominator_analysis.IsReachable(block_id));

  // Starting from the given block, walk the dominator tree backwards until
  // we find the structured-control-construct-with-merge this block most tightly
  // belongs to, or ascertain that it belongs to no such construct.
  for (auto current_dominator = cfg.block(block_id); /* return is guaranteed */;
       current_dominator =
           dominator_analysis.ImmediateDominator(current_dominator)) {
    assert(current_dominator);
    if (current_dominator->GetMergeInst()) {
      // The dominator has a merge instruction, so it heads a structured control
      // flow construct.  Is block_id part of it?
      if (ContainedInStructuredControlFlowConstruct(block_id, current_dominator,
                                                    dominator_analysis)) {
        // Yes.  Return the merge block for the construct.
        return current_dominator->MergeBlockIdIfAny();
      }
    }
    if (current_dominator->id() == enclosing_function_->entry()->id()) {
      // We have walked to the top of the dominator tree and not found any
      // structured-control-flow-construct-with-merge to which this block
      // belongs.
      return 0;
    }
  }
}

bool StructuredLoopToSelectionReductionOpportunity::
    ContainedInStructuredControlFlowConstruct(
        uint32_t block_id, BasicBlock* selection_construct_header,
        const DominatorAnalysis& dominator_analysis) {
  // SPIR-V spec says that to be part of a loop or selection, a block must at
  // a minimum be dominated by the header of the construct.  Check that we
  // only get here when that's the case.

  assert(
      dominator_analysis.Dominates(selection_construct_header->id(), block_id));
  auto merge_inst = selection_construct_header->GetMergeInst();
  assert(merge_inst);

  // Next, the spec says that to be part of the construct the block cannot be
  // dominated by the merge block of the construct.
  auto merge_block_id = merge_inst->GetSingleWordOperand(kMergeNodeIndex);
  if (dominator_analysis.Dominates(merge_block_id, block_id)) {
    // The block is dominated by the construct's merge block.  Whether a loop or
    // selection, the block is not part of the construct.
    return false;
  }

  if (merge_inst->opcode() == SpvOpLoopMerge) {
    // In the case of a loop we have the further requirement that, to be in
    // the construct, the block must not be dominated by the loop's continue
    // target.
    if (dominator_analysis.Dominates(
            merge_inst->GetSingleWordOperand(kContinueNodeIndex), block_id)) {
      return false;
    }
  }

  // The SPIR-V spec's definition of structured control flow is unnatural for
  // various scenarios where loop breaks and continues are nested inside
  // conditionals.  To account for this in the short term, we impose the
  // additional restriction that the block must appear, in program order,
  // before the construct's merge block in order to be part of the construct.
  //
  // TODO: This is a workaround that should be replaced in due course with a
  // more precise check if the rules for structured control flow are refined.
  for (auto& bb : *enclosing_function_) {
    if (bb.id() == block_id) {
      // We find block_id before the merge block, so we decide that block_id
      // is part of the structured control flow construct.
      return true;
    }
    if (bb.id() == merge_block_id) {
      // We find the merge block before block_id.  This suggests that block_id
      // should not really be part of the construct.
      return false;
    }
  }
  return true;
}

void StructuredLoopToSelectionReductionOpportunity::RedirectEdge(
    uint32_t source_id, uint32_t original_target_id, uint32_t new_target_id,
    IRContext* context, const CFG& cfg) {
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

void StructuredLoopToSelectionReductionOpportunity::AdaptPhiNodesForRemovedEdge(
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

void StructuredLoopToSelectionReductionOpportunity::AdaptPhiNodesForAddedEdge(
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

uint32_t StructuredLoopToSelectionReductionOpportunity::FindOrCreateGlobalUndef(
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

uint32_t
StructuredLoopToSelectionReductionOpportunity::FindOrCreateGlobalVariable(
    IRContext* context, uint32_t base_type_id) {
  for (auto& inst : context->module()->types_values()) {
    if (inst.opcode() != SpvOpVariable) {
      continue;
    }
    if (context->get_type_mgr()->GetId(context->get_type_mgr()
                                           ->GetType(inst.type_id())
                                           ->AsPointer()
                                           ->pointee_type()) == base_type_id) {
      return inst.result_id();
    }
  }
  auto pointer_type_id = context->get_type_mgr()->FindPointerToType(
      base_type_id, SpvStorageClassPrivate);

  // TODO: refactor to avoid duplication with FindOrCreateGlobalUndef.
  const uint32_t var_id = context->TakeNextId();
  std::unique_ptr<Instruction> undef_inst(new Instruction(
      context, SpvOpVariable, pointer_type_id, var_id,
      {{SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassPrivate}}}));
  assert(var_id == undef_inst->result_id());
  context->module()->AddGlobalValue(std::move(undef_inst));
  return var_id;
}

void StructuredLoopToSelectionReductionOpportunity::ChangeLoopToSelection(
    IRContext* context, const CFG& cfg) {
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
      // TODO: add a test for the case where they are equal.
      AdaptPhiNodesForAddedEdge(loop_construct_header_->id(),
                                cfg.block(loop_merge_block_id), context);
    }
  }
}

void StructuredLoopToSelectionReductionOpportunity::FixNonDominatedIdUses(
    const DominatorAnalysis& dominator_analysis,
    const spvtools::opt::analysis::DefUseManager& def_use_mgr) {
  for (auto& block : *enclosing_function_) {
    for (auto& def : block) {
      if (def.opcode() == SpvOpVariable) {
        continue;
      }
      def_use_mgr.ForEachUse(&def, [this, &def, &dominator_analysis](
                                       Instruction* use, uint32_t index) {
        if (use->opcode() == SpvOpPhi) {
          // TODO: Consider carefully what to do.  We should probably check
          // dominance on the incoming edge.
        } else if (!dominator_analysis.Dominates(&def, use)) {
          if (def.opcode() == SpvOpAccessChain) {
            auto base_type = def.context()->get_type_mgr()->GetId(
                def.context()
                    ->get_type_mgr()
                    ->GetType(def.type_id())
                    ->AsPointer()
                    ->pointee_type());
            use->SetOperand(
                index, {FindOrCreateGlobalVariable(def.context(), base_type)});
          } else {
            use->SetOperand(
                index, {FindOrCreateGlobalUndef(def.context(), def.type_id())});
          }
        }
      });
    }
  }
}

}  // namespace reduce
}  // namespace spvtools
