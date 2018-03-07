// Copyright (c) 2018 Google LLC.
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

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ir_builder.h"
#include "ir_context.h"
#include "loop_descriptor.h"
#include "loop_peeling.h"
#include "loop_utils.h"

namespace spvtools {
namespace opt {

void LoopPeeling::DuplicateLoop() {
  ir::CFG& cfg = *context_->cfg();

  assert(CanPeelLoop() && "Cannot peel loop!");

  LoopUtils::LoopCloningResult clone_results;

  std::vector<ir::BasicBlock*> ordered_loop_blocks;
  ir::BasicBlock* pre_header = loop_->GetOrCreatePreHeaderBlock();

  loop_->ComputeLoopStructuredOrder(&ordered_loop_blocks);

  new_loop_ = loop_utils_.CloneLoop(&clone_results, ordered_loop_blocks);

  // Add the basic block to the function.
  ir::Function::iterator it =
      loop_utils_.GetFunction()->FindBlock(pre_header->id());
  assert(it != loop_utils_.GetFunction()->end());
  loop_utils_.GetFunction()->AddBasicBlocks(
      clone_results.cloned_bb_.begin(), clone_results.cloned_bb_.end(), ++it);
  // Make the |loop_|'s preheader the |new_loop| one.
  ir::BasicBlock* clonedHeader = new_loop_->GetHeaderBlock();
  pre_header->ForEachSuccessorLabel(
      [clonedHeader](uint32_t* succ) { *succ = clonedHeader->id(); });
  // Update cfg.
  cfg.RemoveEdge(pre_header->id(), loop_->GetHeaderBlock()->id());
  new_loop_->SetPreHeaderBlock(pre_header);
  loop_->SetPreHeaderBlock(nullptr);

  // When cloning the loop, we didn't cloned the merge block, so currently
  // |new_loop| shares the same block as |loop_|.
  // We mutate all branches form |new_loop| block to |loop_|'s merge into a
  // branch to |loop_|'s header (so header will also be the merge of
  // |new_loop|).
  uint32_t after_loop_pred = 0;
  for (uint32_t pred_id : cfg.preds(loop_->GetMergeBlock()->id())) {
    if (loop_->IsInsideLoop(pred_id)) continue;
    ir::BasicBlock* bb = cfg.block(pred_id);
    assert(after_loop_pred == 0 && "Predecessor already registered");
    after_loop_pred = bb->id();
    bb->ForEachSuccessorLabel([this](uint32_t* succ) {
      if (*succ == loop_->GetMergeBlock()->id())
        *succ = loop_->GetHeaderBlock()->id();
    });
  }

  // Update cfg.
  cfg.RemoveNonExistingEdges(loop_->GetMergeBlock()->id());
  cfg.AddEdge(after_loop_pred, loop_->GetHeaderBlock()->id());

  // Set the merge block of the new loop as the old header block.
  new_loop_->SetMergeBlock(loop_->GetHeaderBlock());

  // Force the creation of a new preheader and patch the phi of the header.
  loop_->GetHeaderBlock()->ForEachPhiInst(
      [after_loop_pred, this](ir::Instruction* phi) {
        for (uint32_t i = 1; i < phi->NumInOperands(); i += 2) {
          if (!loop_->IsInsideLoop(phi->GetSingleWordInOperand(i))) {
            phi->SetInOperand(i, {after_loop_pred});
            return;
          }
        }
      });

  ConnectIterators(clone_results);
}

void LoopPeeling::InsertIterator(ir::Instruction* factor) {
  analysis::Type* factor_type =
      context_->get_type_mgr()->GetType(factor->type_id());
  assert(factor_type->kind() == analysis::Type::kInteger);
  analysis::Integer* int_type = factor_type->AsInteger();
  assert(int_type->width() == 32);

  InstructionBuilder builder(context_,
                             &*GetBeforeLoop()->GetLatchBlock()->tail(),
                             ir::IRContext::kAnalysisDefUse |
                                 ir::IRContext::kAnalysisInstrToBlockMapping);
  // Create the increment.
  // Note that "factor->result_id()" is wrong, the proper id should the phi
  // value but we don't have it yet. The operand will be set latter, leave
  // "factor->result_id()" so that the id is a valid and so avoid any assert
  // that's could be added.
  ir::Instruction* iv_inc = builder.AddIAdd(
      factor->type_id(), factor->result_id(),
      builder.Add32BitConstantInteger<uint32_t>(1, int_type->IsSigned())
          ->result_id());

  builder.SetInsertPoint(&*GetBeforeLoop()->GetHeaderBlock()->begin());

  extra_induction_variable_ = builder.AddPhi(
      factor->type_id(),
      {builder.Add32BitConstantInteger<uint32_t>(0, int_type->IsSigned())
           ->result_id(),
       GetBeforeLoop()->GetPreHeaderBlock()->id(), iv_inc->result_id(),
       GetBeforeLoop()->GetLatchBlock()->id()});
  // Connect everything.
  iv_inc->SetInOperand(0, {extra_induction_variable_->result_id()});

  // If do-while form, use the incremented value.
  if (do_while_form_) {
    extra_induction_variable_ = iv_inc;
  }
}

void LoopPeeling::ConnectIterators(
    const LoopUtils::LoopCloningResult& clone_results) {
  ir::BasicBlock* header = loop_->GetHeaderBlock();
  header->ForEachPhiInst([&clone_results, this](ir::Instruction* phi) {
    for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
      uint32_t pred_id = phi->GetSingleWordInOperand(i + 1);
      if (!loop_->IsInsideLoop(pred_id)) {
        phi->SetInOperand(i,
                          {clone_results.value_map_.at(
                              exit_value_.at(phi->result_id())->result_id())});
      }
    }
  });
}

void LoopPeeling::GetIteratorUpdateOperations(
    const ir::Loop* loop, ir::Instruction* iterator,
    std::unordered_set<ir::Instruction*>* operations) {
  opt::analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();
  operations->insert(iterator);
  iterator->ForEachInId([def_use_mgr, loop, operations, this](uint32_t* id) {
    ir::Instruction* insn = def_use_mgr->GetDef(*id);
    if (insn->opcode() == SpvOpLabel) {
      return;
    }
    if (operations->count(insn)) {
      return;
    }
    if (!loop->IsInsideLoop(insn)) {
      return;
    }
    GetIteratorUpdateOperations(loop, insn, operations);
  });
}

void LoopPeeling::GetIteratingExitValue() {
  ir::CFG& cfg = *context_->cfg();

  loop_->GetHeaderBlock()->ForEachPhiInst([this](ir::Instruction* phi) {
    exit_value_[phi->result_id()] = nullptr;
  });

  if (!loop_->GetMergeBlock()) {
    return;
  }
  if (cfg.preds(loop_->GetMergeBlock()->id()).size() != 1) {
    return;
  }
  opt::analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

  uint32_t condition_block_id = cfg.preds(loop_->GetMergeBlock()->id())[0];

  auto& header_pred = cfg.preds(loop_->GetHeaderBlock()->id());
  do_while_form_ = std::find(header_pred.begin(), header_pred.end(),
                             condition_block_id) != header_pred.end();
  if (do_while_form_) {
    loop_->GetHeaderBlock()->ForEachPhiInst(
        [condition_block_id, def_use_mgr, this](ir::Instruction* phi) {
          std::unordered_set<ir::Instruction*> operations;

          for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
            if (condition_block_id == phi->GetSingleWordInOperand(i + 1)) {
              exit_value_[phi->result_id()] =
                  def_use_mgr->GetDef(phi->GetSingleWordInOperand(i));
            }
          }
        });
  } else {
    DominatorTree* dom_tree =
        &context_->GetDominatorAnalysis(loop_utils_.GetFunction(), cfg)
             ->GetDomTree();
    ir::BasicBlock* condition_block = cfg.block(condition_block_id);

    loop_->GetHeaderBlock()->ForEachPhiInst(
        [dom_tree, condition_block, this](ir::Instruction* phi) {
          std::unordered_set<ir::Instruction*> operations;

          // Not the back-edge value, check if the phi instruction is the only
          // possible candidate.
          GetIteratorUpdateOperations(loop_, phi, &operations);

          for (ir::Instruction* insn : operations) {
            if (insn == phi) {
              continue;
            }
            if (dom_tree->Dominates(context_->get_instr_block(insn),
                                    condition_block)) {
              return;
            }
          }
          exit_value_[phi->result_id()] = phi;
        });
  }
}

void LoopPeeling::FixExitCondition(
    const std::function<uint32_t(ir::BasicBlock*)>& condition_builder) {
  ir::CFG& cfg = *context_->cfg();

  uint32_t condition_block_id = 0;
  for (uint32_t id : cfg.preds(GetAfterLoop()->GetHeaderBlock()->id())) {
    if (!GetAfterLoop()->IsInsideLoop(id)) {
      condition_block_id = id;
    }
  }
  assert(condition_block_id != 0 && "2nd loop in improperly connected");

  ir::BasicBlock* condition_block = cfg.block(condition_block_id);
  assert(condition_block->terminator()->opcode() == SpvOpBranchConditional);
  InstructionBuilder builder(context_, &*condition_block->tail(),
                             ir::IRContext::kAnalysisDefUse |
                                 ir::IRContext::kAnalysisInstrToBlockMapping);

  condition_block->terminator()->SetInOperand(
      0, {condition_builder(condition_block)});

  uint32_t to_continue_block =
      condition_block->terminator()->GetSingleWordInOperand(
          condition_block->terminator()->GetSingleWordInOperand(1) ==
                  GetAfterLoop()->GetHeaderBlock()->id()
              ? 2
              : 1);
  condition_block->terminator()->SetInOperand(1, {to_continue_block});
  condition_block->terminator()->SetInOperand(
      2, {GetAfterLoop()->GetHeaderBlock()->id()});
}

void LoopPeeling::PeelBefore(ir::Instruction* factor) {
  DuplicateLoop();

  InsertIterator(factor);

  FixExitCondition([factor, this](ir::BasicBlock* condition_block) {
    InstructionBuilder builder(context_, &*condition_block->tail(),
                               ir::IRContext::kAnalysisDefUse |
                                   ir::IRContext::kAnalysisInstrToBlockMapping);
    return builder
        .AddLessThan(extra_induction_variable_->result_id(),
                     factor->result_id())
        ->result_id();
  });
}

void LoopPeeling::PeelAfter(ir::Instruction* factor,
                            ir::Instruction* iteration_count) {
  DuplicateLoop();

  InsertIterator(factor);

  FixExitCondition([factor, iteration_count,
                    this](ir::BasicBlock* condition_block) {
    InstructionBuilder builder(context_, &*condition_block->tail(),
                               ir::IRContext::kAnalysisDefUse |
                                   ir::IRContext::kAnalysisInstrToBlockMapping);
    // Build the following check: extra_induction_variable_ + factor <
    // iteration_count
    return builder
        .AddLessThan(builder
                         .AddIAdd(extra_induction_variable_->type_id(),
                                  extra_induction_variable_->result_id(),
                                  factor->result_id())
                         ->result_id(),
                     iteration_count->result_id())
        ->result_id();
  });
}

}  // namespace opt
}  // namespace spvtools
