// Copyright (c) 2017 Google Inc.
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

#include "opt/loop_descriptor.h"
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "opt/cfg.h"
#include "opt/dominator_tree.h"
#include "opt/ir_builder.h"
#include "opt/ir_context.h"
#include "opt/iterator.h"
#include "opt/make_unique.h"
#include "opt/tree_iterator.h"

namespace spvtools {
namespace ir {

Loop::Loop(IRContext* context, opt::DominatorAnalysis* dom_analysis,
           BasicBlock* header, BasicBlock* continue_target,
           BasicBlock* merge_target)
    : context_(context),
      loop_header_(header),
      loop_continue_(continue_target),
      loop_merge_(merge_target),
      loop_preheader_(nullptr),
      parent_(nullptr) {
  assert(context);
  assert(dom_analysis);
  loop_preheader_ = FindLoopPreheader(dom_analysis);
  AddBasicBlockToLoop(header);
  AddBasicBlockToLoop(continue_target);
}

BasicBlock* Loop::FindLoopPreheader(opt::DominatorAnalysis* dom_analysis) {
  CFG* cfg = context_->cfg();
  opt::DominatorTree& dom_tree = dom_analysis->GetDomTree();
  opt::DominatorTreeNode* header_node = dom_tree.GetTreeNode(loop_header_);

  // The loop predecessor.
  BasicBlock* loop_pred = nullptr;

  auto header_pred = cfg->preds(loop_header_->id());
  for (uint32_t p_id : header_pred) {
    opt::DominatorTreeNode* node = dom_tree.GetTreeNode(p_id);
    if (node && !dom_tree.Dominates(header_node, node)) {
      // The predecessor is not part of the loop, so potential loop preheader.
      if (loop_pred && node->bb_ != loop_pred) {
        // If we saw 2 distinct predecessors that are outside the loop, we don't
        // have a loop preheader.
        return nullptr;
      }
      loop_pred = node->bb_;
    }
  }
  // Safe guard against invalid code, SPIR-V spec forbids loop with the entry
  // node as header.
  assert(loop_pred && "The header node is the entry block ?");

  // So we have a unique basic block that can enter this loop.
  // If this loop is the unique successor of this block, then it is a loop
  // preheader.
  bool is_preheader = true;
  uint32_t loop_header_id = loop_header_->id();
  const auto* const_loop_pred = loop_pred;
  const_loop_pred->ForEachSuccessorLabel(
      [&is_preheader, loop_header_id](const uint32_t id) {
        if (id != loop_header_id) is_preheader = false;
      });
  if (is_preheader) return loop_pred;
  return nullptr;
}

bool Loop::IsInsideLoop(Instruction* inst) const {
  const BasicBlock* parent_block = context_->get_instr_block(inst);
  if (!parent_block) return false;
  return IsInsideLoop(parent_block);
}

bool Loop::IsBasicBlockInLoopSlow(const BasicBlock* bb) {
  assert(bb->GetParent() && "The basic block does not belong to a function");

  opt::DominatorAnalysis* dom_analysis =
      context_->GetDominatorAnalysis(bb->GetParent(), *context_->cfg());
  if (!dom_analysis->Dominates(GetHeaderBlock(), bb)) return false;

  opt::PostDominatorAnalysis* postdom_analysis =
      context_->GetPostDominatorAnalysis(bb->GetParent(), *context_->cfg());
  if (!postdom_analysis->Dominates(GetMergeBlock(), bb)) return false;
  return true;
}

BasicBlock* Loop::GetOrCreatePreHeaderBlock() {
  if (loop_preheader_) return loop_preheader_;

  Function* fn = loop_header_->GetParent();
  // Find the insertion point for the preheader.
  Function::iterator header_it =
      std::find_if(fn->begin(), fn->end(),
                   [this](BasicBlock& bb) { return &bb == loop_header_; });
  assert(header_it != fn->end());

  // Create the preheader basic block.
  loop_preheader_ = &*header_it.InsertBefore(std::unique_ptr<ir::BasicBlock>(
      new ir::BasicBlock(std::unique_ptr<ir::Instruction>(new ir::Instruction(
          context_, SpvOpLabel, 0, context_->TakeNextId(), {})))));
  loop_preheader_->SetParent(fn);
  uint32_t loop_preheader_id = loop_preheader_->id();

  // Redirect the branches and patch the phi:
  //  - For each phi instruction in the header:
  //    - If the header has only 1 out-of-loop incoming branch:
  //      - Change the incomning branch to be the preheader.
  //    - If the header has more than 1 out-of-loop incoming branch:
  //      - Create a new phi in the preheader, gathering all out-of-loops
  //      incoming values;
  //      - Patch the header phi instruction to use the preheader phi
  //      instruction;
  //  - Redirect all edges coming from outside the loop to the preheader.
  opt::InstructionBuilder builder(
      context_, loop_preheader_,
      ir::IRContext::kAnalysisDefUse |
          ir::IRContext::kAnalysisInstrToBlockMapping);
  // Patch all the phi instructions.
  loop_header_->ForEachPhiInst([&builder, this](Instruction* phi) {
    std::vector<uint32_t> preheader_phi_ops;
    std::vector<uint32_t> header_phi_ops;
    for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
      uint32_t def_id = phi->GetSingleWordInOperand(i);
      uint32_t branch_id = phi->GetSingleWordInOperand(i + 1);
      if (IsInsideLoop(branch_id)) {
        header_phi_ops.push_back(def_id);
        header_phi_ops.push_back(branch_id);
      } else {
        preheader_phi_ops.push_back(def_id);
        preheader_phi_ops.push_back(branch_id);
      }
    }

    Instruction* preheader_insn_def = nullptr;
    // Create a phi instruction if and only if the preheader_phi_ops has more
    // than one pair.
    if (preheader_phi_ops.size() > 2)
      preheader_insn_def = builder.AddPhi(phi->type_id(), preheader_phi_ops);
    else
      preheader_insn_def =
          context_->get_def_use_mgr()->GetDef(preheader_phi_ops[0]);
    // Build the new incoming edge.
    header_phi_ops.push_back(preheader_insn_def->result_id());
    header_phi_ops.push_back(loop_preheader_->id());
    // Rewrite operands of the header's phi instruction.
    uint32_t idx = 0;
    for (; idx < header_phi_ops.size(); idx++)
      phi->SetInOperand(idx, {header_phi_ops[idx]});
    // Remove extra operands, from last to first (more efficient).
    for (uint32_t j = phi->NumInOperands() - 1; j >= idx; j--)
      phi->RemoveInOperand(j);
  });
  // Branch from the preheader to the header.
  builder.AddBranch(loop_header_->id());

  // Redirect all out of loop branches to the header to the preheader.
  CFG* cfg = context_->cfg();
  cfg->RegisterBlock(loop_preheader_);
  for (uint32_t pred_id : cfg->preds(loop_header_->id())) {
    if (pred_id == loop_preheader_->id()) continue;
    if (IsInsideLoop(pred_id)) continue;
    BasicBlock* pred = cfg->block(pred_id);
    pred->ForEachSuccessorLabel([this, loop_preheader_id](uint32_t* id) {
      if (*id == loop_header_->id()) *id = loop_preheader_id;
    });
    cfg->AddEdge(pred_id, loop_preheader_id);
  }
  // Delete predecessors that are no longer predecessors of the loop header.
  cfg->RemoveNonExistingEdges(loop_header_->id());
  // Update the loop descriptors.
  if (HasParent()) {
    GetParent()->AddBasicBlock(loop_preheader_);
    context_->GetLoopDescriptor(fn)->SetBasicBlockToLoop(loop_preheader_->id(),
                                                         GetParent());
  }

  context_->InvalidateAnalysesExceptFor(
      builder.GetPreservedAnalysis() |
      ir::IRContext::Analysis::kAnalysisLoopAnalysis |
      ir::IRContext::kAnalysisCFG);

  return loop_preheader_;
}

void Loop::SetLatchBlock(BasicBlock* latch) {
#ifndef NDEBUG
  assert(latch->GetParent() && "The basic block does not belong to a function");

  const auto* const_latch = latch;
  const_latch->ForEachSuccessorLabel([this](uint32_t id) {
    assert((!IsInsideLoop(id) || id == GetHeaderBlock()->id()) &&
           "A predecessor of the continue block does not belong to the loop");
  });
#endif  // NDEBUG
  assert(IsInsideLoop(latch) && "The continue block is not in the loop");

  SetLatchBlockImpl(latch);
}

void Loop::SetMergeBlock(BasicBlock* merge) {
#ifndef NDEBUG
  assert(merge->GetParent() && "The basic block does not belong to a function");
  CFG& cfg = *merge->GetParent()->GetParent()->context()->cfg();

  for (uint32_t pred : cfg.preds(merge->id())) {
    assert(IsInsideLoop(pred) &&
           "A predecessor of the merge block does not belong to the loop");
  }
  assert(!IsInsideLoop(merge) && "The merge block is in the loop");
#endif  // NDEBUG

  SetMergeBlockImpl(merge);
  if (GetHeaderBlock()->GetLoopMergeInst()) {
    UpdateLoopMergeInst();
  }
}

void Loop::GetExitBlocks(std::unordered_set<uint32_t>* exit_blocks) const {
  ir::CFG* cfg = context_->cfg();
  exit_blocks->clear();

  for (uint32_t bb_id : GetBlocks()) {
    const spvtools::ir::BasicBlock* bb = cfg->block(bb_id);
    bb->ForEachSuccessorLabel([exit_blocks, this](uint32_t succ) {
      if (!IsInsideLoop(succ)) {
        exit_blocks->insert(succ);
      }
    });
  }
}

void Loop::GetMergingBlocks(
    std::unordered_set<uint32_t>* merging_blocks) const {
  assert(GetMergeBlock() && "This loop is not structured");
  ir::CFG* cfg = context_->cfg();
  merging_blocks->clear();

  std::stack<const ir::BasicBlock*> to_visit;
  to_visit.push(GetMergeBlock());
  while (!to_visit.empty()) {
    const ir::BasicBlock* bb = to_visit.top();
    to_visit.pop();
    merging_blocks->insert(bb->id());
    for (uint32_t pred_id : cfg->preds(bb->id())) {
      if (!IsInsideLoop(pred_id) && !merging_blocks->count(pred_id)) {
        to_visit.push(cfg->block(pred_id));
      }
    }
  }
}

bool Loop::IsLCSSA() const {
  ir::CFG* cfg = context_->cfg();
  opt::analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

  std::unordered_set<uint32_t> exit_blocks;
  GetExitBlocks(&exit_blocks);

  // Declare ir_context so we can capture context_ in the below lambda
  ir::IRContext* ir_context = context_;

  for (uint32_t bb_id : GetBlocks()) {
    for (Instruction& insn : *cfg->block(bb_id)) {
      // All uses must be either:
      //  - In the loop;
      //  - In an exit block and in a phi instruction.
      if (!def_use_mgr->WhileEachUser(
              &insn,
              [&exit_blocks, ir_context, this](ir::Instruction* use) -> bool {
                BasicBlock* parent = ir_context->get_instr_block(use);
                assert(parent && "Invalid analysis");
                if (IsInsideLoop(parent)) return true;
                if (use->opcode() != SpvOpPhi) return false;
                return exit_blocks.count(parent->id());
              }))
        return false;
    }
  }
  return true;
}

bool Loop::ShouldHoistInstruction(IRContext* context, Instruction* inst) {
  return AreAllOperandsOutsideLoop(context, inst) &&
         inst->IsOpcodeCodeMotionSafe();
}

bool Loop::AreAllOperandsOutsideLoop(IRContext* context, Instruction* inst) {
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  bool all_outside_loop = true;

  const std::function<void(uint32_t*)> operand_outside_loop =
      [this, &def_use_mgr, &all_outside_loop](uint32_t* id) {
        if (this->IsInsideLoop(def_use_mgr->GetDef(*id))) {
          all_outside_loop = false;
          return;
        }
      };

  inst->ForEachInId(operand_outside_loop);
  return all_outside_loop;
}

LoopDescriptor::LoopDescriptor(const Function* f) : loops_() {
  PopulateList(f);
}

LoopDescriptor::~LoopDescriptor() { ClearLoops(); }

void LoopDescriptor::PopulateList(const Function* f) {
  IRContext* context = f->GetParent()->context();
  opt::DominatorAnalysis* dom_analysis =
      context->GetDominatorAnalysis(f, *context->cfg());

  ClearLoops();

  // Post-order traversal of the dominator tree to find all the OpLoopMerge
  // instructions.
  opt::DominatorTree& dom_tree = dom_analysis->GetDomTree();
  for (opt::DominatorTreeNode& node :
       ir::make_range(dom_tree.post_begin(), dom_tree.post_end())) {
    Instruction* merge_inst = node.bb_->GetLoopMergeInst();
    if (merge_inst) {
      // The id of the merge basic block of this loop.
      uint32_t merge_bb_id = merge_inst->GetSingleWordOperand(0);

      // The id of the continue basic block of this loop.
      uint32_t continue_bb_id = merge_inst->GetSingleWordOperand(1);

      // The merge target of this loop.
      BasicBlock* merge_bb = context->cfg()->block(merge_bb_id);

      // The continue target of this loop.
      BasicBlock* continue_bb = context->cfg()->block(continue_bb_id);

      // The basic block containing the merge instruction.
      BasicBlock* header_bb = context->get_instr_block(merge_inst);

      // Add the loop to the list of all the loops in the function.
      Loop* current_loop =
          new Loop(context, dom_analysis, header_bb, continue_bb, merge_bb);
      loops_.push_back(current_loop);

      // We have a bottom-up construction, so if this loop has nested-loops,
      // they are by construction at the tail of the loop list.
      for (auto itr = loops_.rbegin() + 1; itr != loops_.rend(); ++itr) {
        Loop* previous_loop = *itr;

        // If the loop already has a parent, then it has been processed.
        if (previous_loop->HasParent()) continue;

        // If the current loop does not dominates the previous loop then it is
        // not nested loop.
        if (!dom_analysis->Dominates(header_bb,
                                     previous_loop->GetHeaderBlock()))
          continue;
        // If the current loop merge dominates the previous loop then it is
        // not nested loop.
        if (dom_analysis->Dominates(merge_bb, previous_loop->GetHeaderBlock()))
          continue;

        current_loop->AddNestedLoop(previous_loop);
      }
      opt::DominatorTreeNode* dom_merge_node = dom_tree.GetTreeNode(merge_bb);
      for (opt::DominatorTreeNode& loop_node :
           make_range(node.df_begin(), node.df_end())) {
        // Check if we are in the loop.
        if (dom_tree.Dominates(dom_merge_node, &loop_node)) continue;
        current_loop->AddBasicBlockToLoop(loop_node.bb_);
        basic_block_to_loop_.insert(
            std::make_pair(loop_node.bb_->id(), current_loop));
      }
    }
  }
  for (Loop* loop : loops_) {
    if (!loop->HasParent()) dummy_top_loop_.nested_loops_.push_back(loop);
  }
}

void LoopDescriptor::ClearLoops() {
  for (Loop* loop : loops_) {
    delete loop;
  }
  loops_.clear();
}

}  // namespace ir
}  // namespace spvtools
