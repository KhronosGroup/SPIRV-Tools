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

#include "cfg.h"
#include "cfa.h"
#include "ir_context.h"
#include "module.h"

namespace spvtools {
namespace ir {

namespace {

// Universal Limit of ResultID + 1
const int kInvalidId = 0x400000;

}  // namespace

CFG::CFG(ir::Module* module)
    : module_(module),
      pseudo_entry_block_(std::unique_ptr<ir::Instruction>(
          new ir::Instruction(module->context(), SpvOpLabel, 0, 0, {}))),
      pseudo_exit_block_(std::unique_ptr<ir::Instruction>(new ir::Instruction(
          module->context(), SpvOpLabel, 0, kInvalidId, {}))) {
  for (auto& fn : *module) {
    for (auto& blk : fn) {
      RegisterBlock(&blk);
    }
  }
}

void CFG::AddEdges(ir::BasicBlock* blk) {
  uint32_t blk_id = blk->id();
  // Force the creation of an entry, not all basic block have predecessors
  // (such as the entry blocks and some unreachables).
  label2preds_[blk_id];
  const auto* const_blk = blk;
  const_blk->ForEachSuccessorLabel(
      [blk_id, this](const uint32_t succ_id) { AddEdge(blk_id, succ_id); });
}

void CFG::RemoveNonExistingEdges(uint32_t blk_id) {
  std::vector<uint32_t> updated_pred_list;
  for (uint32_t id : preds(blk_id)) {
    const ir::BasicBlock* pred_blk = block(id);
    bool has_branch = false;
    pred_blk->ForEachSuccessorLabel([&has_branch, blk_id](uint32_t succ) {
      if (succ == blk_id) {
        has_branch = true;
      }
    });
    if (has_branch) updated_pred_list.push_back(id);
  }

  label2preds_.at(blk_id) = std::move(updated_pred_list);
}

void CFG::ComputeStructuredOrder(ir::Function* func, ir::BasicBlock* root,
                                 std::list<ir::BasicBlock*>* order) {
  assert(module_->context()->get_feature_mgr()->HasCapability(
             SpvCapabilityShader) &&
         "This only works on structured control flow");

  // Compute structured successors and do DFS.
  ComputeStructuredSuccessors(func);
  auto ignore_block = [](cbb_ptr) {};
  auto ignore_edge = [](cbb_ptr, cbb_ptr) {};
  auto get_structured_successors = [this](const ir::BasicBlock* b) {
    return &(block2structured_succs_[b]);
  };

  // TODO(greg-lunarg): Get rid of const_cast by making moving const
  // out of the cfa.h prototypes and into the invoking code.
  auto post_order = [&](cbb_ptr b) {
    order->push_front(const_cast<ir::BasicBlock*>(b));
  };
  spvtools::CFA<ir::BasicBlock>::DepthFirstTraversal(
      root, get_structured_successors, ignore_block, post_order, ignore_edge);
}

void CFG::ForEachBlockInReversePostOrder(
    BasicBlock* bb, const std::function<void(BasicBlock*)>& f) {
  std::vector<BasicBlock*> po;
  std::unordered_set<BasicBlock*> seen;
  ComputePostOrderTraversal(bb, &po, &seen);

  for (auto current_bb = po.rbegin(); current_bb != po.rend(); ++current_bb) {
    if (!IsPseudoExitBlock(*current_bb) && !IsPseudoEntryBlock(*current_bb)) {
      f(*current_bb);
    }
  }
}

void CFG::ComputeStructuredSuccessors(ir::Function* func) {
  block2structured_succs_.clear();
  for (auto& blk : *func) {
    // If no predecessors in function, make successor to pseudo entry.
    if (label2preds_[blk.id()].size() == 0)
      block2structured_succs_[&pseudo_entry_block_].push_back(&blk);

    // If header, make merge block first successor and continue block second
    // successor if there is one.
    uint32_t mbid = blk.MergeBlockIdIfAny();
    if (mbid != 0) {
      block2structured_succs_[&blk].push_back(id2block_[mbid]);
      uint32_t cbid = blk.ContinueBlockIdIfAny();
      if (cbid != 0) block2structured_succs_[&blk].push_back(id2block_[cbid]);
    }

    // Add true successors.
    const auto& const_blk = blk;
    const_blk.ForEachSuccessorLabel([&blk, this](const uint32_t sbid) {
      block2structured_succs_[&blk].push_back(id2block_[sbid]);
    });
  }
}

void CFG::ComputePostOrderTraversal(BasicBlock* bb, vector<BasicBlock*>* order,
                                    unordered_set<BasicBlock*>* seen) {
  seen->insert(bb);
  static_cast<const BasicBlock*>(bb)->ForEachSuccessorLabel(
      [&order, &seen, this](const uint32_t sbid) {
        BasicBlock* succ_bb = id2block_[sbid];
        if (!seen->count(succ_bb)) {
          ComputePostOrderTraversal(succ_bb, order, seen);
        }
      });
  order->push_back(bb);
}

}  // namespace ir
}  // namespace spvtools
