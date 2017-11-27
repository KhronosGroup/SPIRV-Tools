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
      uint32_t blkId = blk.id();
      id2block_[blkId] = &blk;
      blk.ForEachSuccessorLabel([&blkId, this](uint32_t sbid) {
        label2preds_[sbid].push_back(blkId);
      });
    }
  }
}

void CFG::ComputeStructuredOrder(ir::Function* func, ir::BasicBlock* root,
                                 std::list<ir::BasicBlock*>* order) {
  assert(module_->HasCapability(SpvCapabilityShader) &&
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
    blk.ForEachSuccessorLabel([&blk, this](uint32_t sbid) {
      block2structured_succs_[&blk].push_back(id2block_[sbid]);
    });
  }
}

}  // namespace ir
}  // namespace spvtools
