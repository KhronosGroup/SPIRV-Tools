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

#ifndef LIBSPIRV_OPT_CFG_H_
#define LIBSPIRV_OPT_CFG_H_

#include "basic_block.h"

#include <list>
#include <unordered_map>

namespace spvtools {
namespace ir {

class CFG {
 public:
  CFG(ir::Module* module);

  // Return the module described by this CFG.
  ir::Module* get_module() const { return module_; }

  // Return the list of predecesors for basic block with label |blkid|.
  // TODO(dnovillo): Move this to ir::BasicBlock.
  const std::vector<uint32_t>& preds(uint32_t blk_id) const {
    return label2preds_.at(blk_id);
  }

  // Return a pointer to the basic block instance corresponding to the label
  // |blk_id|.
  ir::BasicBlock* block(uint32_t blk_id) const { return id2block_.at(blk_id); }

  // Return the pseudo entry and exit blocks.
  const ir::BasicBlock* pseudo_entry_block() const {
    return &pseudo_entry_block_;
  }
  ir::BasicBlock* pseudo_entry_block() { return &pseudo_entry_block_; }

  const ir::BasicBlock* pseudo_exit_block() const {
    return &pseudo_exit_block_;
  }
  ir::BasicBlock* pseudo_exit_block() { return &pseudo_exit_block_; }

  // Return true if |block_ptr| is the pseudo-entry block.
  bool IsPseudoEntryBlock(ir::BasicBlock* block_ptr) const {
    return block_ptr == &pseudo_entry_block_;
  }

  // Return true if |block_ptr| is the pseudo-exit block.
  bool IsPseudoExitBlock(ir::BasicBlock* block_ptr) const {
    return block_ptr == &pseudo_exit_block_;
  }

  // Compute structured block order into |order| for |func| starting at |root|.
  // This order has the property that dominators come before all blocks they
  // dominate and merge blocks come after all blocks that are in the control
  // constructs of their header.
  void ComputeStructuredOrder(ir::Function* func, ir::BasicBlock* root,
                              std::list<ir::BasicBlock*>* order);

 private:
  using cbb_ptr = const ir::BasicBlock*;

  // Compute structured successors for function |func|. A block's structured
  // successors are the blocks it branches to together with its declared merge
  // block and continue block if it has them. When order matters, the merge
  // block and continue block always appear first. This assures correct depth
  // first search in the presence of early returns and kills. If the successor
  // vector contain duplicates of the merge or continue blocks, they are safely
  // ignored by DFS.
  void ComputeStructuredSuccessors(ir::Function* func);

  // Module for this CFG.
  ir::Module* module_;

  // Map from block to its structured successor blocks. See
  // ComputeStructuredSuccessors() for definition.
  std::unordered_map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>>
      block2structured_succs_;

  // Extra block whose successors are all blocks with no predecessors
  // in function.
  ir::BasicBlock pseudo_entry_block_;

  // Augmented CFG Exit Block.
  ir::BasicBlock pseudo_exit_block_;

  // Map from block's label id to its predecessor blocks ids
  std::unordered_map<uint32_t, std::vector<uint32_t>> label2preds_;

  // Map from block's label id to block.
  std::unordered_map<uint32_t, ir::BasicBlock*> id2block_;
};

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_CFG_H_
