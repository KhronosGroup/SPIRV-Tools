// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#ifndef LIBSPIRV_OPT_DEAD_BRANCH_ELIM_PASS_H_
#define LIBSPIRV_OPT_DEAD_BRANCH_ELIM_PASS_H_

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_block.h"
#include "def_use_manager.h"
#include "mem_pass.h"
#include "module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class DeadBranchElimPass : public MemPass {
  using cbb_ptr = const ir::BasicBlock*;

 public:
  DeadBranchElimPass();
  const char* name() const override { return "eliminate-dead-branches"; }
  Status Process(ir::IRContext* context) override;

  ir::IRContext::Analysis GetPreservedAnalyses() override {
    return ir::IRContext::kAnalysisDefUse;
  }

 private:
  // If |condId| is boolean constant, return conditional value in |condVal| and
  // return true, otherwise return false.
  bool GetConstCondition(uint32_t condId, bool* condVal);

  // If |valId| is a 32-bit integer constant, return value via |value| and
  // return true, otherwise return false.
  bool GetConstInteger(uint32_t valId, uint32_t* value);

  // Add branch to |labelId| to end of block |bp|.
  void AddBranch(uint32_t labelId, ir::BasicBlock* bp);

  // For function |func|, look for BranchConditionals with constant condition
  // and convert to a Branch to the indicated label. Delete resulting dead
  // blocks. Note some such branches and blocks may be left to avoid creating
  // invalid control flow.
  // TODO(greg-lunarg): Remove remaining constant conditional branches and dead
  // blocks.
  bool EliminateDeadBranches(ir::Function* func);

  // Returns the basic block containing |id|.
  // Note: this pass only requires correct instruction block mappings for the
  // input. This pass does not preserve the block mapping, so it is not kept
  // up-to-date during processing.
  ir::BasicBlock* GetParentBlock(uint32_t id);

  // Marks live blocks reachable from the entry of |func|. Simplifies constant
  // branches and switches as it proceeds, to limit the number of live blocks.
  // It is careful not to eliminate backedges even if they are dead, but the
  // header is live. Likewise, unreachable merge blocks named in live merge
  // instruction must be retained (though they may be clobbered).
  bool MarkLiveBlocks(ir::Function* func,
                      std::unordered_set<ir::BasicBlock*>* live_blocks);

  // Checks for unreachable merge and continue blocks with live headers, those
  // blocks must be retained. Continues are tracked separately so that when
  // updating live phi nodes with an edge from a continue they can be replaced
  // with an undef (because we clobber the instructions inside continue block).
  //
  // |unreachable_continues| maps continue targets that cannot be reached to
  // merge instruction that declares them.
  void MarkUnreachableStructuredTargets(
      const std::unordered_set<ir::BasicBlock*>& live_blocks,
      std::unordered_set<ir::BasicBlock*>* unreachable_merges,
      std::unordered_map<ir::BasicBlock*, ir::BasicBlock*>*
          unreachable_continues);

  // Fix phis in reachable blocks so that only live (or unremovable) incoming
  // edges are present. If the block now only has a single live incoming edge,
  // remove the phi and replace its uses with its data input.
  //
  // |unreachable_continues| maps continue targets that cannot be reached to
  // merge instruction that declares them.
  bool FixPhiNodesInLiveBlocks(
      ir::Function* func,
      const std::unordered_set<ir::BasicBlock*>& live_blocks,
      const std::unordered_map<ir::BasicBlock*, ir::BasicBlock*>&
          unreachable_continues);

  // Erases dead blocks. Any block captured in |unreachable_merges| or
  // |unreachable_continues| is a dead block that is required to remain due to
  // a live merge instruction in the corresponding header. These blocks will
  // have their instructions clobbered and will become a label and terminator.
  // Unreachable merge blocks are terminated by OpReachable, while unreachable
  // continue blocks are terminated by an unconditional branch to the header.
  // Otherwise, blocks are dead if not explicitly captured in |live_blocks| and
  // are totally removed.
  //
  // |unreachable_continues| maps continue targets that cannot be reached to
  // merge instruction that declares them.
  bool EraseDeadBlocks(
      ir::Function* func,
      const std::unordered_set<ir::BasicBlock*>& live_blocks,
      const std::unordered_set<ir::BasicBlock*>& unreachable_merges,
      const std::unordered_map<ir::BasicBlock*, ir::BasicBlock*>&
          unreachable_continues);

  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  void Initialize(ir::IRContext* c);
  Pass::Status ProcessImpl();

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DEAD_BRANCH_ELIM_PASS_H_
