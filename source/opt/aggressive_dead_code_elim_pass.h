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

#ifndef LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_
#define LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_

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
class AggressiveDCEPass : public MemPass {
  using cbb_ptr = const ir::BasicBlock*;

 public:
  using GetBlocksFunction =
      std::function<std::vector<ir::BasicBlock*>*(const ir::BasicBlock*)>;

  AggressiveDCEPass();
  const char* name() const override { return "eliminate-dead-code-aggressive"; }
  Status Process(ir::IRContext* c) override;

  ir::IRContext::Analysis GetPreservedAnalyses() override {
    return ir::IRContext::kAnalysisDefUse;
  }

 private:
  // Return true if |varId| is variable of |storageClass|.
  bool IsVarOfStorage(uint32_t varId, uint32_t storageClass);

  // Return true if |varId| is variable of function storage class or is
  // private variable and privates can be optimized like locals (see
  // privates_like_local_)
  bool IsLocalVar(uint32_t varId);

  // Return true if |inst| is marked live
  bool IsLive(ir::Instruction* inst) {
    return live_insts_.find(inst) != live_insts_.end();
  }

  // Add |inst| to worklist_ and live_insts_.
  void AddToWorklist(ir::Instruction* inst) {
    worklist_.push(inst);
    live_insts_.insert(inst);
  }

  // Add all store instruction which use |ptrId|, directly or indirectly,
  // to the live instruction worklist.
  void AddStores(uint32_t ptrId);

  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are supported by this pass.
  bool AllExtensionsSupported() const;

  // Kill debug or annotation |inst| if target operand is dead. Return true
  // if inst killed.
  bool KillInstIfTargetDead(ir::Instruction* inst);

  // If |varId| is local, mark all stores of varId as live.
  void ProcessLoad(uint32_t varId);

  // If |bp| is structured if header block, return true and set |branchInst|
  // to the conditional branch and |mergeBlockId| to the merge block.
  bool IsStructuredIfHeader(ir::BasicBlock* bp, ir::Instruction** mergeInst,
                            ir::Instruction** branchInst,
                            uint32_t* mergeBlockId);

  // Initialize block2branch_ and block2merge_ using |structuredOrder| to
  // order blocks.
  void ComputeBlock2HeaderMaps(std::list<ir::BasicBlock*>& structuredOrder);

  // Initialize inst2block_ for |func|.
  void ComputeInst2BlockMap(ir::Function* func);

  // Add branch to |labelId| to end of block |bp|.
  void AddBranch(uint32_t labelId, ir::BasicBlock* bp);

  // Add all branches to |labelId| to worklist if not already live
  void AddBranchesToWorklist(uint32_t labelId);

  // For function |func|, mark all Stores to non-function-scope variables
  // and block terminating instructions as live. Recursively mark the values
  // they use. When complete, delete any non-live instructions. Return true
  // if the function has been modified.
  //
  // Note: This function does not delete useless control structures. All
  // existing control structures will remain. This can leave not-insignificant
  // sequences of ultimately useless code.
  // TODO(): Remove useless control constructs.
  bool AggressiveDCE(ir::Function* func);

  void Initialize(ir::IRContext* c);
  Pass::Status ProcessImpl();

  // True if current function has a call instruction contained in it
  bool call_in_func_;

  // True if current function is an entry point
  bool func_is_entry_point_;

  // True if current function is entry point and has no function calls.
  bool private_like_local_;

  // Live Instruction Worklist.  An instruction is added to this list
  // if it might have a side effect, either directly or indirectly.
  // If we don't know, then add it to this list.  Instructions are
  // removed from this list as the algorithm traces side effects,
  // building up the live instructions set |live_insts_|.
  std::queue<ir::Instruction*> worklist_;

  // Map from block to the branch instruction in the header of the most
  // immediate controlling structured if.
  std::unordered_map<ir::BasicBlock*, ir::Instruction*> block2headerBranch_;

  // Map from block to the merge instruction in the header of the most
  // immediate controlling structured if.
  std::unordered_map<ir::BasicBlock*, ir::Instruction*> block2headerMerge_;

  // Map from instruction containing block
  std::unordered_map<ir::Instruction*, ir::BasicBlock*> inst2block_;

  // Map from block's label id to block.
  std::unordered_map<uint32_t, ir::BasicBlock*> id2block_;

  // Map from block to its structured successor blocks. See
  // ComputeStructuredSuccessors() for definition.
  std::unordered_map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>>
      block2structured_succs_;

  // Store instructions to variables of private storage
  std::vector<ir::Instruction*> private_stores_;

  // Live Instructions
  std::unordered_set<const ir::Instruction*> live_insts_;

  // Live Local Variables
  std::unordered_set<uint32_t> live_local_vars_;

  // Dead instructions. Use for debug cleanup.
  std::unordered_set<const ir::Instruction*> dead_insts_;

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_
