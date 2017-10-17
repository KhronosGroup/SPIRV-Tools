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
#include "module.h"
#include "mem_pass.h"

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
  Status Process(ir::Module*) override;

 private:
  // Return true if |varId| is variable of |storageClass|.
  bool IsVarOfStorage(uint32_t varId, uint32_t storageClass);

  // Return true if |varId| is variable of function storage class or is
  // private variable and privates can be optimized like locals (see
  // privates_like_local_)
  bool IsLocalVar(uint32_t varId);

  // Return true if |op| is branch instruction
  bool IsBranch(SpvOp op) {
    return op == SpvOpBranch || op == SpvOpBranchConditional;
  }

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

  // Initialize combinator data structures
  void InitCombinatorSets();

  // Return true if core operator |op| has no side-effects. Currently returns
  // true only for shader capability operations.
  // TODO(greg-lunarg): Add kernel and other operators
  bool IsCombinator(uint32_t op) const;

  // Return true if OpExtInst |inst| has no side-effects. Currently returns
  // true only for std.GLSL.450 extensions
  // TODO(greg-lunarg): Add support for other extensions
  bool IsCombinatorExt(ir::Instruction* inst) const;

  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are supported by this pass.
  bool AllExtensionsSupported() const;

  // Kill debug or annotation |inst| if target operand is dead. Return true
  // if inst killed.
  bool KillInstIfTargetDead(ir::Instruction* inst);

  // If |varId| is local, mark all stores of varId as live.
  void ProcessLoad(uint32_t varId);

  // Returns the id of the merge block declared by a merge instruction in 
  // this block |blk|, if any. If none, returns zero. If loop merge, returns
  // the continue target id in |cbid|. Otherwise sets to zero.
  uint32_t MergeBlockIdIfAny(const ir::BasicBlock& blk, uint32_t* cbid) const;

  // Compute structured successors for function |func|.
  // A block's structured successors are the blocks it branches to
  // together with its declared merge block if it has one.
  // When order matters, the merge block always appears first and if
  // a loop merge block, the continue target always appears second.
  // This assures correct depth first search in the presence of early 
  // returns and kills. If the successor vector contain duplicates
  // of the merge and continue blocks, they are safely ignored by DFS.
  void ComputeStructuredSuccessors(ir::Function* func);

  // Compute structured block order |order| for function |func|. This order
  // has the property that dominators are before all blocks they dominate and
  // merge blocks are after all blocks that are in the control constructs of
  // their header.
  void ComputeStructuredOrder(
      ir::Function* func, std::list<ir::BasicBlock*>* order);

  // If |bp| is structured if header block, return true and set |branchInst|
  // to the conditional branch and |mergeBlockId| to the merge block.
  bool IsStructuredIfHeader(ir::BasicBlock* bp,
    ir::Instruction** mergeInst, ir::Instruction** branchInst,
    uint32_t* mergeBlockId);

  // Initialize block2branch_ and block2merge_ using |structuredOrder| to
  // order blocks.
  void ComputeBlock2BranchMaps(std::list<ir::BasicBlock*>& structuredOrder);

  // Initialize inst2block_ for |func|.
  void ComputeInst2BlockMap(ir::Function* func);

  // Add branch to |labelId| to end of block |bp|.
  void AddBranch(uint32_t labelId, ir::BasicBlock* bp);

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

  void Initialize(ir::Module* module);
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

  // Map from block to most immediate controlling structured conditional
  // branch.
  std::unordered_map<ir::BasicBlock*, ir::Instruction*> block2branch_;

  // Map from block to most immediate controlling conditional merge.
  std::unordered_map<ir::BasicBlock*, ir::Instruction*> block2merge_;

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

  // Opcodes of shader capability core executable instructions
  // without side-effect. This is a whitelist of operators
  // that can safely be left unmarked as live at the beginning of
  // aggressive DCE.
  std::unordered_set<uint32_t> combinator_ops_shader_;

  // Opcodes of GLSL_std_450 extension executable instructions
  // without side-effect. This is a whitelist of operators
  // that can safely be left unmarked as live at the beginning of
  // aggressive DCE.
  std::unordered_set<uint32_t> combinator_ops_glsl_std_450_;

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;

  // Set id for glsl_std_450 extension instructions
  uint32_t glsl_std_450_id_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_

