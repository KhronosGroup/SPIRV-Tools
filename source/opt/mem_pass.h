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

#ifndef LIBSPIRV_OPT_MEM_PASS_H_
#define LIBSPIRV_OPT_MEM_PASS_H_

#include <algorithm>
#include <list>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_block.h"
#include "def_use_manager.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// A common base class for mem2reg-type passes.  Provides common
// utility functions and supporting state.
class MemPass : public Pass {
 public:
  MemPass();
  virtual ~MemPass() = default;

 protected:
  // Returns true if |typeInst| is a scalar type
  // or a vector or matrix
  bool IsBaseTargetType(const ir::Instruction* typeInst) const;

  // Returns true if |typeInst| is a math type or a struct or array
  // of a math type.
  // TODO(): Add more complex types to convert
  bool IsTargetType(const ir::Instruction* typeInst) const;

  // Returns true if |opcode| is a non-ptr access chain op
  bool IsNonPtrAccessChain(const SpvOp opcode) const;

  // Given the id |ptrId|, return true if the top-most non-CopyObj is
  // a variable, a non-ptr access chain or a parameter of pointer type.
  bool IsPtr(uint32_t ptrId);

  // Given the id of a pointer |ptrId|, return the top-most non-CopyObj.
  // Also return the base variable's id in |varId|.  If no base variable is
  // found, |varId| will be 0.
  ir::Instruction* GetPtr(uint32_t ptrId, uint32_t* varId);

  // Given a load or store |ip|, return the pointer instruction.
  // Also return the base variable's id in |varId|.  If no base variable is
  // found, |varId| will be 0.
  ir::Instruction* GetPtr(ir::Instruction* ip, uint32_t* varId);

  // Return true if all uses of |id| are only name or decorate ops.
  bool HasOnlyNamesAndDecorates(uint32_t id) const;

  // Kill all instructions in block |bp|. Whether or not to kill the label is
  // indicated by |killLabel|.
  void KillAllInsts(ir::BasicBlock* bp, bool killLabel = true);

  // Return true if any instruction loads from |varId|
  bool HasLoads(uint32_t varId) const;

  // Return true if |varId| is not a function variable or if it has
  // a load
  bool IsLiveVar(uint32_t varId) const;

  // Return true if |storeInst| is not a function variable or if its
  // base variable has a load
  bool IsLiveStore(ir::Instruction* storeInst);

  // Add stores using |ptr_id| to |insts|
  void AddStores(uint32_t ptr_id, std::queue<ir::Instruction*>* insts);

  // Delete |inst| and iterate DCE on all its operands if they are now
  // useless. If a load is deleted and its variable has no other loads,
  // delete all its variable's stores.
  void DCEInst(ir::Instruction* inst,
               const std::function<void(ir::Instruction*)>&);

  // Call all the cleanup helper functions on |func|.
  bool CFGCleanup(ir::Function* func);

  // Return true if |op| is supported decorate.
  inline bool IsNonTypeDecorate(uint32_t op) const {
    return (op == SpvOpDecorate || op == SpvOpDecorateId);
  }

  // Return true if |varId| is a previously identified target variable.
  // Return false if |varId| is a previously identified non-target variable.
  //
  // Non-target variables are variable of function scope of a target type that
  // are accessed with constant-index access chains. not accessed with
  // non-constant-index access chains. Also cache non-target variables.
  //
  // If variable is not cached, return true if variable is a function scope
  // variable of target type, false otherwise. Updates caches of target and
  // non-target variables.
  bool IsTargetVar(uint32_t varId);

  // Return undef in function for type. Create and insert an undef after the
  // first non-variable in the function if it doesn't already exist. Add
  // undef to function undef map.
  uint32_t Type2Undef(uint32_t type_id);

  // Insert Phi instructions in the CFG of |func|.  This removes extra
  // load/store operations to local storage while preserving the SSA form of the
  // code.
  Pass::Status InsertPhiInstructions(ir::Function* func);

  // Cache of verified target vars
  std::unordered_set<uint32_t> seen_target_vars_;

  // Cache of verified non-target vars
  std::unordered_set<uint32_t> seen_non_target_vars_;

 private:
  // Return true if all uses of |varId| are only through supported reference
  // operations ie. loads and store. Also cache in supported_ref_vars_.
  // TODO(dnovillo): This function is replicated in other passes and it's
  // slightly different in every pass. Is it possible to make one common
  // implementation?
  bool HasOnlySupportedRefs(uint32_t varId);

  // Patch phis in loop header block |header_id| now that the map is complete
  // for the backedge predecessor |back_id|. Specifically, for each phi, find
  // the value corresponding to the backedge predecessor. That was temporarily
  // set with the variable id that this phi corresponds to. Change this phi
  // operand to the the value which corresponds to that variable in the
  // predecessor map.
  void PatchPhis(uint32_t header_id, uint32_t back_id);

  // Initialize data structures used by EliminateLocalMultiStore for
  // function |func|, specifically block predecessors and target variables.
  void InitSSARewrite(ir::Function* func);

  // Initialize label2ssa_map_ entry for block |block_ptr| with single
  // predecessor.
  void SSABlockInitSinglePred(ir::BasicBlock* block_ptr);

  // Initialize label2ssa_map_ entry for loop header block pointed to
  // |block_itr| by merging entries from all predecessors. If any value
  // ids differ for any variable across predecessors, create a phi function
  // in the block and use that value id for the variable in the new map.
  // Assumes all predecessors have been visited by EliminateLocalMultiStore
  // except the back edge. Use a dummy value in the phi for the back edge
  // until the back edge block is visited and patch the phi value then.
  void SSABlockInitLoopHeader(std::list<ir::BasicBlock*>::iterator block_itr);

  // Initialize label2ssa_map_ entry for multiple predecessor block
  // |block_ptr| by merging label2ssa_map_ entries for all predecessors.
  // If any value ids differ for any variable across predecessors, create
  // a phi function in the block and use that value id for the variable in
  // the new map. Assumes all predecessors have been visited by
  // EliminateLocalMultiStore.
  void SSABlockInitMultiPred(ir::BasicBlock* block_ptr);

  // Initialize the label2ssa_map entry for a block pointed to by |block_itr|.
  // Insert phi instructions into block when necessary. All predecessor
  // blocks must have been visited by EliminateLocalMultiStore except for
  // backedges.
  void SSABlockInit(std::list<ir::BasicBlock*>::iterator block_itr);

  // Return true if variable is loaded in block with |label| or in any
  // succeeding block in structured order.
  bool IsLiveAfter(uint32_t var_id, uint32_t label) const;

  // Remove all the unreachable basic blocks in |func|.
  bool RemoveUnreachableBlocks(ir::Function* func);

  // Remove the block pointed by the iterator |*bi|. This also removes
  // all the instructions in the pointed-to block.
  void RemoveBlock(ir::Function::iterator* bi);

  // Remove Phi operands in |phi| that are coming from blocks not in
  // |reachable_blocks|.
  void RemovePhiOperands(ir::Instruction* phi,
                         std::unordered_set<ir::BasicBlock*> reachable_blocks);

  // Map from block's label id to a map of a variable to its value at the
  // end of the block.
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>
      label2ssa_map_;

  // Set of label ids of visited blocks
  std::unordered_set<uint32_t> visitedBlocks_;

  // Variables that are only referenced by supported operations for this
  // pass ie. loads and stores.
  std::unordered_set<uint32_t> supported_ref_vars_;

  // Map from type to undef
  std::unordered_map<uint32_t, uint32_t> type2undefs_;

  // The Ids of OpPhi instructions that are in a loop header and which require
  // patching of the value for the loop back-edge.
  std::unordered_set<uint32_t> phis_to_patch_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_MEM_PASS_H_
