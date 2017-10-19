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

#ifndef LIBSPIRV_OPT_CFG_CLEANUP_PASS_H_
#define LIBSPIRV_OPT_CFG_CLEANUP_PASS_H_

#include "function.h"
#include "mem_pass.h"
#include "module.h"

namespace spvtools {
namespace opt {

class CFGCleanupPass : public MemPass {
 public:
  CFGCleanupPass() = default;
  const char* name() const override { return "cfg-cleanup"; }
  Status Process(ir::Module*) override;

 private:
  // Call all the cleanup helper functions on |func|.
  bool CFGCleanup(ir::Function* func);

  // Remove all the unreachable basic blocks in |func|.
  bool RemoveUnreachableBlocks(ir::Function* func);

  // Remove the block pointed by the iterator |*bi|. This also removes
  // all the instructions in the pointed-to block.
  void RemoveBlock(ir::Function::iterator* bi);

  // Initialize the pass.
  void Initialize(ir::Module* module);

  // Remove Phi operands in |phi| that are coming from blocks not in
  // |reachable_blocks|.
  void RemovePhiOperands(ir::Instruction* phi,
                         std::unordered_set<ir::BasicBlock*> reachable_blocks);

  // Return the next available Id and increment it. TODO(dnovillo): Refactor
  // into a new type pool manager to be used for all passes.
  inline uint32_t TakeNextId() { return next_id_++; }

  // Save next available id into |module|. TODO(dnovillo): Refactor
  // into a new type pool manager to be used for all passes.
  inline void FinalizeNextId(ir::Module* module) {
    module->SetIdBound(next_id_);
  }

  // Return undef in function for type. Create and insert an undef after the
  // first non-variable in the function if it doesn't already exist. Add
  // undef to function undef map. TODO(dnovillo): Refactor into a new
  // type pool manager to be used for all passes.
  uint32_t TypeToUndef(uint32_t type_id);

  // Map from block's label id to block. TODO(dnovillo): Basic blocks ought to
  // have basic blocks in their pred/succ list.
  std::unordered_map<uint32_t, ir::BasicBlock*> label2block_;

  // Map from an instruction result ID to the block that holds it.
  // TODO(dnovillo): This would be unnecessary if ir::Instruction instances
  // knew what basic block they belong to.
  std::unordered_map<uint32_t, ir::BasicBlock*> def_block_;

  // Map from type to undef values. TODO(dnovillo): This is replicated from
  // class LocalMultiStoreElimPass. It should be refactored into a type
  // pool manager.
  std::unordered_map<uint32_t, uint32_t> type2undefs_;

  // Next unused ID. TODO(dnovillo): Refactor this to some common utility class.
  // Seems to be implemented in very many passes.
  uint32_t next_id_;
};

}  // namespace opt
}  // namespace spvtools

#endif
