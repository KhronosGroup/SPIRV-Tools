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

#ifndef LIBSPIRV_OPT_LOCAL_SINGLE_BLOCK_ELIM_PASS_H_
#define LIBSPIRV_OPT_LOCAL_SINGLE_BLOCK_ELIM_PASS_H_


#include <algorithm>
#include <map>
#include <queue>
#include <utility>
#include <unordered_map>
#include <unordered_set>

#include "basic_block.h"
#include "def_use_manager.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class LocalSingleBlockElimPass : public Pass {
 public:
  LocalSingleBlockElimPass();
  const char* name() const override { return "eliminate-local-single-block"; }
  Status Process(ir::Module*) override;

 private:
  // Module this pass is processing
  ir::Module* module_;

  // Def-Uses for the module we are processing
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  // Set of verified target types
  std::unordered_set<uint32_t> seen_target_vars_;

  // Set of verified non-target types
  std::unordered_set<uint32_t> seen_non_target_vars_;

  // Map from variable to its live store in block
  std::unordered_map<uint32_t, ir::Instruction*> var2store_;

  // Map from variable to its live load in block
  std::unordered_map<uint32_t, ir::Instruction*> var2load_;

  // Set of undeletable variables
  std::unordered_set<uint32_t> pinned_vars_;

  // Next unused ID
  uint32_t next_id_;

  // Save next available id into |module|.
  inline void FinalizeNextId(ir::Module* module) {
    module->SetIdBound(next_id_);
  }

  // Return next available id and calculate next.
  inline uint32_t TakeNextId() {
    return next_id_++;
  }

  // Returns true if |typeInst| is a scalar type
  // or a vector or matrix
  bool IsMathType(const ir::Instruction* typeInst);

  // Returns true if |typeInst| is a math type or a struct or array
  // of a math type.
  bool IsTargetType(const ir::Instruction* typeInst);

  // Given a load or store |ip|, return the pointer instruction.
  // Also return the base variable's id in |varId|.
  ir::Instruction* GetPtr(ir::Instruction* ip, uint32_t* varId);

  // Return true if |varId| is function scope variable of targeted type.
  bool IsTargetVar(uint32_t varId);

  // Replace all instances of |loadInst|'s id with |replId| and delete
  // |loadInst|.
  void ReplaceAndDeleteLoad(ir::Instruction* loadInst, uint32_t replId);

  // Return true if any instruction loads from |varId|
  bool HasLoads(uint32_t varId);

  // Return true if |varId| is not a function variable or if it has
  // a load
  bool IsLiveVar(uint32_t varId);

  // Return true if |storeInst| is not to function variable or if its
  // base variable has a load
  bool IsLiveStore(ir::Instruction* storeInst);

  // Add stores using |ptr_id| to |insts|
  void AddStores(uint32_t ptr_id, std::queue<ir::Instruction*>* insts);

  // Delete |inst| and iterate DCE on all its operands 
  void DCEInst(ir::Instruction* inst);

  // On all entry point functions, within a single block, eliminate
  // loads and stores to function variables where possible. For
  // loads, if previous load or store to same variable, replace
  // load id with previous id and delete load. Finally, check if
  // remaining stores are useless, and delete store and variable
  // where possible.
  bool LocalSingleBlockElim(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_LOCAL_SINGLE_BLOCK_ELIM_PASS_H_

