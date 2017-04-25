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

#ifndef LIBSPIRV_OPT_INLINE_PASS_H_
#define LIBSPIRV_OPT_INLINE_PASS_H_

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "def_use_manager.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InlinePass : public Pass {
 public:
  InlinePass();
  Status Process(ir::Module*) override;

  const char* name() const override { return "inline"; }

 private:
  // Return the next available Id and increment it.
  inline uint32_t TakeNextId() { return next_id_++; }

  // Write the next available Id back to the module.
  inline void FinalizeNextId(ir::Module* module) {
    module->SetIdBound(next_id_);
  }

  // Find pointer to type and storage in module, return its resultId,
  // 0 if not found. TODO(greg-lunarg): Move this into type manager.
  uint32_t FindPointerToType(uint32_t type_id, SpvStorageClass storage_class);

  // Add pointer to type to module and return resultId.
  uint32_t AddPointerToType(uint32_t type_id, SpvStorageClass storage_class);

  // Add unconditional branch to labelId to end of block block_ptr.
  void AddBranch(uint32_t labelId, std::unique_ptr<ir::BasicBlock>* block_ptr);

  // Add store of valId to ptrId to end of block block_ptr.
  void AddStore(uint32_t ptrId, uint32_t valId,
                std::unique_ptr<ir::BasicBlock>* block_ptr);

  // Add load of ptrId into resultId to end of block block_ptr.
  void AddLoad(uint32_t typeId, uint32_t resultId, uint32_t ptrId,
               std::unique_ptr<ir::BasicBlock>* block_ptr);

  // Return new label.
  std::unique_ptr<ir::Instruction> NewLabel(uint32_t label_id);

  // Map callee params to caller args
  void MapParams(ir::Function* calleeFn,
                 ir::UptrVectorIterator<ir::Instruction> call_inst_itr,
                 std::unordered_map<uint32_t, uint32_t>* callee2caller);

  // Clone and map callee locals
  void CloneAndMapLocals(
      ir::Function* calleeFn,
      std::vector<std::unique_ptr<ir::Instruction>>* new_vars,
      std::unordered_map<uint32_t, uint32_t>* callee2caller);

  // Create return variable for callee clone code if needed. Return id
  // if created, otherwise 0.
  uint32_t CreateReturnVar(
      ir::Function* calleeFn,
      std::vector<std::unique_ptr<ir::Instruction>>* new_vars);

  // Return true if instruction must be in the same block that its result
  // is used.
  bool IsSameBlockOp(const ir::Instruction* inst) const;

  // Clone operands which must be in same block as consumer instructions.
  // Look in preCallSB for instructions that need cloning. Look in
  // postCallSB for instructions already cloned. Add cloned instruction
  // to postCallSB.
  void CloneSameBlockOps(
      std::unique_ptr<ir::Instruction>* inst,
      std::unordered_map<uint32_t, uint32_t>* postCallSB,
      std::unordered_map<uint32_t, ir::Instruction*>* preCallSB,
      std::unique_ptr<ir::BasicBlock>* block_ptr);

  // Return in new_blocks the result of inlining the call at call_inst_itr
  // within its block at call_block_itr. The block at call_block_itr can
  // just be replaced with the blocks in new_blocks. Any additional branches
  // are avoided. Debug instructions are cloned along with their callee
  // instructions. Early returns are replaced by a store to a local return
  // variable and a branch to a (created) exit block where the local variable
  // is returned. Formal parameters are trivially mapped to their actual
  // parameters. Note that the first block in new_blocks retains the label
  // of the original calling block. Also note that if an exit block is
  // created, it is the last block of new_blocks.
  //
  // Also return in new_vars additional OpVariable instructions required by
  // and to be inserted into the caller function after the block at
  // call_block_itr is replaced with new_blocks.
  void GenInlineCode(std::vector<std::unique_ptr<ir::BasicBlock>>* new_blocks,
                     std::vector<std::unique_ptr<ir::Instruction>>* new_vars,
                     ir::UptrVectorIterator<ir::Instruction> call_inst_itr,
                     ir::UptrVectorIterator<ir::BasicBlock> call_block_itr);

  // Returns true if |inst| is a function call that can be inlined.
  bool IsInlinableFunctionCall(const ir::Instruction* inst);

  // Returns true if |func| is a function that can be inlined.
  bool IsInlinableFunction(const ir::Function* func);

  // Exhaustively inline all function calls in func as well as in
  // all code that is inlined into func. Return true if func is modified.
  bool Inline(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  ir::Module* module_;
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from function's result id to function.
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  // Map from block's label id to block.
  std::unordered_map<uint32_t, ir::BasicBlock*> id2block_;

  // Set of ids of inlinable function 
  std::set<uint32_t> inlinable_;

  // Next unused ID
  uint32_t next_id_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INLINE_PASS_H_
