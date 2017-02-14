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
  // Write the next available Id back to the module
  inline void FinalizeNextId(ir::Module* module) {
    module->SetIdBound(next_id_);
  }

  // Return the next available Id and increment it
  inline uint32_t TakeNextId() { return next_id_++; }

  // Find pointer to type and storage in module, return its resultId.
  // 0 if not found.
  uint32_t FindPointerToType(uint32_t type_id, uint32_t storage_id);

  // Add pointer to type to module and return resultId.
  uint32_t AddPointerToType(uint32_t type_id, uint32_t storage_id);

  // Add unconditional branch to labelId to end of block block_ptr
  void AddBranch(uint32_t labelId, std::unique_ptr<ir::BasicBlock>* block_ptr);

  // Add store of valId to ptrId to end of block block_ptr
  void AddStore(uint32_t ptrId, uint32_t valId,
                std::unique_ptr<ir::BasicBlock>* block_ptr);

  // Add load of ptrId into resultId to end of block block_ptr
  void AddLoad(uint32_t typeId, uint32_t resultId, uint32_t ptrId,
               std::unique_ptr<ir::BasicBlock>* block_ptr);

  // Return in new_blocks the result of inlining the call at call_ii within
  // its block call_bi. The block call_bi can just be replaced with the blocks
  // in new_blocks. Any additional branches are avoided. Debug instructions
  // are cloned along with their callee instructions. Early returns are
  // replaced by storing to a local return variable and branching to a 
  // (created) exit block where the local variable is returned. Formal
  // parameters are trivially mapped to their actual parameters.
  //
  // Also return in new_vars additional OpVariable instructions required by 
  // and to be inserted into the caller function after the block call_bi is 
  // replaced with new_blocks.
  void GenInlineCode(std::vector<std::unique_ptr<ir::BasicBlock>>* new_blocks,
                     std::vector<std::unique_ptr<ir::Instruction>>* new_vars,
                     ir::UptrVectorIterator<ir::Instruction> call_inst_itr,
                     ir::UptrVectorIterator<ir::BasicBlock> call_block_itr);

  // Exhaustively inline all function calls in func as well as in
  // all code that is inlined into func. Return true if func is modified.
  bool Inline(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  ir::Module* module_;
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  // Map from block's label id to block
  std::unordered_map<uint32_t, ir::BasicBlock*> id2block_;

  // Next unused ID
  uint32_t next_id_;

};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INLINE_PASS_H_
