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

#include "block_merge_pass.h"

#include "iterator.h"

static const int kSpvEntryPointFunctionId = 1;

namespace spvtools {
namespace opt {

bool BlockMergePass::IsLoopHeader(ir::BasicBlock* block_ptr) {
  auto iItr = block_ptr->end();
  --iItr;
  if (iItr == block_ptr->begin())
    return false;
  --iItr;
  return iItr->opcode() == SpvOpLoopMerge;
}

bool BlockMergePass::MergeBlocks(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ) {
    // Do not merge loop header blocks, at least for now.
    if (IsLoopHeader(&*bi)) {
      ++bi;
      continue;
    }
    // Find block with single successor which has
    // no other predecessors
    auto ii = bi->end();
    --ii;
    ir::Instruction* br = &*ii;
    if (br->opcode() != SpvOpBranch) {
      ++bi;
      continue;
    }
    uint32_t labId = br->GetSingleWordInOperand(0);
    analysis::UseList* uses = def_use_mgr_->GetUses(labId);
    if (uses->size() > 1) {
      ++bi;
      continue;
    }
    // Merge blocks
    def_use_mgr_->KillInst(br);
    auto sbi = bi;
    for (; sbi != func->end(); ++sbi)
      if (sbi->id() == labId)
        break;
    assert(sbi != func->end());
    bi->AddInstructions(&*sbi);
    def_use_mgr_->KillInst(sbi->GetLabelInst());
    (void) sbi.Erase();
    // reprocess block
    modified = true;
  }
  return modified;
}

void BlockMergePass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  for (auto& fn : *module_) 
    id2function_[fn.result_id()] = &fn;

  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));
};

Pass::Status BlockMergePass::ProcessImpl() {
  bool modified = false;

  // Call Mem2Reg on all remaining functions.
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = modified || MergeBlocks(fn);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

BlockMergePass::BlockMergePass()
    : module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status BlockMergePass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

