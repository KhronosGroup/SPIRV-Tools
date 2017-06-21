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

namespace spvtools {
namespace opt {

namespace {

const int kEntryPointFunctionIdInIdx = 1;

} // anonymous namespace

bool BlockMergePass::IsLoopHeader(ir::BasicBlock* block_ptr) {
  auto iItr = block_ptr->tail();
  if (iItr == block_ptr->begin())
    return false;
  --iItr;
  return iItr->opcode() == SpvOpLoopMerge;
}

bool BlockMergePass::HasMultipleRefs(uint32_t labId) {
  const analysis::UseList* uses = def_use_mgr_->GetUses(labId);
  int rcnt = 0;
  for (const auto u : *uses) {
    // Don't count OpName
    if (u.inst->opcode() == SpvOpName)
      continue;
    if (rcnt == 1)
      return true;
    ++rcnt;
  }
  return false;
}

void BlockMergePass::KillInstAndName(ir::Instruction* inst) {
  const uint32_t id = inst->result_id();
  if (id != 0) {
    analysis::UseList* uses = def_use_mgr_->GetUses(id);
    if (uses != nullptr)
      for (auto u : *uses)
        if (u.inst->opcode() == SpvOpName) {
          def_use_mgr_->KillInst(u.inst);
          break;
        }
  }
  def_use_mgr_->KillInst(inst);
}

bool BlockMergePass::MergeBlocks(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ) {
    // Do not merge loop header blocks, at least for now.
    if (IsLoopHeader(&*bi)) {
      ++bi;
      continue;
    }
    // Find block with single successor which has no other predecessors.
    // Continue and Merge blocks are currently ruled out as second blocks.
    // Happily any such candidate blocks will have >1 uses due to their
    // LoopMerge instruction.
    // TODO(): Deal with phi instructions that reference the
    // second block. Happily, these references currently inhibit
    // the merge.
    auto ii = bi->end();
    --ii;
    ir::Instruction* br = &*ii;
    if (br->opcode() != SpvOpBranch) {
      ++bi;
      continue;
    }
    const uint32_t labId = br->GetSingleWordInOperand(0);
    if (HasMultipleRefs(labId)) {
      ++bi;
      continue;
    }
    // Merge blocks
    def_use_mgr_->KillInst(br);
    auto sbi = bi;
    for (; sbi != func->end(); ++sbi)
      if (sbi->id() == labId)
        break;
    // If bi is sbi's only predecessor, it dominates sbi and thus
    // sbi must follow bi in func's ordering.
    assert(sbi != func->end());
    bi->AddInstructions(&*sbi);
    KillInstAndName(sbi->GetLabelInst());
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
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx)];
    modified = MergeBlocks(fn) || modified;
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

