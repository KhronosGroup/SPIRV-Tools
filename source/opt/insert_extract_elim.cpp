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

#include "insert_extract_elim.h"

#include "iterator.h"

static const int kSpvEntryPointFunctionId = 1;
static const int kSpvExtractCompositeId = 0;
static const int kSpvInsertObjectId = 0;
static const int kSpvInsertCompositeId = 1;

namespace spvtools {
namespace opt {

bool InsertExtractElimPass::ExtInsMatch(const ir::Instruction* extInst,
    const ir::Instruction* insInst) const {
  if (extInst->NumInOperands() != insInst->NumInOperands() - 1)
    return false;
  uint32_t numIdx = extInst->NumInOperands() - 1;
  for (uint32_t i = 0; i < numIdx; ++i)
    if (extInst->GetSingleWordInOperand(i + 1) !=
        insInst->GetSingleWordInOperand(i + 2))
      return false;
  return true;
}

bool InsertExtractElimPass::ExtInsConflict(const ir::Instruction* extInst,
    const ir::Instruction* insInst) const {
  if (extInst->NumInOperands() == insInst->NumInOperands() - 1)
    return false;
  uint32_t extNumIdx = extInst->NumInOperands() - 1;
  uint32_t insNumIdx = insInst->NumInOperands() - 2;
  uint32_t numIdx = extNumIdx < insNumIdx ? extNumIdx : insNumIdx;
  for (uint32_t i = 0; i < numIdx; ++i)
    if (extInst->GetSingleWordInOperand(i + 1) !=
        insInst->GetSingleWordInOperand(i + 2))
      return false;
  return true;
}

bool InsertExtractElimPass::EliminateInsertExtract(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpCompositeExtract: {
        uint32_t cid = ii->GetSingleWordInOperand(kSpvExtractCompositeId);
        ir::Instruction* cinst = def_use_mgr_->GetDef(cid);
        uint32_t replId = 0;
        while (cinst->opcode() == SpvOpCompositeInsert) {
          if (ExtInsConflict(&*ii, cinst))
            break;
          if (ExtInsMatch(&*ii, cinst)) {
            replId = cinst->GetSingleWordInOperand(kSpvInsertObjectId);
            break;
          }
          cid = cinst->GetSingleWordInOperand(kSpvInsertCompositeId);
          cinst = def_use_mgr_->GetDef(cid);
        }
        if (replId == 0)
          break;
        const uint32_t extId = ii->result_id();
        (void)def_use_mgr_->ReplaceAllUsesWith(extId, replId);
        def_use_mgr_->KillInst(&*ii);
        modified = true;
      } break;
      default:
        break;
      }
    }
  }
  return modified;
}

void InsertExtractElimPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  for (auto& fn : *module_)
    id2function_[fn.result_id()] = &fn;

  // Do def/use on whole module
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));
};

Pass::Status InsertExtractElimPass::ProcessImpl() {
  bool modified = false;

  // Process all entry point functions.
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = modified || EliminateInsertExtract(fn);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InsertExtractElimPass::InsertExtractElimPass()
    : module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status InsertExtractElimPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

