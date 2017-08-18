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

#include "inline_opaque_pass.h"

namespace spvtools {
namespace opt {

namespace {

  const uint32_t kTypePointerTypeIdInIdx = 1;

} // anonymous namespace

bool InlineOpaquePass::IsOpaqueType(uint32_t typeId) {
  const ir::Instruction* typeInst = def_use_mgr_->GetDef(typeId);
  switch (typeInst->opcode()) {
    case SpvOpTypeSampler:
    case SpvOpTypeImage:
    case SpvOpTypeSampledImage:
      return true;
    case SpvOpTypePointer:
      return IsOpaqueType(typeInst->GetSingleWordInOperand(
          kTypePointerTypeIdInIdx));
    default:
      break;
  }
  // TODO(greg-lunarg): Handle arrays containing opaque type
  if (typeInst->opcode() != SpvOpTypeStruct)
    return false;
  // Return true if any member is opaque
  int ocnt = 0;
  typeInst->ForEachInId([&ocnt,this](const uint32_t* tid) {
    if (ocnt == 0 && IsOpaqueType(*tid)) ++ocnt;
  });
  return ocnt > 0;
}

bool InlineOpaquePass::HasOpaqueArgsOrReturn(const ir::Instruction* callInst) {
  // Check return type
  if (IsOpaqueType(callInst->type_id()))
    return true;
  // Check args
  int icnt = 0;
  int ocnt = 0;
  callInst->ForEachInId([&icnt,&ocnt,this](const uint32_t *iid) {
    if (icnt > 0) {
      const ir::Instruction* argInst = def_use_mgr_->GetDef(*iid);
      if (IsOpaqueType(argInst->type_id()))
        ++ocnt;
    }
    ++icnt;
  });
  return ocnt > 0;
}

bool InlineOpaquePass::InlineOpaque(ir::Function* func) {
  bool modified = false;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (IsInlinableFunctionCall(&*ii) && HasOpaqueArgsOrReturn(&*ii)) {
        // Inline call.
        std::vector<std::unique_ptr<ir::BasicBlock>> newBlocks;
        std::vector<std::unique_ptr<ir::Instruction>> newVars;
        GenInlineCode(&newBlocks, &newVars, ii, bi);
        // If call block is replaced with more than one block, point
        // succeeding phis at new last block.
        if (newBlocks.size() > 1)
          UpdateSucceedingPhis(newBlocks);
        // Replace old calling block with new block(s).
        bi = bi.Erase();
        bi = bi.InsertBefore(&newBlocks);
        // Insert new function variables.
        if (newVars.size() > 0) func->begin()->begin().InsertBefore(&newVars);
        // Restart inlining at beginning of calling block.
        ii = bi->begin();
        modified = true;
      } else {
        ++ii;
      }
    }
  }
  return modified;
}

void InlineOpaquePass::Initialize(ir::Module* module) {
  InitializeInline(module);
};

Pass::Status InlineOpaquePass::ProcessImpl() {
  // Do opaque inlining on each function in entry point call tree
  ProcessFunction pfn = [this](ir::Function* fp) {
    return InlineOpaque(fp);
  };
  bool modified = ProcessEntryPointCallTree(pfn, module_);
  FinalizeNextId(module_);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InlineOpaquePass::InlineOpaquePass() {}

Pass::Status InlineOpaquePass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
