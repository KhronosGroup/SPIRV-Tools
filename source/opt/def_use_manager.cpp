// Copyright (c) 2016 Google Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#include "def_use_manager.h"

#include <cassert>
#include <functional>

#include "instruction.h"
#include "module.h"

namespace spvtools {
namespace opt {
namespace analysis {

void DefUseManager::AnalyzeDefUse(ir::Module* module) {
  module->ForEachInst(std::bind(&DefUseManager::AnalyzeInstDefUse, this,
                                std::placeholders::_1));
}

ir::Instruction* DefUseManager::GetDef(uint32_t id) {
  if (id_to_def_.count(id) == 0) return nullptr;
  return id_to_def_.at(id);
}

UseList* DefUseManager::GetUses(uint32_t id) {
  if (id_to_uses_.count(id) == 0) return nullptr;
  return &id_to_uses_.at(id);
}

bool DefUseManager::KillDef(uint32_t id) {
  if (id_to_def_.count(id) == 0) return false;

  // Go through all ids usd by this instruction, remove this instruction's uses
  // of them.
  for (const auto use_id : result_id_to_used_ids_[id]) {
    if (id_to_uses_.count(use_id) == 0) continue;
    auto& uses = id_to_uses_[use_id];
    for (auto it = uses.begin(); it != uses.end();) {
      if (it->inst->result_id() == id) {
        it = uses.erase(it);
      } else {
        ++it;
      }
    }
    if (uses.empty()) id_to_uses_.erase(use_id);
  }
  result_id_to_used_ids_.erase(id);

  id_to_uses_.erase(id);  // Remove all uses of this id.
  // This must happen at the last since we use information inside the instuction
  // in the above.
  id_to_def_[id]->ToNop();
  id_to_def_.erase(id);
  return true;
}

bool DefUseManager::ReplaceAllUsesWith(uint32_t before, uint32_t after) {
  if (before == after) return false;
  if (id_to_uses_.count(before) == 0) return false;

  for (auto it = id_to_uses_[before].cbegin(); it != id_to_uses_[before].cend();
       ++it) {
    // Make the modification in the instruction.
    it->inst->SetOperand(it->operand_index, {after});
    // Register the use of |after| id into id_to_uses_.
    // TODO(antiagainst): de-duplication.
    id_to_uses_[after].push_back({it->inst, it->operand_index});
  }
  id_to_uses_.erase(before);
  return true;
}

void DefUseManager::AnalyzeInstDefUse(ir::Instruction* inst) {
  const uint32_t def_id = inst->result_id();
  if (def_id != 0) id_to_def_[def_id] = inst;

  for (uint32_t i = 0; i < inst->NumOperands(); ++i) {
    switch (inst->GetOperand(i).type) {
      // For any id type but result id type
      case SPV_OPERAND_TYPE_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
      case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      case SPV_OPERAND_TYPE_SCOPE_ID: {
        uint32_t use_id = inst->GetSingleWordOperand(i);
        // use_id is used by the instruction generating def_id.
        id_to_uses_[use_id].push_back({inst, i});
        if (def_id != 0) result_id_to_used_ids_[def_id].push_back(use_id);
      } break;
      default:
        break;
    }
  }
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
