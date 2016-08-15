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
  Reset();
  module->ForEachInst(std::bind(&DefUseManager::AnalyzeInstDefUse, this,
                                std::placeholders::_1));
}

void DefUseManager::AnalyzeInstDefUse(ir::Instruction* inst) {
  // Clear the records of the given instruction if it has been analyzed before.
  ClearInst(inst);

  const uint32_t def_id = inst->result_id();
  if (def_id != 0) {
    // If the new instruction defines an existing result id, clear the records
    // of the existing id first.
    ClearDef(def_id);
    id_to_def_[def_id] = inst;
  }

  for (uint32_t i = 0; i < inst->NumOperands(); ++i) {
    if (IsIdUse(inst->GetOperand(i))) {
      uint32_t use_id = inst->GetSingleWordOperand(i);
      // use_id is used by this instruction.
      id_to_uses_[use_id].push_back({inst, i});
    }
  }

  analyzed_insts_.insert(inst);
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
  KillInst(id_to_def_[id]);
  return true;
}

void DefUseManager::KillInst(ir::Instruction* inst) {
  if (!inst) return;
  ClearInst(inst);
  inst->ToNop();
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

void DefUseManager::ClearDef(uint32_t def_id) {
  const auto& iter = id_to_def_.find(def_id);
  if (iter == id_to_def_.end()) return;
  EraseInstUsesOfOperands(*iter->second);
  // If a result id is defined by this instruction, remove the use records of
  // the id.
  id_to_uses_.erase(def_id);  // Remove all uses of this id.
  id_to_def_.erase(def_id);
}

void DefUseManager::ClearInst(ir::Instruction* inst) {
  // Do nothing if the instruction is a nullptr or it has not been analyzed
  // before.
  if (!inst || analyzed_insts_.count(inst) == 0) return;

  EraseInstUsesOfOperands(*inst);
  // If a result id is defined by this instruction, remove the use records of
  // the id.
  if (inst->result_id() != 0) {
    assert(id_to_def_[inst->result_id()] == inst);
    id_to_uses_.erase(inst->result_id());  // Remove all uses of this id.
    id_to_def_.erase(inst->result_id());
  }
}

bool DefUseManager::IsIdUse(const ir::Operand& operand) const {
  switch (operand.type) {
    // For any id type but result id type
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_TYPE_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
      return true;
    default:
      return false;
  }
}

void DefUseManager::EraseInstUsesOfOperands(const ir::Instruction& inst) {
  // Go through all ids, except the result id, used by this instruction, remove
  // this instruction's uses of those ids.
  for (uint32_t i = 0; i < inst.NumOperands(); i++) {
    if (IsIdUse(inst.GetOperand(i))) {
      uint32_t operand_id = inst.GetSingleWordOperand(i);
      auto iter = id_to_uses_.find(operand_id);
      if (iter != id_to_uses_.end()) {
        auto& uses = iter->second;
        for (auto it = uses.begin(); it != uses.end();) {
          if (it->inst == &inst) {
            it = uses.erase(it);
          } else {
            ++it;
          }
        }
        if (uses.empty()) id_to_uses_.erase(operand_id);
      }
    }
  }
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
