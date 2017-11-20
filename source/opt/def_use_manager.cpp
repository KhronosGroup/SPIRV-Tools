// Copyright (c) 2016 Google Inc.
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

#include "def_use_manager.h"

#include "log.h"
#include "reflect.h"

namespace spvtools {
namespace opt {
namespace analysis {

void DefUseManager::AnalyzeInstDef(ir::Instruction* inst) {
  const uint32_t def_id = inst->result_id();
  if (def_id != 0) {
    auto iter = id_to_def_.find(def_id);
    if (iter != id_to_def_.end()) {
      // Clear the original instruction that defining the same result id of the
      // new instruction.
      ClearInst(iter->second);
    }
    id_to_def_[def_id] = inst;
  } else {
    ClearInst(inst);
  }
}

void DefUseManager::AnalyzeInstUse(ir::Instruction* inst) {
  // Create entry for the given instruction. Note that the instruction may
  // not have any in-operands. In such cases, we still need a entry for those
  // instructions so this manager knows it has seen the instruction later.
  inst_to_used_ids_[inst] = {};

  for (uint32_t i = 0; i < inst->NumOperands(); ++i) {
    switch (inst->GetOperand(i).type) {
      // For any id type but result id type
      case SPV_OPERAND_TYPE_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
      case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      case SPV_OPERAND_TYPE_SCOPE_ID: {
        uint32_t use_id = inst->GetSingleWordOperand(i);
        id_to_users_.insert(UserEntry(GetDef(use_id), inst));
        inst_to_used_ids_[inst].push_back(use_id);
      } break;
      default:
        break;
    }
  }
}

void DefUseManager::AnalyzeInstDefUse(ir::Instruction* inst) {
  AnalyzeInstDef(inst);
  AnalyzeInstUse(inst);
}

ir::Instruction* DefUseManager::GetDef(uint32_t id) {
  auto iter = id_to_def_.find(id);
  if (iter == id_to_def_.end()) return nullptr;
  return iter->second;
}

const ir::Instruction* DefUseManager::GetDef(uint32_t id) const {
  const auto iter = id_to_def_.find(id);
  if (iter == id_to_def_.end()) return nullptr;
  return iter->second;
}

<<<<<<< HEAD
=======
//UseList* DefUseManager::GetUses(uint32_t id) {
//  auto iter = id_to_uses_.find(id);
//  if (iter == id_to_uses_.end()) return nullptr;
//  return &iter->second;
//}
//
//const UseList* DefUseManager::GetUses(uint32_t id) const {
//  const auto iter = id_to_uses_.find(id);
//  if (iter == id_to_uses_.end()) return nullptr;
//  return &iter->second;
//}

>>>>>>> Replaced representation of uses
void DefUseManager::ForEachUser(const ir::Instruction* def,
                                const std::function<void(ir::Instruction*)>& f) const {
  // Ensure that |def| has been registered.
  assert(def && def == GetDef(def->result_id()) && "Definition is not registered.");
  auto iter = id_to_users_.lower_bound(UserEntry(const_cast<ir::Instruction*>(def), nullptr));
  while (iter != id_to_users_.end() && iter->first == def) {
    f(iter->second);
    ++iter;
  }
}

void DefUseManager::ForEachUser(uint32_t id,
                                const std::function<void(ir::Instruction*)>& f) const {
  ForEachUser(GetDef(id), f);
<<<<<<< HEAD
}

void DefUseManager::ForEachUse(const ir::Instruction* def,
                               const std::function<void(ir::Instruction*, uint32_t)>& f) const {
  // Ensure that |def| has been registered.
  assert(def && def == GetDef(def->result_id()) && "Definition is not registered.");
  auto iter = id_to_users_.lower_bound(UserEntry(const_cast<ir::Instruction*>(def), nullptr));
  while (iter != id_to_users_.end() && iter->first == def) {
    ir::Instruction* user = iter->second;
    for (uint32_t idx = 0; idx != user->NumOperands(); ++idx) {
      const ir::Operand& op = user->GetOperand(idx);
      switch (op.type) {
        case SPV_OPERAND_TYPE_ID:
        case SPV_OPERAND_TYPE_TYPE_ID:
        case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
        case SPV_OPERAND_TYPE_SCOPE_ID: {
          if (def->result_id() == op.words[0])
            f(user, idx);
          break;
        }
        default:
          break;
      }
    }
    ++iter;
  }
}

=======
}

void DefUseManager::ForEachUse(const ir::Instruction* def,
                               const std::function<void(ir::Instruction*, uint32_t)>& f) const {
  // Ensure that |def| has been registered.
  assert(def && def == GetDef(def->result_id()) && "Definition is not registered.");
  auto iter = id_to_users_.lower_bound(UserEntry(const_cast<ir::Instruction*>(def), nullptr));
  while (iter != id_to_users_.end() && iter->first == def) {
    ir::Instruction* user = iter->second;
    for (uint32_t idx = 0; idx != user->NumOperands(); ++idx) {
      const ir::Operand& op = user->GetOperand(idx);
      switch (op.type) {
        case SPV_OPERAND_TYPE_ID:
        case SPV_OPERAND_TYPE_TYPE_ID:
        case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
        case SPV_OPERAND_TYPE_SCOPE_ID: {
          if (def->result_id() == op.words[0])
            f(user, idx);
          break;
        }
        default:
          break;
      }
    }
    //uint32_t numInOps = user->NumInOperands();
    //for (uint32_t idx = user->NumOperands() - numInOps; idx != numInOps; ++idx) {
    //  const ir::Operand &op = user->GetInOperand(idx);
    //  if (op.type == SPV_OPERAND_TYPE_ID && op.words[0] == def->result_id()) {
    //    f(user, idx);
    //  }
    //}
    ++iter;
  }
}

>>>>>>> Replaced representation of uses
void DefUseManager::ForEachUse(uint32_t id,
                               const std::function<void(ir::Instruction*, uint32_t)>& f) const {
  ForEachUse(GetDef(id), f);
}

std::vector<ir::Instruction*> DefUseManager::GetAnnotations(uint32_t id) const {
  std::vector<ir::Instruction*> annos;
<<<<<<< HEAD
  if (!GetDef(id)) return annos;

=======
>>>>>>> Replaced representation of uses
  ForEachUser(id, [&annos](ir::Instruction* user) {
    if (ir::IsAnnotationInst(user->opcode())) {
      annos.push_back(user);
    }
  });
<<<<<<< HEAD
=======
  //const auto* uses = GetUses(id);
  //if (!uses) return annos;
  //for (const auto& c : *uses) {
  //  if (ir::IsAnnotationInst(c.inst->opcode())) {
  //    annos.push_back(c.inst);
  //  }
  //}
>>>>>>> Replaced representation of uses
  return annos;
}

void DefUseManager::AnalyzeDefUse(ir::Module* module) {
  if (!module) return;
<<<<<<< HEAD
  // Analyze all the defs before any uses to catch forward references.
=======
  //module->ForEachInst(std::bind(&DefUseManager::AnalyzeInstDefUse, this,
  //                              std::placeholders::_1));
>>>>>>> Replaced representation of uses
  module->ForEachInst(std::bind(&DefUseManager::AnalyzeInstDef, this,
                                std::placeholders::_1));
  module->ForEachInst(std::bind(&DefUseManager::AnalyzeInstUse, this,
                                std::placeholders::_1));
}

void DefUseManager::ClearInst(ir::Instruction* inst) {
  auto iter = inst_to_used_ids_.find(inst);
  if (iter != inst_to_used_ids_.end()) {
    EraseUseRecordsOfOperandIds(inst);
    if (inst->result_id() != 0) {
<<<<<<< HEAD
=======
      //id_to_uses_.erase(inst->result_id());  // Remove all uses of this id.
>>>>>>> Replaced representation of uses
      // Remove all uses of this inst.
      auto user_end = id_to_users_.end();
      auto user_iter = id_to_users_.lower_bound(UserEntry(inst, nullptr));
      while (user_iter != user_end && user_iter->first == inst) {
        // Increment to next element before invalidating iterator.
        auto use = user_iter++;
        id_to_users_.erase(use);
      }
      id_to_def_.erase(inst->result_id());
    }
  }
}

void DefUseManager::EraseUseRecordsOfOperandIds(const ir::Instruction* inst) {
  // Go through all ids used by this instruction, remove this instruction's
  // uses of them.
  auto iter = inst_to_used_ids_.find(inst);
  if (iter != inst_to_used_ids_.end()) {
<<<<<<< HEAD
    for (auto use_id : iter->second) {
      id_to_users_.erase(UserEntry(GetDef(use_id), const_cast<ir::Instruction*>(inst)));
=======
    // Cache the end iterator on the map.  The end iterator on
    // an unordered map does not get invalidated when erasing an
    // element.
    //const auto& id_to_uses_end = id_to_uses_.end();
    //const auto& id_to_users_end = id_to_users_.end();
    for (auto use_id : iter->second) {
      id_to_users_.erase(UserEntry(GetDef(use_id), const_cast<ir::Instruction*>(inst)));
      //auto uses_iter = id_to_uses_.find(use_id);
      //if (uses_iter == id_to_uses_end) continue;
      //auto& uses = uses_iter->second;
      //// Similarly, cache this end iterator.  It is not invalidated
      //// by erasure of an element from the list.
      //const auto& uses_end = uses.end();
      //for (auto it = uses.begin(); it != uses_end;) {
      //  if (it->inst == inst) {
      //    it = uses.erase(it);
      //  } else {
      //    ++it;
      //  }
      //}
      //if (uses.empty()) id_to_uses_.erase(use_id);
>>>>>>> Replaced representation of uses
    }
    inst_to_used_ids_.erase(inst);
  }
}

bool operator==(const DefUseManager& lhs, const DefUseManager& rhs) {
  if (lhs.id_to_def_ != rhs.id_to_def_) {
    return false;
  }

  if (lhs.id_to_users_ != rhs.id_to_users_) {
    return false;
  }
  //for (auto use : lhs.id_to_uses_) {
  //  auto rhs_iter = rhs.id_to_uses_.find(use.first);
  //  if (rhs_iter == rhs.id_to_uses_.end()) {
  //    return false;
  //  }
  //  use.second.sort();
  //  UseList rhs_uselist = rhs_iter->second;
  //  rhs_uselist.sort();
  //  if (use.second != rhs_uselist) {
  //    return false;
  //  }
  //}

  if (lhs.inst_to_used_ids_ != lhs.inst_to_used_ids_) {
    return false;
  }
  return true;
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
