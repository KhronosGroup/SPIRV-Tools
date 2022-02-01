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

#include "source/opt/def_use_manager.h"

namespace spvtools {
namespace opt {
namespace analysis {

void DefUseManager::AnalyzeInstDef(Instruction* inst) {
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

void DefUseManager::AnalyzeInstUse(Instruction* inst) {
  // It might have existed before.
  EraseUseRecordsOfOperandIds(inst);

  // Create entry for the given instruction. Note that the instruction may
  // not have any in-operands. In such cases, we still need a entry for those
  // instructions so this manager knows it has seen the instruction later.
  UsedIdRange* instInfo = &inst_to_used_info_[inst];
  instInfo->first = uint32_t(used_ids_.size());

  for (uint32_t i = 0; i < inst->NumOperands(); ++i) {
    switch (inst->GetOperand(i).type) {
      // For any id type but result id type
      case SPV_OPERAND_TYPE_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
      case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      case SPV_OPERAND_TYPE_SCOPE_ID: {
        uint32_t use_id = inst->GetSingleWordOperand(i);
        Instruction* def = GetDef(use_id);
        assert(def && "Definition is not registered.");

        // Add to inst's use records
        used_ids_.push_back(use_id);

        // Add to the users, taking care to avoid adding duplicates.  We know
        // the duplicate for this instruction will always be at the tail.
        UseListHead& list_head = inst_to_users_.insert({def, UseListHead()}).first->second;
        if (use_pool_.empty(list_head) ||
            use_pool_[list_head.tail].element != inst) {
          use_pool_.push_back(list_head, inst);
        }
      } break;
      default:
        break;
    }
  }
  instInfo->second = uint32_t(used_ids_.size() - instInfo->first);
}

void DefUseManager::AnalyzeInstDefUse(Instruction* inst) {
  AnalyzeInstDef(inst);
  AnalyzeInstUse(inst);
  // Analyze lines last otherwise they will be cleared when inst is
  // cleared by preceding two calls
  for (auto& l_inst : inst->dbg_line_insts()) AnalyzeInstDefUse(&l_inst);
}

void DefUseManager::UpdateDefUse(Instruction* inst) {
  const uint32_t def_id = inst->result_id();
  if (def_id != 0) {
    auto iter = id_to_def_.find(def_id);
    if (iter == id_to_def_.end()) {
      AnalyzeInstDef(inst);
    }
  }
  AnalyzeInstUse(inst);
}

Instruction* DefUseManager::GetDef(uint32_t id) {
  auto iter = id_to_def_.find(id);
  if (iter == id_to_def_.end()) return nullptr;
  return iter->second;
}

const Instruction* DefUseManager::GetDef(uint32_t id) const {
  const auto iter = id_to_def_.find(id);
  if (iter == id_to_def_.end()) return nullptr;
  return iter->second;
}

bool DefUseManager::WhileEachUser(
    const Instruction* def, const std::function<bool(Instruction*)>& f) const {
  // Ensure that |def| has been registered.
  assert(def && (!def->HasResultId() || def == GetDef(def->result_id())) &&
         "Definition is not registered.");
  if (!def->HasResultId()) return true;

  auto iter = inst_to_users_.find(def);
  if (iter != inst_to_users_.end()) {
    for (int32_t index = iter->second.head; index != -1; /**/) {
      const auto& node = use_pool_[index];
      index = node.next;
      if (!f(node.element)) return false;
    }
  }
  return true;
}

bool DefUseManager::WhileEachUser(
    uint32_t id, const std::function<bool(Instruction*)>& f) const {
  return WhileEachUser(GetDef(id), f);
}

void DefUseManager::ForEachUser(
    const Instruction* def, const std::function<void(Instruction*)>& f) const {
  WhileEachUser(def, [&f](Instruction* user) {
    f(user);
    return true;
  });
}

void DefUseManager::ForEachUser(
    uint32_t id, const std::function<void(Instruction*)>& f) const {
  ForEachUser(GetDef(id), f);
}

bool DefUseManager::WhileEachUse(
    const Instruction* def,
    const std::function<bool(Instruction*, uint32_t)>& f) const {
  // Ensure that |def| has been registered.
  assert(def && (!def->HasResultId() || def == GetDef(def->result_id())) &&
         "Definition is not registered.");
  if (!def->HasResultId()) return true;

  auto iter = inst_to_users_.find(def);
  if (iter != inst_to_users_.end()) {
    for (int32_t list_idx = iter->second.head; list_idx != -1; /**/) {
      const auto& node = use_pool_[list_idx];
      Instruction* const user = node.element;
      list_idx = node.next;

      for (uint32_t idx = 0; idx != user->NumOperands(); ++idx) {
        const Operand& op = user->GetOperand(idx);
        if (op.type != SPV_OPERAND_TYPE_RESULT_ID && spvIsIdType(op.type)) {
          if (def->result_id() == op.words[0]) {
            if (!f(user, idx)) return false;
          }
        }
      }
    }
  }
  return true;
}

bool DefUseManager::WhileEachUse(
    uint32_t id, const std::function<bool(Instruction*, uint32_t)>& f) const {
  return WhileEachUse(GetDef(id), f);
}

void DefUseManager::ForEachUse(
    const Instruction* def,
    const std::function<void(Instruction*, uint32_t)>& f) const {
  WhileEachUse(def, [&f](Instruction* user, uint32_t index) {
    f(user, index);
    return true;
  });
}

void DefUseManager::ForEachUse(
    uint32_t id, const std::function<void(Instruction*, uint32_t)>& f) const {
  ForEachUse(GetDef(id), f);
}

uint32_t DefUseManager::NumUsers(const Instruction* def) const {
  uint32_t count = 0;
  ForEachUser(def, [&count](Instruction*) { ++count; });
  return count;
}

uint32_t DefUseManager::NumUsers(uint32_t id) const {
  return NumUsers(GetDef(id));
}

uint32_t DefUseManager::NumUses(const Instruction* def) const {
  uint32_t count = 0;
  ForEachUse(def, [&count](Instruction*, uint32_t) { ++count; });
  return count;
}

uint32_t DefUseManager::NumUses(uint32_t id) const {
  return NumUses(GetDef(id));
}

std::vector<Instruction*> DefUseManager::GetAnnotations(uint32_t id) const {
  std::vector<Instruction*> annos;
  const Instruction* def = GetDef(id);
  if (!def) return annos;

  ForEachUser(def, [&annos](Instruction* user) {
    if (IsAnnotationInst(user->opcode())) {
      annos.push_back(user);
    }
  });
  return annos;
}

void DefUseManager::AnalyzeDefUse(Module* module) {
  if (!module) return;
  // Analyze all the defs before any uses to catch forward references.
  module->ForEachInst(
      std::bind(&DefUseManager::AnalyzeInstDef, this, std::placeholders::_1),
      true);
  module->ForEachInst(
      std::bind(&DefUseManager::AnalyzeInstUse, this, std::placeholders::_1),
      true);
}

void DefUseManager::ClearInst(Instruction* inst) {
  if (inst_to_used_info_.find(inst) != inst_to_used_info_.end()) {
    EraseUseRecordsOfOperandIds(inst);
    if (inst->result_id() != 0) {
      inst_to_users_.erase(inst);
      id_to_def_.erase(inst->result_id());
    }
  }
}

void DefUseManager::EraseUseRecordsOfOperandIds(const Instruction* inst) {
  // Go through all ids used by this instruction, remove this instruction's
  // uses of them.
  auto iter = inst_to_used_info_.find(inst);
  if (iter != inst_to_used_info_.end()) {
    const UsedIdRange& range = iter->second;
    for (uint32_t idx = range.first, i = 0; i < range.second; ++i, ++idx) {
      auto def_iter = inst_to_users_.find(GetDef(used_ids_[idx]));
      if (def_iter != inst_to_users_.end()) {
        use_pool_.remove_first(def_iter->second,
                               const_cast<Instruction*>(inst));
      }
    }
    free_id_count_ += range.second;
    inst_to_used_info_.erase(inst);

    // Determine if we should compact use_records_ and used_ids_ based on how
    // much space has been freed so far compared to the amount actually in-use.
    // Don't bother doing any compaction until we're using a reasonable amount
    // of memory (64kb), regardless of how much is being wasted.
    // These thresholds are fungible: they exist to stop unbounded memory use.
    size_t in_use = used_ids_.size() - free_id_count_;
    size_t compact_min_used = (32 * 1024) / sizeof(used_ids_[0]);
    size_t compact_min_free = in_use * 16;
    if (in_use > compact_min_used && free_id_count_ > compact_min_free) {
      CompactStorage();
    }
  }
}

void DefUseManager::CompactStorage() {
  CompactUseRecords();
  CompactUsedIds();
}

void DefUseManager::CompactUseRecords() {
  UseList new_pool;
  for (auto& iter : inst_to_users_) {
    use_pool_.move_to(iter.second, new_pool);
  }
  use_pool_ = std::move(new_pool);
}

void DefUseManager::CompactUsedIds() {
  std::vector<uint32_t> new_ids;
  new_ids.reserve(used_ids_.size() - free_id_count_);
  for (auto& iter : inst_to_used_info_) {
    UsedIdRange& use_range = iter.second;
    new_ids.insert(new_ids.end(), &used_ids_[use_range.first],
                   &used_ids_[use_range.first] + use_range.second);
    use_range.first = int32_t(new_ids.size()) - use_range.second;
  }
  used_ids_ = std::move(new_ids);
  free_id_count_ = 0;
}

bool CompareAndPrintDifferences(const DefUseManager& lhs,
                                const DefUseManager& rhs) {
  bool same = true;

  if (lhs.id_to_def_ != rhs.id_to_def_) {
    for (auto p : lhs.id_to_def_) {
      if (rhs.id_to_def_.find(p.first) == rhs.id_to_def_.end()) {
        printf("Diff in id_to_def: missing value in rhs\n");
      }
    }
    for (auto p : rhs.id_to_def_) {
      if (lhs.id_to_def_.find(p.first) == lhs.id_to_def_.end()) {
        printf("Diff in id_to_def: missing value in lhs\n");
      }
    }
    same = false;
  }

  if (lhs.inst_to_used_info_.size() != rhs.inst_to_used_info_.size()) {
    printf("Diff in id_to_users: missing value in rhs\n");
    same = false;
  } else {
    for (auto p : lhs.inst_to_used_info_) {
      auto it_r = rhs.inst_to_used_info_.find(p.first);
      if (it_r == rhs.inst_to_used_info_.end()) {
        printf("Diff in id_to_used_info_: missing value in rhs\n");
        same = false;
        continue;
      }
      const auto range_l = p.second;
      const auto range_r = it_r->second;
      if (range_l.second != range_r.second) {
        printf("Diff in id_to_used_info_: different number of used in rhs\n");
        continue;
      }
      for (uint32_t i = 0; i < range_l.second; ++i) {
        if (lhs.used_ids_[range_l.first + i] !=
            rhs.used_ids_[range_r.first + i]) {
          printf("Diff in id_to_used_info_: different used in rhs\n");
          same = false;
        }
      }
    }
  }

  for (auto l : lhs.inst_to_users_) {
    std::set<Instruction*> ul, ur;
    lhs.ForEachUser(l.first, [&ul](Instruction* use) { ul.insert(use); });
    rhs.ForEachUser(l.first, [&ur](Instruction* use) { ur.insert(use); });
    if (ul != ur) {
      printf("Diff in inst_to_users_: different users\n");
      same = false;
    }
  }
  for (auto r : rhs.inst_to_users_) {
    auto iter_l = lhs.inst_to_users_.find(r.first);
    if (r.second.head == -1 &&
        !(iter_l == lhs.inst_to_users_.end() || iter_l->second.head == -1)) {
      printf("Diff in inst_to_users_: unexpected instr in rhs\n");
      same = false;
    }
  }
  return same;
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
