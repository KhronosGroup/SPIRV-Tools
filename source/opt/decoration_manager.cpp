// Copyright (c) 2017 Pierre Moreau
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

#include "decoration_manager.h"

#include <algorithm>
#include <set>
#include <stack>

namespace spvtools {
namespace opt {
namespace analysis {

void DecorationManager::RemoveDecorationsFrom(uint32_t id) {
  auto const ids_iter = id_to_decoration_insts_.find(id);
  if (ids_iter == id_to_decoration_insts_.end()) return;
  id_to_decoration_insts_.erase(ids_iter);
}

std::vector<ir::Instruction*> DecorationManager::GetDecorationsFor(
    uint32_t id, bool include_linkage) {
  return InternalGetDecorationsFor<ir::Instruction*>(id, include_linkage);
}

std::vector<const ir::Instruction*> DecorationManager::GetDecorationsFor(
    uint32_t id, bool include_linkage) const {
  return const_cast<DecorationManager*>(this)
      ->InternalGetDecorationsFor<const ir::Instruction*>(id, include_linkage);
}

bool DecorationManager::HaveTheSameDecorations(uint32_t id1,
                                               uint32_t id2) const {
  using InstructionList = std::vector<const ir::Instruction*>;
  using DecorationSet = std::set<std::u32string>;

  const InstructionList decorationsFor1 = GetDecorationsFor(id1, false);
  const InstructionList decorationsFor2 = GetDecorationsFor(id2, false);

  // This function splits the decoration instructions into different sets,
  // based on their opcode; only OpDecorate, OpDecorateId and OpMemberDecorate
  // are considered, the other opcodes are ignored.
  const auto fillDecorationSets =
      [](const InstructionList& decorationList, DecorationSet* decorateSet,
         DecorationSet* decorateIdSet, DecorationSet* memberDecorateSet) {
        for (const ir::Instruction* inst : decorationList) {
          std::u32string decorationPayload;
          // Ignore the opcode and the target as we do not want them to be
          // compared.
          for (uint32_t i = 1u; i < inst->NumInOperands(); ++i)
            for (uint32_t word : inst->GetInOperand(i).words)
              decorationPayload.push_back(word);

          switch (inst->opcode()) {
            case SpvOpDecorate:
              decorateSet->emplace(std::move(decorationPayload));
              break;
            case SpvOpMemberDecorate:
              memberDecorateSet->emplace(std::move(decorationPayload));
              break;
            case SpvOpDecorateId:
              decorateIdSet->emplace(std::move(decorationPayload));
              break;
            default:
              break;
          }
        }
      };

  DecorationSet decorateSetFor1;
  DecorationSet decorateIdSetFor1;
  DecorationSet memberDecorateSetFor1;
  fillDecorationSets(decorationsFor1, &decorateSetFor1, &decorateIdSetFor1,
                     &memberDecorateSetFor1);

  DecorationSet decorateSetFor2;
  DecorationSet decorateIdSetFor2;
  DecorationSet memberDecorateSetFor2;
  fillDecorationSets(decorationsFor2, &decorateSetFor2, &decorateIdSetFor2,
                     &memberDecorateSetFor2);

  return decorateSetFor1 == decorateSetFor2 &&
         decorateIdSetFor1 == decorateIdSetFor2 &&
         memberDecorateSetFor1 == memberDecorateSetFor2;
}

// TODO(pierremoreau): If OpDecorateId is referencing an OpConstant, one could
//                     check that the constants are the same rather than just
//                     looking at the constant ID.
bool DecorationManager::AreDecorationsTheSame(const ir::Instruction* inst1,
                                              const ir::Instruction* inst2,
                                              bool ignore_target) const {
  switch (inst1->opcode()) {
    case SpvOpDecorate:
    case SpvOpMemberDecorate:
    case SpvOpDecorateId:
      break;
    default:
      return false;
  }

  if (inst1->opcode() != inst2->opcode() ||
      inst1->NumInOperands() != inst2->NumInOperands())
    return false;

  for (uint32_t i = ignore_target ? 1u : 0u; i < inst1->NumInOperands(); ++i)
    if (inst1->GetInOperand(i) != inst2->GetInOperand(i)) return false;

  return true;
}

void DecorationManager::AnalyzeDecorations() {
  if (!module_) return;

  // Collect all group ids.
  for (const ir::Instruction& inst : module_->annotations()) {
    switch (inst.opcode()) {
      case SpvOpDecorationGroup:
        group_to_decoration_insts_.insert({inst.result_id(), {}});
        break;
      default:
        break;
    }
  }

  // For each group and instruction, collect all their decoration instructions.
  for (ir::Instruction& inst : module_->annotations()) {
    AddDecoration(&inst);
  }
}
void DecorationManager::AddDecoration(ir::Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpDecorate:
    case SpvOpDecorateId:
    case SpvOpMemberDecorate: {
      auto const target_id = inst->GetSingleWordInOperand(0u);
      auto const group_iter = group_to_decoration_insts_.find(target_id);
      if (group_iter != group_to_decoration_insts_.end())
        group_iter->second.push_back(inst);
      else
        id_to_decoration_insts_[target_id].push_back(inst);
      break;
    }
    case SpvOpGroupDecorate:
      for (uint32_t i = 1u; i < inst->NumInOperands(); ++i) {
        auto const target_id = inst->GetSingleWordInOperand(i);
        auto const group_iter = group_to_decoration_insts_.find(target_id);
        if (group_iter != group_to_decoration_insts_.end())
          group_iter->second.push_back(inst);
        else
          id_to_decoration_insts_[target_id].push_back(inst);
      }
      break;
    case SpvOpGroupMemberDecorate:
      for (uint32_t i = 1u; i < inst->NumInOperands(); i += 2u) {
        auto const target_id = inst->GetSingleWordInOperand(i);
        auto const group_iter = group_to_decoration_insts_.find(target_id);
        if (group_iter != group_to_decoration_insts_.end())
          group_iter->second.push_back(inst);
        else
          id_to_decoration_insts_[target_id].push_back(inst);
      }
      break;
    default:
      break;
  }
}

template <typename T>
std::vector<T> DecorationManager::InternalGetDecorationsFor(
    uint32_t id, bool include_linkage) {
  std::vector<T> decorations;
  std::stack<uint32_t> ids_to_process;

  const auto process = [&ids_to_process, &decorations](T inst) {
    if (inst->opcode() == SpvOpGroupDecorate ||
        inst->opcode() == SpvOpGroupMemberDecorate)
      ids_to_process.push(inst->GetSingleWordInOperand(0u));
    else
      decorations.push_back(inst);
  };

  const auto ids_iter = id_to_decoration_insts_.find(id);
  // |id| has no decorations
  if (ids_iter == id_to_decoration_insts_.end()) return decorations;

  // Process |id|'s decorations. Some of them might be groups, in which case
  // add them to the stack.
  for (ir::Instruction* inst : ids_iter->second) {
    const bool is_linkage =
        inst->opcode() == SpvOpDecorate &&
        inst->GetSingleWordInOperand(1u) == SpvDecorationLinkageAttributes;
    if (include_linkage || !is_linkage) process(inst);
  }

  // If the stack is not empty, then it contains groups ID: retrieve their
  // decorations and process them. If any of those decorations is applying a
  // group, push that group ID onto the stack.
  while (!ids_to_process.empty()) {
    const uint32_t id_to_process = ids_to_process.top();
    ids_to_process.pop();

    // Retrieve the decorations of that group
    const auto group_iter = group_to_decoration_insts_.find(id_to_process);
    if (group_iter != group_to_decoration_insts_.end()) {
      // Process all the decorations applied by the group.
      for (T inst : group_iter->second) process(inst);
    } else {
      // Something went wrong.
      assert(false);
      return std::vector<T>();
    }
  }

  return decorations;
}

bool DecorationManager::WhileEachDecoration(
    uint32_t id, uint32_t decoration,
    std::function<bool(const ir::Instruction&)> f) {
  for (const ir::Instruction* inst : GetDecorationsFor(id, true)) {
    switch (inst->opcode()) {
      case SpvOpMemberDecorate:
        if (inst->GetSingleWordInOperand(2) == decoration) {
          if (!f(*inst)) return false;
        }
        break;
      case SpvOpDecorate:
      case SpvOpDecorateId:
        if (inst->GetSingleWordInOperand(1) == decoration) {
          if (!f(*inst)) return false;
        }
        break;
      default:
        assert(false && "Unexpected decoration instruction");
    }
  }
  return true;
}

void DecorationManager::ForEachDecoration(
    uint32_t id, uint32_t decoration,
    std::function<void(const ir::Instruction&)> f) {
  WhileEachDecoration(id, decoration, [&f](const ir::Instruction& inst) {
    f(inst);
    return true;
  });
}

void DecorationManager::CloneDecorations(
    uint32_t from, uint32_t to, std::function<void(ir::Instruction&, bool)> f) {
  assert(f && "Missing function parameter f");
  auto const decoration_list = id_to_decoration_insts_.find(from);
  if (decoration_list == id_to_decoration_insts_.end()) return;
  for (ir::Instruction* inst : decoration_list->second) {
    switch (inst->opcode()) {
      case SpvOpGroupDecorate:
        f(*inst, false);
        // add |to| to list of decorated id's
        inst->AddOperand(
            ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID, {to}));
        id_to_decoration_insts_[to].push_back(inst);
        f(*inst, true);
        break;
      case SpvOpGroupMemberDecorate: {
        f(*inst, false);
        // for each (id == from), add (to, literal) as operands
        const uint32_t num_operands = inst->NumOperands();
        for (uint32_t i = 1; i < num_operands; i += 2) {
          ir::Operand op = inst->GetOperand(i);
          if (op.words[0] == from) {  // add new pair of operands: (to, literal)
            inst->AddOperand(
                ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID, {to}));
            op = inst->GetOperand(i + 1);
            inst->AddOperand(std::move(op));
          }
        }
        id_to_decoration_insts_[to].push_back(inst);
        f(*inst, true);
        break;
      }
      case SpvOpDecorate:
      case SpvOpMemberDecorate:
      case SpvOpDecorateId: {
        // simply clone decoration and change |target-id| to |to|
        std::unique_ptr<ir::Instruction> new_inst(
            inst->Clone(module_->context()));
        new_inst->SetInOperand(0, {to});
        id_to_decoration_insts_[to].push_back(new_inst.get());
        module_->AddAnnotationInst(std::move(new_inst));
        auto decoration_iter = --module_->annotation_end();
        f(*decoration_iter, true);
        break;
      }
      default:
        assert(false && "Unexpected decoration instruction");
    }
  }
}

void DecorationManager::RemoveDecoration(ir::Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpDecorate:
    case SpvOpDecorateId:
    case SpvOpMemberDecorate: {
      auto const target_id = inst->GetSingleWordInOperand(0u);
      RemoveInstructionFromTarget(inst, target_id);
    } break;
    case SpvOpGroupDecorate:
      for (uint32_t i = 1u; i < inst->NumInOperands(); ++i) {
        auto const target_id = inst->GetSingleWordInOperand(i);
        RemoveInstructionFromTarget(inst, target_id);
      }
      break;
    case SpvOpGroupMemberDecorate:
      for (uint32_t i = 1u; i < inst->NumInOperands(); i += 2u) {
        auto const target_id = inst->GetSingleWordInOperand(i);
        RemoveInstructionFromTarget(inst, target_id);
      }
      break;
    default:
      break;
  }
}

void DecorationManager::RemoveInstructionFromTarget(ir::Instruction* inst,
                                                    const uint32_t target_id) {
  auto const group_iter = group_to_decoration_insts_.find(target_id);
  if (group_iter != group_to_decoration_insts_.end()) {
    auto& insts = group_iter->second;
    insts.erase(std::remove(insts.begin(), insts.end(), inst), insts.end());
  } else {
    auto target_list_iter = id_to_decoration_insts_.find(target_id);
    if (target_list_iter != id_to_decoration_insts_.end()) {
      auto& insts = target_list_iter->second;
      insts.erase(std::remove(insts.begin(), insts.end(), inst), insts.end());
    }
  }
}
}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
