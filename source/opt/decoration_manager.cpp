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
#include <iostream>
#include <stack>

namespace {

using spvtools::ir::Instruction;
using spvtools::ir::Operand;

// Returns true if |inst1| is strictly less than |inst2|, for SpvOpDecorate and
// SpvOpDecorateId.
//
// |inst1| is considered to be strictly less than |inst2| if:
// * the decoration of |inst1| is strictly less than the one from |inst2|;
// * and |inst1| has strictly fewer operands than |inst2|;
// * and when comparing operands
//   * |inst1|'s operand has strictly fewer words than |inst2|'s one;
//   * and each word of |inst1|'s operand is strictly less than the ones from
//     |inst2| (using their representation as uint32_t).
bool compareBeforeDecorate(const Instruction* inst1, const Instruction* inst2) {
  if (inst1->GetSingleWordInOperand(1u) > inst2->GetSingleWordInOperand(1u))
    return false;
  if (inst1->NumInOperands() > inst2->NumInOperands()) return false;
  for (uint32_t i = 2u; i < inst1->NumInOperands(); ++i) {
    const Operand& op1 = inst1->GetInOperand(i);
    const Operand& op2 = inst2->GetInOperand(i);
    if (op1.words.size() > op2.words.size()) return false;
    for (uint32_t j = 0u; j < op1.words.size(); ++j)
      if (op1.words[j] > op2.words[j]) return false;
  }
  return true;
}

// Returns true if |inst1| is strictly less than |inst2|, for
// SpvOpMemberDecorate.
//
// |inst1| is considered to be strictly less than |inst2| if:
// * the decoration of |inst1| is strictly less than the one from |inst2|;
// * the member index of |inst1| is strictly less than the one from |inst2|;
// * and |inst1| has strictly fewer operands than |inst2|;
// * and when comparing operands
//   * |inst1|'s operand has strictly fewer words than |inst2|'s one;
//   * and each word of |inst1|'s operand is strictly less than the ones from
//     |inst2| (using their representation as uint32_t).
bool compareBeforeMemberDecorate(const Instruction* inst1,
                                 const Instruction* inst2) {
  if (inst1->GetSingleWordInOperand(2u) > inst2->GetSingleWordInOperand(2u))
    return false;
  if (inst1->GetSingleWordInOperand(1u) > inst2->GetSingleWordInOperand(1u))
    return false;
  if (inst1->NumInOperands() > inst2->NumInOperands()) return false;
  for (uint32_t i = 3u; i < inst1->NumInOperands(); ++i) {
    const Operand& op1 = inst1->GetInOperand(i);
    const Operand& op2 = inst2->GetInOperand(i);
    if (op1.words.size() > op2.words.size()) return false;
    for (uint32_t j = 0u; j < op1.words.size(); ++j)
      if (op1.words[j] > op2.words[j]) return false;
  }
  return true;
}

// Returns true if |inst1| is equal to |inst2|, for SpvOpDecorate and
// SpvOpDecorateId.
//
// When checking for equality, the target is not considered. The actual check
// is the same as done by compareBeforeDecorate, but replacing all instances of
// "strictly less than" by "equal to".
bool compareEqualDecorate(const Instruction* inst1, const Instruction* inst2) {
  if (inst1->NumInOperands() != inst2->NumInOperands()) return false;
  for (uint32_t i = 2u; i < inst1->NumInOperands(); ++i)
    if (inst1->GetInOperand(i) != inst2->GetInOperand(i)) return false;
  return inst1->GetSingleWordInOperand(1u) == inst2->GetSingleWordInOperand(1u);
}

// Returns true if |inst1| is equal to |inst2|, for SpvOpMemberDecorate.
//
// When checking for equality, the target is not considered. The actual check
// is the same as done by compareBeforeMemberDecorate, but replacing all
// instances of "strictly less than" by "equal to".
bool compareEqualMemberDecorate(const Instruction* inst1,
                                const Instruction* inst2) {
  if (inst1->NumInOperands() != inst2->NumInOperands()) return false;
  if (inst1->GetSingleWordInOperand(1u) != inst2->GetSingleWordInOperand(1u))
    return false;
  for (uint32_t i = 3u; i < inst1->NumInOperands(); ++i)
    if (inst1->GetInOperand(i) != inst2->GetInOperand(i)) return false;
  return inst1->GetSingleWordInOperand(2u) == inst2->GetSingleWordInOperand(2u);
}

}  // end of namespace

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

  const InstructionList decorationsFor1 = GetDecorationsFor(id1, false);
  const InstructionList decorationsFor2 = GetDecorationsFor(id2, false);

  // This function will sort and remove all duplicates for each of the three
  // given lists.
  const auto generateUniqueLists = [](const InstructionList& decorationList,
                                      InstructionList* decorateList,
                                      InstructionList* decorateIdList,
                                      InstructionList* memberDecorateList) {
    for (const ir::Instruction* inst : decorationList) {
      switch (inst->opcode()) {
        case SpvOpDecorate:
          decorateList->push_back(inst);
          break;
        case SpvOpMemberDecorate:
          memberDecorateList->push_back(inst);
          break;
        case SpvOpDecorateId:
          decorateIdList->push_back(inst);
          break;
        default:
          break;
      }
    }
    std::sort(decorateList->begin(), decorateList->end(),
              compareBeforeDecorate);
    decorateList->erase(std::unique(decorateList->begin(), decorateList->end(),
                                    compareEqualDecorate),
                        decorateList->end());
    std::sort(decorateIdList->begin(), decorateIdList->end(),
              compareBeforeDecorate);
    decorateIdList->erase(
        std::unique(decorateIdList->begin(), decorateIdList->end(),
                    compareEqualDecorate),
        decorateIdList->end());
    std::sort(memberDecorateList->begin(), memberDecorateList->end(),
              compareBeforeMemberDecorate);
    memberDecorateList->erase(
        std::unique(memberDecorateList->begin(), memberDecorateList->end(),
                    compareEqualMemberDecorate),
        memberDecorateList->end());
  };

  InstructionList decorateInstructionsFor1;
  InstructionList decorateIdInstructionsFor1;
  InstructionList decorateMemberInstructionsFor1;
  generateUniqueLists(decorationsFor1, &decorateInstructionsFor1,
                      &decorateIdInstructionsFor1,
                      &decorateMemberInstructionsFor1);

  InstructionList decorateInstructionsFor2;
  InstructionList decorateIdInstructionsFor2;
  InstructionList decorateMemberInstructionsFor2;
  generateUniqueLists(decorationsFor2, &decorateInstructionsFor2,
                      &decorateIdInstructionsFor2,
                      &decorateMemberInstructionsFor2);

  // This function checks that both lists contain the exact same decorations.
  const auto compareLists = [this](const InstructionList& list1,
                                   const InstructionList& list2) {
    if (list1.size() != list2.size()) return false;
    for (uint32_t i = 0u; i < list1.size(); ++i)
      if (!AreDecorationsTheSame(list1[i], list2[i], true)) return false;
    return true;
  };

  if (!compareLists(decorateInstructionsFor1, decorateInstructionsFor2))
    return false;
  if (!compareLists(decorateIdInstructionsFor1, decorateIdInstructionsFor2))
    return false;
  if (!compareLists(decorateMemberInstructionsFor1,
                    decorateMemberInstructionsFor2))
    return false;

  return true;
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

void DecorationManager::ForEachDecoration(
    uint32_t id, uint32_t decoration,
    std::function<void(const ir::Instruction&)> f) {
  for (const ir::Instruction* inst : GetDecorationsFor(id, true)) {
    switch (inst->opcode()) {
      case SpvOpMemberDecorate:
        if (inst->GetSingleWordInOperand(2) == decoration) {
          f(*inst);
        }
        break;
      case SpvOpDecorate:
      case SpvOpDecorateId:
        if (inst->GetSingleWordInOperand(1) == decoration) {
          f(*inst);
        }
        break;
      default:
        assert(false && "Unexpected decoration instruction");
    }
  }
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
