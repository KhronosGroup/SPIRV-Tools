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

#include <stack>

namespace spvtools {
namespace opt {
namespace analysis {

void DecorationManager::RemoveDecorationsFrom(uint32_t id, bool keep_linkage) {
  auto const ids_iter = id_to_decoration_insts_.find(id);
  if (ids_iter == id_to_decoration_insts_.end())
    return;

  for (ir::Instruction* inst : ids_iter->second) {
    switch (inst->opcode()) {
      case SpvOpDecorate:
      case SpvOpDecorateId:
      case SpvOpMemberDecorate:
        if (!(keep_linkage && inst->GetSingleWordInOperand(1u) == SpvDecorationLinkageAttributes))
          inst->ToNop();
        break;
      case SpvOpGroupDecorate:
        for (uint32_t i = 1u; i < inst->NumInOperands(); ++i) {
          if (inst->GetSingleWordInOperand(i) == inst->result_id()) {
            inst->RemoveInOperand(i);
            break;
          }
        }
        break;
      case SpvOpGroupMemberDecorate:
        for (uint32_t i = 1u; i < inst->NumInOperands(); i += 2u) {
          if (inst->GetSingleWordInOperand(i) == inst->result_id()) {
            inst->RemoveInOperand(i);
            break;
          }
        }
        break;
      default:
        break;
    }
  }
}

std::vector<ir::Instruction*> DecorationManager::GetDecorationsFor(
    uint32_t id, bool include_linkage) {
  std::vector<ir::Instruction*> decorations;
  std::stack<uint32_t> ids_to_process;
  ids_to_process.push(id);
  const auto process = [&ids_to_process,&decorations](ir::Instruction* inst){
    if (inst->opcode() == SpvOpGroupDecorate || inst->opcode() == SpvOpGroupMemberDecorate)
      ids_to_process.push(inst->GetSingleWordInOperand(0u));
    else
      decorations.push_back(inst);
  };
  while (!ids_to_process.empty()) {
    const uint32_t id_to_process = ids_to_process.top();
    ids_to_process.pop();
    const auto group_iter = group_to_decoration_insts_.find(id_to_process);
    if (group_iter != group_to_decoration_insts_.end()) {
      for (ir::Instruction* inst : group_iter->second)
        process(inst);
    } else {
      const auto ids_iter = id_to_decoration_insts_.find(id_to_process);
      if (ids_iter == id_to_decoration_insts_.end())
        return std::vector<ir::Instruction*>();
      for (ir::Instruction* inst : ids_iter->second) {
        const bool is_linkage = inst->opcode() == SpvOpDecorate && inst->GetSingleWordInOperand(1u) == SpvDecorationLinkageAttributes;
        if (include_linkage || !is_linkage)
          process(inst);
      }
    }
  }
  return decorations;
}

// TODO(pierremoreau): The code will return true for { deco1, deco1 }, { deco1,
//                     deco2 } when it should return false.
bool DecorationManager::HaveTheSameDecorations(uint32_t id1, uint32_t id2) {
  const auto decorationsFor1 = GetDecorationsFor(id1, false);
  const auto decorationsFor2 = GetDecorationsFor(id2, false);
  if (decorationsFor1.size() != decorationsFor2.size())
    return false;

  for (const ir::Instruction* inst1 : decorationsFor1) {
    bool didFindAMatch = false;
    for (const ir::Instruction* inst2 : decorationsFor2) {
      if (AreDecorationsTheSame(inst1, inst2)) {
        didFindAMatch = true;
        break;
      }
    }
    if (!didFindAMatch)
      return false;
  }
  return true;
}

// TODO(pierremoreau): Handle SpvOpDecorateId by converting them to a regular
//                     SpvOpDecorate.
bool DecorationManager::AreDecorationsTheSame(const ir::Instruction* inst1, const ir::Instruction* inst2) {
//  const auto decorateIdToDecorate = [&constants](const Instruction& inst) {
//    std::vector<Operand> operands;
//    operands.reserve(inst.NumInOperands());
//    for (uint32_t i = 2u; i < inst.NumInOperands(); ++i) {
//      const auto& j = constants.find(inst.GetSingleWordInOperand(i));
//      if (j == constants.end())
//        return Instruction();
//      const auto operand = j->second->GetOperand(0u);
//      operands.emplace_back(operand.type, operand.words);
//    }
//    return Instruction(SpvOpDecorate, 0u, 0u, operands);
//  };
//  Instruction tmpA = (deco1.opcode() == SpvOpDecorateId) ? decorateIdToDecorate(deco1) : deco1;
//  Instruction tmpB = (deco2.opcode() == SpvOpDecorateId) ? decorateIdToDecorate(deco2) : deco2;
//
  if (inst1->opcode() == SpvOpDecorateId || inst2->opcode() == SpvOpDecorateId)
    return false;

  ir::Instruction tmpA = *inst1, tmpB = *inst2;
  if (tmpA.opcode() != tmpB.opcode() || tmpA.NumInOperands() != tmpB.NumInOperands() ||
      tmpA.opcode() == SpvOpNop || tmpB.opcode() == SpvOpNop)
    return false;

  for (uint32_t i = (tmpA.opcode() == SpvOpDecorate) ? 1u : 2u; i < tmpA.NumInOperands(); ++i)
    if (tmpA.GetInOperand(i) != tmpB.GetInOperand(i))
      return false;

  return true;
}

void DecorationManager::AnalyzeDecorations(ir::Module* module) {
  if (!module) return;

  // Collect all group ids.
  for (const ir::Instruction& inst : module->annotations()) {
    switch (inst.opcode()) {
      case SpvOpDecorationGroup:
        group_to_decoration_insts_.insert({inst.GetSingleWordInOperand(0u), {}});
        break;
      default:
        break;
    }
  }

  // For each group and instruction, collect all their decoration instructions.
  for (ir::Instruction& inst : module->annotations()) {
    switch (inst.opcode()) {
      case SpvOpDecorate:
      case SpvOpDecorateId:
      case SpvOpMemberDecorate: {
        auto const target_id = inst.GetSingleWordInOperand(0u);
        auto const group_iter =
            group_to_decoration_insts_.find(target_id);
        if (group_iter != group_to_decoration_insts_.end())
          group_iter->second.push_back(&inst);
        else
          id_to_decoration_insts_[target_id].push_back(&inst);
        break;
      }
      case SpvOpGroupDecorate:
        for (uint32_t i = 1u; i < inst.NumInOperands(); ++i) {
          auto const target_id = inst.GetSingleWordInOperand(i);
          auto const group_iter =
              group_to_decoration_insts_.find(target_id);
          if (group_iter != group_to_decoration_insts_.end())
            group_iter->second.push_back(&inst);
          else
            id_to_decoration_insts_[target_id].push_back(&inst);
        }
        break;
      case SpvOpGroupMemberDecorate:
        for (uint32_t i = 1u; i < inst.NumInOperands(); i += 2u) {
          auto const target_id = inst.GetSingleWordInOperand(i);
          auto const group_iter =
              group_to_decoration_insts_.find(target_id);
          if (group_iter != group_to_decoration_insts_.end())
            group_iter->second.push_back(&inst);
          else
            id_to_decoration_insts_[target_id].push_back(&inst);
        }
        break;
      default:
        break;
    }
  }
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
