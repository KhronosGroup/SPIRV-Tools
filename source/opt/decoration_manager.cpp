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

namespace spvtools {
namespace opt {
namespace analysis {

void DecorationManager::RemoveDecorationsFrom(uint32_t id) {
  auto const ids_iter = id_to_decoration_insts_.find(id);
  if (ids_iter == id_to_decoration_insts_.end())
    return;

  for (ir::Instruction* inst : ids_iter->second) {
    switch (inst->opcode()) {
      case SpvOpDecorate:
      case SpvOpDecorateId:
      case SpvOpMemberDecorate:
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
    uint32_t id) {
  std::vector<ir::Instruction*> decorations;
  return decorations;
}

bool DecorationManager::HaveTheSameDecorations(uint32_t id1, uint32_t id2) {
  return false;
}

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
//  if (tmpA.opcode() != tmpB.opcode() || tmpA.NumInOperands() != tmpB.NumInOperands() ||
//      tmpA.opcode() == SpvOpNop || tmpB.opcode() == SpvOpNop)
//    return false;
//
//  for (uint32_t i = (tmpA.opcode() == SpvOpDecorate) ? 1u : 2u; i < tmpA.NumInOperands(); ++i)
//    if (tmpA.GetInOperand(i) != tmpB.GetInOperand(i))
//      return false;
//
//  return true;
  return false;
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
