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

#include "remove_duplicates_pass.h"

#include <cstring>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "opcode.h"

namespace spvtools {
namespace opt {

using ir::Instruction;
using ir::Module;
using ir::Operand;
using opt::analysis::DefUseManager;

Pass::Status RemoveDuplicatesPass::Process(Module* module) {
  DefUseManager defUseManager(consumer(), module);

  bool modified  = RemoveDuplicateCapabilities(module);
       modified |= RemoveDuplicatesExtInstImports(module, defUseManager);
       modified |= RemoveDuplicateTypes(module, defUseManager);
       modified |= RemoveDuplicateDecorations(module);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool RemoveDuplicatesPass::RemoveDuplicateCapabilities(
    Module* module) const {
  bool modified = false;

  std::unordered_set<uint32_t> capabilities;
  for (auto i = module->capability_begin(); i != module->capability_end();) {
    auto res = capabilities.insert(i->GetSingleWordOperand(0u));

    if (res.second) {
      // Never seen before, keep it.
      ++i;
    } else {
      // It's a duplicate, remove it.
      i = i.Erase();
      modified = true;
    }
  }

  return modified;
}

bool RemoveDuplicatesPass::RemoveDuplicatesExtInstImports(
    Module* module, analysis::DefUseManager& defUseManager) const {
  bool modified = false;

  std::unordered_map<std::string, SpvId> extInstImports;
  for (auto i = module->ext_inst_import_begin();
       i != module->ext_inst_import_end();) {
    auto res = extInstImports.emplace(
        reinterpret_cast<const char*>(i->GetInOperand(0u).words.data()),
        i->result_id());
    if (res.second) {
      // Never seen before, keep it.
      ++i;
    } else {
      // It's a duplicate, remove it.
      defUseManager.ReplaceAllUsesWith(i->result_id(), res.first->second);
      i = i.Erase();
      modified = true;
    }
  }

  return modified;
}

bool RemoveDuplicatesPass::RemoveDuplicateTypes(
    Module* module, analysis::DefUseManager& defUseManager) const {
  bool modified = false;

  std::vector<Instruction> visitedTypes;
  visitedTypes.reserve(module->types_values().size());

  for (auto i = module->types_values_begin();
       i != module->types_values_end();) {
    // We only care about types.
    if (!spvOpcodeGeneratesType((i->opcode())) &&
        i->opcode() != SpvOpTypeForwardPointer) {
      ++i;
      continue;
    }

    // Is the current type equal to one of the types we have aready visited?
    SpvId idToKeep = 0u;
    for (auto j : visitedTypes) {
      if (AreTypesEqual(*i, j, defUseManager)) {
        idToKeep = j.result_id();
        break;
      }
    }

    if (idToKeep == 0u) {
      // This is a never seen before type, keep it around.
      visitedTypes.emplace_back(*i);
      ++i;
    } else {
      // The same type has already been seen before, remove this one.
      defUseManager.ReplaceAllUsesWith(i->result_id(), idToKeep);
      modified = true;
      i = i.Erase();
    }
  }

  return modified;
}

bool RemoveDuplicatesPass::RemoveDuplicateDecorations(ir::Module* module) const {
  bool modified = false;

  std::unordered_map<SpvId, const Instruction*> constants;
  for (const auto& i : module->types_values())
    if (i.opcode() == SpvOpConstant)
      constants[i.result_id()] = &i;
  for (const auto& i : module->types_values())
    if (i.opcode() == SpvOpConstant)
      constants[i.result_id()] = &i;

  std::vector<Instruction> visitedDecorations;
  visitedDecorations.reserve(module->annotations().size());

  for (auto i = module->annotation_begin();
       i != module->annotation_end();) {

    // Is the current decoration equal to one of the decorations we have aready visited?
    bool alreadyVisited = false;
    for (auto j : visitedDecorations) {
      if (AreDecorationsEqual(*i, j, constants)) {
        alreadyVisited = true;
        break;
      }
    }

    if (!alreadyVisited) {
      // This is a never seen before decoration, keep it around.
      visitedDecorations.emplace_back(*i);
      ++i;
    } else {
      // The same decoration has already been seen before, remove this one.
      modified = true;
      i = i.Erase();
    }
  }

  return modified;
}

// TODO(pierremoreau): take decorations into account
// Returns whether two types are equal, decorations not taken into account
// yet.
bool RemoveDuplicatesPass::AreTypesEqual(const Instruction& inst1,
                                         const Instruction& inst2,
                                         const DefUseManager& defUseManager) {
  if (inst1.opcode() != inst2.opcode()) return false;

  switch (inst1.opcode()) {
    case SpvOpTypeVoid:
    case SpvOpTypeBool:
    case SpvOpTypeSampler:
    case SpvOpTypeEvent:
    case SpvOpTypeDeviceEvent:
    case SpvOpTypeReserveId:
    case SpvOpTypeQueue:
    case SpvOpTypePipeStorage:
    case SpvOpTypeNamedBarrier:
      return true;
    case SpvOpTypeInt:
      return inst1.GetSingleWordInOperand(0u) ==
                 inst2.GetSingleWordInOperand(0u) &&
             inst1.GetSingleWordInOperand(1u) ==
                 inst2.GetSingleWordInOperand(1u);
    case SpvOpTypeFloat:
    case SpvOpTypePipe:
    case SpvOpTypeForwardPointer:
      return inst1.GetSingleWordInOperand(0u) ==
             inst2.GetSingleWordInOperand(0u);
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
      return AreTypesEqual(
                 *defUseManager.GetDef(inst1.GetSingleWordInOperand(0u)),
                 *defUseManager.GetDef(inst2.GetSingleWordInOperand(0u)),
                 defUseManager) &&
             inst1.GetSingleWordInOperand(1u) ==
                 inst2.GetSingleWordInOperand(1u);
    case SpvOpTypeImage:
      return AreTypesEqual(
                 *defUseManager.GetDef(inst1.GetSingleWordInOperand(0u)),
                 *defUseManager.GetDef(inst2.GetSingleWordInOperand(0u)),
                 defUseManager) &&
             inst1.GetSingleWordInOperand(1u) ==
                 inst2.GetSingleWordInOperand(1u) &&
             inst1.GetSingleWordInOperand(2u) ==
                 inst2.GetSingleWordInOperand(2u) &&
             inst1.GetSingleWordInOperand(3u) ==
                 inst2.GetSingleWordInOperand(3u) &&
             inst1.GetSingleWordInOperand(4u) ==
                 inst2.GetSingleWordInOperand(4u) &&
             inst1.GetSingleWordInOperand(5u) ==
                 inst2.GetSingleWordInOperand(5u) &&
             inst1.GetSingleWordInOperand(6u) ==
                 inst2.GetSingleWordInOperand(6u) &&
             inst1.NumOperands() == inst2.NumOperands() &&
             (inst1.NumInOperands() == 7u ||
              inst1.GetSingleWordInOperand(7u) ==
                  inst2.GetSingleWordInOperand(7u));
    case SpvOpTypeSampledImage:
    case SpvOpTypeRuntimeArray:
      return AreTypesEqual(
          *defUseManager.GetDef(inst1.GetSingleWordInOperand(0u)),
          *defUseManager.GetDef(inst2.GetSingleWordInOperand(0u)),
          defUseManager);
    case SpvOpTypeArray:
      return AreTypesEqual(
                 *defUseManager.GetDef(inst1.GetSingleWordInOperand(0u)),
                 *defUseManager.GetDef(inst2.GetSingleWordInOperand(0u)),
                 defUseManager) &&
             AreTypesEqual(
                 *defUseManager.GetDef(inst1.GetSingleWordInOperand(1u)),
                 *defUseManager.GetDef(inst2.GetSingleWordInOperand(1u)),
                 defUseManager);
    case SpvOpTypeStruct:
    case SpvOpTypeFunction: {
      bool res = inst1.NumInOperands() == inst2.NumInOperands();
      for (uint32_t i = 0u; i < inst1.NumInOperands() && res; ++i)
        res &= AreTypesEqual(
            *defUseManager.GetDef(inst1.GetSingleWordInOperand(i)),
            *defUseManager.GetDef(inst2.GetSingleWordInOperand(i)),
            defUseManager);
      return res;
    }
    case SpvOpTypeOpaque:
      return std::strcmp(reinterpret_cast<const char*>(
                             inst1.GetInOperand(0u).words.data()),
                         reinterpret_cast<const char*>(
                             inst2.GetInOperand(0u).words.data())) == 0;
    case SpvOpTypePointer:
      return inst1.GetSingleWordInOperand(0u) ==
                 inst2.GetSingleWordInOperand(0u) &&
             AreTypesEqual(
                 *defUseManager.GetDef(inst1.GetSingleWordInOperand(1u)),
                 *defUseManager.GetDef(inst2.GetSingleWordInOperand(1u)),
                 defUseManager);
    default:
      return false;
  }
}

IdDecorationsList RemoveDuplicatesPass::GetDecorationsForId(SpvId id, Module* module) {
  IdDecorationsList decorations;
  auto decoIter = module->annotation_end();

  std::vector<std::pair<SpvId, uint32_t>> idsToLookFor;
  idsToLookFor.emplace_back(id, std::numeric_limits<uint32_t>::max());

  // pierremoreau: Assume that OpGroupDecorate can't target an
  //               OpDecorationGroup.
  do {
    --decoIter;
    if (decoIter->opcode() == SpvOpGroupDecorate) {
      for (uint32_t j = 1u; j < decoIter->NumInOperands(); ++j)
        if (decoIter->GetSingleWordInOperand(j) == id) {
          idsToLookFor.emplace_back(decoIter->GetSingleWordInOperand(0u), std::numeric_limits<uint32_t>::max());
          break;
        }
    } else if (decoIter->opcode() == SpvOpGroupMemberDecorate) {
      for (uint32_t j = 1u; j < decoIter->NumInOperands(); j += 2u)
        if (decoIter->GetSingleWordInOperand(j) == id) {
          idsToLookFor.emplace_back(decoIter->GetSingleWordInOperand(0u), decoIter->GetSingleWordInOperand(j + 1u));
          break;
        }
    } else if (decoIter->opcode() == SpvOpMemberDecorate) {
      if (decoIter->GetSingleWordInOperand(0u) == id)
        decorations[decoIter->GetSingleWordInOperand(1u)].push_back(&*decoIter);
    } else if ((decoIter->opcode() == SpvOpDecorate ||
                decoIter->opcode() == SpvOpDecorateId) &&
               decoIter->GetSingleWordInOperand(1u) != SpvDecorationLinkageAttributes &&
               decoIter->GetSingleWordInOperand(1u) != SpvDecorationFuncParamAttr) { // FuncParamAttr are always taken from the definition anyway
      auto iter = std::find_if(idsToLookFor.cbegin(), idsToLookFor.cend(), [&decoIter](const std::pair<SpvId, uint32_t>& p) {
        return p.first == decoIter->GetSingleWordInOperand(0u);
      });
      if (iter != idsToLookFor.cend())
        decorations[iter->second].push_back(&*decoIter);
    }
  } while (decoIter != module->annotation_begin());

  return decorations;
}

// Remove the whole instruction for SpvOpDecorate, SpvDecorateId and
// SpvMemberDecorate. For group decorations, juste remove the ID (and its
// structure index if present) from the list.
void RemoveDuplicatesPass::RemoveDecorationsFor(SpvId id, Module* module) {
  for (auto i = module->annotation_begin();
      i != module->annotation_end();) {
    if ((i->opcode() == SpvOpDecorate || i->opcode() == SpvOpDecorateId ||
        i->opcode() == SpvOpMemberDecorate) && i->GetSingleWordInOperand(0u) == id)
      i = i.Erase();
    else if (i->opcode() == SpvOpGroupDecorate) {
      std::vector<Operand> operands;
      operands.reserve(i->NumInOperands());
      for (uint32_t j = 2u; j < i->NumInOperands(); ++j)
        if (i->GetSingleWordInOperand(j) != id)
          operands.push_back(i->GetOperand(j));
      *i = Instruction(i->opcode(), 0u, 0u, operands);
    } else if (i->opcode() == SpvOpGroupMemberDecorate) {
      std::vector<Operand> operands;
      operands.reserve(i->NumInOperands());
      for (uint32_t j = 2u; j < i->NumInOperands(); j += 2u) {
        if (i->GetSingleWordInOperand(j) != id) {
          operands.push_back(i->GetOperand(j));
          operands.push_back(i->GetOperand(j + 1u));
        }
      }
      *i = Instruction(i->opcode(), 0u, 0u, operands);
    } else
      ++i;
  }
}

bool RemoveDuplicatesPass::AreDecorationsEqual(const Instruction& deco1, const Instruction& deco2, const std::unordered_map<SpvId, const Instruction*>& constants) {
  const auto decorateIdToDecorate = [&constants](const Instruction& inst) {
    std::vector<Operand> operands;
    operands.reserve(inst.NumInOperands());
    for (uint32_t i = 2u; i < inst.NumInOperands(); ++i) {
      const auto& j = constants.find(inst.GetSingleWordInOperand(i));
      if (j == constants.end())
        return Instruction();
      const auto operand = j->second->GetOperand(0u);
      operands.emplace_back(operand.type, operand.words);
    }
    return Instruction(SpvOpDecorate, 0u, 0u, operands);
  };
  Instruction tmpA = (deco1.opcode() == SpvOpDecorateId) ? decorateIdToDecorate(deco1) : deco1;
  Instruction tmpB = (deco2.opcode() == SpvOpDecorateId) ? decorateIdToDecorate(deco2) : deco2;

  if (tmpA.opcode() != tmpB.opcode() || tmpA.NumInOperands() != tmpB.NumInOperands() ||
      tmpA.opcode() == SpvOpNop || tmpB.opcode() == SpvOpNop)
    return false;

  for (uint32_t i = (tmpA.opcode() == SpvOpDecorate) ? 1u : 2u; i < tmpA.NumInOperands(); ++i)
    if (tmpA.GetInOperand(i) != tmpB.GetInOperand(i))
      return false;

  return true;
}

bool RemoveDuplicatesPass::HaveIdsSimilarDecorations(SpvId id1, SpvId id2, Module* module) {
  const IdDecorationsList decorationsList1 = GetDecorationsForId(id1, module);
  const IdDecorationsList decorationsList2 = GetDecorationsForId(id2, module);

  if (decorationsList1.size() != decorationsList2.size())
    return false;

  // Grab all SpvOpConstant: those should be the only constant instructions
  // used in SpvOpGroupDecorateId besides SpvOpSpecConstant, however there is
  // no way to decide whether two decorations are the same if we rely on value
  // that might change due to specialisation (which should occur before linking
  // anyway?)
  std::unordered_map<SpvId, const Instruction*> constants;
  for (const auto& i : module->types_values())
    if (i.opcode() == SpvOpConstant)
      constants[i.result_id()] = &i;
  for (const auto& i : module->types_values())
    if (i.opcode() == SpvOpConstant)
      constants[i.result_id()] = &i;

  for (const auto& i : decorationsList1) {
    const auto j = decorationsList2.find(i.first);
    if (j == decorationsList2.end() || i.second.size() != j->second.size())
      return false;

    for (size_t k = 0u; k < i.second.size(); ++k)
      if (!AreDecorationsEqual(*i.second[k], *j->second[k], constants))
        return false;
  }

  return true;
}

}  // namespace opt
}  // namespace spvtools
