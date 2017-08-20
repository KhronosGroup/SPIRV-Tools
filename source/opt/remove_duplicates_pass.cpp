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

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "opcode.h"

namespace spvtools {
namespace opt {

using ir::Instruction;
using ir::Operand;
using opt::analysis::DefUseManager;

Pass::Status RemoveDuplicatesPass::Process(ir::Module* module) {
  DefUseManager defUseManager(consumer(), module);

  bool modified  = RemoveDuplicateCapabilities(module);
       modified |= RemoveDuplicatesExtInstImports(module, defUseManager);
       modified |= RemoveDuplicateTypes(module, defUseManager);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool RemoveDuplicatesPass::RemoveDuplicateCapabilities(
    ir::Module* module) const {
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
    ir::Module* module, analysis::DefUseManager& defUseManager) const {
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
    ir::Module* module, analysis::DefUseManager& defUseManager) const {
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

// TODO(pierremoreau): take decorations into account
// Returns whether two types are equal, decorations not taken into account
// yet., decorations not taken into account yet.
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

}  // namespace opt
}  // namespace spvtools
