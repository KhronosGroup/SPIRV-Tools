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

namespace {

using spvtools::ir::Instruction;

static bool areTypesSimilar(
    const Instruction& a, const Instruction& b,
    std::unordered_map<SpvId, const Instruction>& typesMap) {
  if (a.opcode() != b.opcode()) return false;

  switch (a.opcode()) {
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
      return a.GetSingleWordInOperand(0u) == b.GetSingleWordInOperand(0u) &&
             a.GetSingleWordInOperand(1u) == b.GetSingleWordInOperand(1u);
    case SpvOpTypeFloat:
    case SpvOpTypePipe:
    case SpvOpTypeForwardPointer:
      return a.GetSingleWordInOperand(0u) == b.GetSingleWordInOperand(0u);
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
      return areTypesSimilar(typesMap[a.GetSingleWordInOperand(0u)],
                             typesMap[b.GetSingleWordInOperand(0u)],
                             typesMap) &&
             a.GetSingleWordInOperand(1u) == b.GetSingleWordInOperand(1u);
    case SpvOpTypeImage:
      return areTypesSimilar(typesMap[a.GetSingleWordInOperand(0u)],
                             typesMap[b.GetSingleWordInOperand(0u)],
                             typesMap) &&
             a.GetSingleWordInOperand(1u) == b.GetSingleWordInOperand(1u) &&
             a.GetSingleWordInOperand(2u) == b.GetSingleWordInOperand(2u) &&
             a.GetSingleWordInOperand(3u) == b.GetSingleWordInOperand(3u) &&
             a.GetSingleWordInOperand(4u) == b.GetSingleWordInOperand(4u) &&
             a.GetSingleWordInOperand(5u) == b.GetSingleWordInOperand(5u) &&
             a.GetSingleWordInOperand(6u) == b.GetSingleWordInOperand(6u) &&
             a.NumOperands() == b.NumOperands() &&
             (a.NumInOperands() == 8u ||
              a.GetSingleWordInOperand(7u) == b.GetSingleWordInOperand(7u));
    case SpvOpTypeSampledImage:
    case SpvOpTypeRuntimeArray:
      return areTypesSimilar(typesMap[a.GetSingleWordInOperand(0u)],
                             typesMap[b.GetSingleWordInOperand(0u)], typesMap);
    case SpvOpTypeArray:
      return areTypesSimilar(typesMap[a.GetSingleWordInOperand(0u)],
                             typesMap[b.GetSingleWordInOperand(0u)],
                             typesMap) &&
             areTypesSimilar(typesMap[a.GetSingleWordInOperand(1u)],
                             typesMap[b.GetSingleWordInOperand(1u)], typesMap);
    case SpvOpTypeStruct:
    case SpvOpTypeFunction: {
      bool res = a.NumInOperands() == b.NumInOperands();
      for (uint32_t i = 0u; i < a.NumInOperands() && res; ++i)
        res &= areTypesSimilar(typesMap[a.GetSingleWordInOperand(i)],
                               typesMap[b.GetSingleWordInOperand(i)], typesMap);
      return res;
    }
    case SpvOpTypeOpaque:
      return std::strcmp(
                 reinterpret_cast<const char*>(a.GetInOperand(0u).words.data()),
                 reinterpret_cast<const char*>(
                     b.GetInOperand(0u).words.data())) == 0;
    case SpvOpTypePointer:
      return a.GetSingleWordInOperand(0u) == b.GetSingleWordInOperand(0u) &&
             areTypesSimilar(typesMap[a.GetSingleWordInOperand(1u)],
                             typesMap[b.GetSingleWordInOperand(1u)], typesMap);
    default:
      return false;
  }
}

}  // anonymous namespace

namespace spvtools {
namespace opt {

using ir::Instruction;
using ir::Operand;

Pass::Status RemoveDuplicatesPass::Process(ir::Module* module) {
  bool modified = false;

  // Remove duplicate capabilities
  std::unordered_set<uint32_t> capabilities;
  for (auto i = module->capability_begin(); i != module->capability_end();) {
    auto res = capabilities.insert(i->GetSingleWordOperand(0u));
    i = (res.second) ? ++i : i.Erase();
    modified |= res.second;
  }

  // Remove duplicate ext inst imports
  std::unordered_map<std::string, SpvId> extInstImports;
  std::unordered_map<SpvId, SpvId> extInstImportsReplacementMap;
  for (auto i = module->ext_inst_import_begin();
       i != module->ext_inst_import_end();) {
    auto res = extInstImports.emplace(
        reinterpret_cast<const char*>(i->GetInOperand(0u).words.data()),
        i->result_id());
    if (!res.second)
      extInstImportsReplacementMap[i->result_id()] = res.first->second;
    i = (res.second) ? ++i : i.Erase();
    modified |= res.second;
  }
  module->ForEachInst(
      [&extInstImportsReplacementMap](Instruction* inst) {
        inst->ForEachInId([&extInstImportsReplacementMap](uint32_t* op) {
          auto iter = extInstImportsReplacementMap.find(*op);
          if (iter != extInstImportsReplacementMap.end()) *op = iter->second;
        });
      },
      false);

  // Remove duplicate types
  // TODO(pierremoreau): optimise this part of the pass
  const auto isType = [](SpvOp op) {
    switch (op) {
    case SpvOpTypeVoid:
    case SpvOpTypeBool:
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
    case SpvOpTypeImage:
    case SpvOpTypeSampler:
    case SpvOpTypeSampledImage:
    case SpvOpTypeArray:
    case SpvOpTypeRuntimeArray:
    case SpvOpTypeStruct:
    case SpvOpTypeOpaque:
    case SpvOpTypePointer:
    case SpvOpTypeFunction:
    case SpvOpTypeEvent:
    case SpvOpTypeDeviceEvent:
    case SpvOpTypeReserveId:
    case SpvOpTypeQueue:
    case SpvOpTypePipe:
    case SpvOpTypeForwardPointer:
    case SpvOpTypePipeStorage:
    case SpvOpTypeNamedBarrier:
      return true;
    default:
      return false;
    }
  };

  std::unordered_map<SpvId, const Instruction> typesMap;
  typesMap.reserve(module->types_values().size());
  for (auto i : module->types_values())
    if (isType(i.opcode()))
      typesMap.emplace(i.result_id(), i);

  std::vector<Instruction> visitedTypes;
  visitedTypes.reserve(module->types_values().size());
  std::unordered_map<SpvId, SpvId> typesReplacementMap;
  typesReplacementMap.reserve(module->types_values().size());


  for (auto i = module->types_values_begin();
       i != module->types_values_end();) {
    if (!isType(i->opcode())) {
      ++i;
      continue;
    }

    SpvId idToKeep = 0u;
    for (auto j : visitedTypes) {
      if (areTypesSimilar(*i, j, typesMap)) {
        idToKeep = j.result_id();
        break;
      }
    }
    if (idToKeep == 0u)
      visitedTypes.emplace_back(*i);
    else
      typesReplacementMap[i->result_id()] = idToKeep;
    i = (idToKeep == 0u) ? ++i : i.Erase();
  }

  if (!typesReplacementMap.empty()) {
    module->ForEachInst(
        [&typesReplacementMap](Instruction* inst) {
          inst->ForEachId([&typesReplacementMap](uint32_t* op) {
            auto iter = typesReplacementMap.find(*op);
            if (iter != typesReplacementMap.end()) *op = iter->second;
          });
        },
        false);
    modified = true;
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
