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

#include "decoration_manager.h"
#include "ir_context.h"
#include "opcode.h"
#include "reflect.h"

namespace spvtools {
namespace opt {

using ir::Instruction;
using ir::Module;
using ir::Operand;
using opt::analysis::DecorationManager;
using opt::analysis::DefUseManager;

Pass::Status RemoveDuplicatesPass::Process(ir::IRContext* irContext) {
  bool modified = RemoveDuplicateCapabilities(irContext);
  modified |= RemoveDuplicatesExtInstImports(irContext);
  modified |= RemoveDuplicateTypes(irContext);
  modified |= RemoveDuplicateDecorations(irContext);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool RemoveDuplicatesPass::RemoveDuplicateCapabilities(
    ir::IRContext* irContext) const {
  bool modified = false;

  if (irContext->capabilities().empty()) {
    return modified;
  }

  std::unordered_set<uint32_t> capabilities;
  for (auto* i = &*irContext->capability_begin(); i;) {
    auto res = capabilities.insert(i->GetSingleWordOperand(0u));

    if (res.second) {
      // Never seen before, keep it.
      i = i->NextNode();
    } else {
      // It's a duplicate, remove it.
      i = irContext->KillInst(i);
      modified = true;
    }
  }

  return modified;
}

bool RemoveDuplicatesPass::RemoveDuplicatesExtInstImports(
    ir::IRContext* irContext) const {
  bool modified = false;

  if (irContext->ext_inst_imports().empty()) {
    return modified;
  }

  std::unordered_map<std::string, SpvId> extInstImports;
  for (auto* i = &*irContext->ext_inst_import_begin(); i;) {
    auto res = extInstImports.emplace(
        reinterpret_cast<const char*>(i->GetInOperand(0u).words.data()),
        i->result_id());
    if (res.second) {
      // Never seen before, keep it.
      i = i->NextNode();
    } else {
      // It's a duplicate, remove it.
      irContext->ReplaceAllUsesWith(i->result_id(), res.first->second);
      i = irContext->KillInst(i);
      modified = true;
    }
  }

  return modified;
}

bool RemoveDuplicatesPass::RemoveDuplicateTypes(
    ir::IRContext* irContext) const {
  bool modified = false;

  if (irContext->types_values().empty()) {
    return modified;
  }

  std::vector<Instruction*> visitedTypes;
  std::vector<Instruction*> toDelete;
  for (auto* i = &*irContext->types_values_begin(); i; i = i->NextNode()) {
    // We only care about types.
    if (!spvOpcodeGeneratesType((i->opcode())) &&
        i->opcode() != SpvOpTypeForwardPointer) {
      continue;
    }

    // Is the current type equal to one of the types we have aready visited?
    SpvId idToKeep = 0u;
    for (auto j : visitedTypes) {
      if (AreTypesEqual(*i, *j, irContext)) {
        idToKeep = j->result_id();
        break;
      }
    }

    if (idToKeep == 0u) {
      // This is a never seen before type, keep it around.
      visitedTypes.emplace_back(i);
    } else {
      // The same type has already been seen before, remove this one.
      irContext->ReplaceAllUsesWith(i->result_id(), idToKeep);
      modified = true;
      toDelete.emplace_back(i);
    }
  }

  for (auto i : toDelete) {
    irContext->KillInst(i);
  }

  return modified;
}

// TODO(pierremoreau): Duplicate decoration groups should be removed. For
// example, in
//     OpDecorate %1 Constant
//     %1 = OpDecorationGroup
//     OpDecorate %2 Constant
//     %2 = OpDecorationGroup
//     OpGroupDecorate %1 %3
//     OpGroupDecorate %2 %4
// group %2 could be removed.
bool RemoveDuplicatesPass::RemoveDuplicateDecorations(
    ir::IRContext* irContext) const {
  bool modified = false;

  std::vector<const Instruction*> visitedDecorations;

  opt::analysis::DecorationManager decorationManager(irContext->module());
  for (auto* i = &*irContext->annotation_begin(); i;) {
    // Is the current decoration equal to one of the decorations we have aready
    // visited?
    bool alreadyVisited = false;
    for (const Instruction* j : visitedDecorations) {
      if (decorationManager.AreDecorationsTheSame(&*i, j, false)) {
        alreadyVisited = true;
        break;
      }
    }

    if (!alreadyVisited) {
      // This is a never seen before decoration, keep it around.
      visitedDecorations.emplace_back(&*i);
      i = i->NextNode();
    } else {
      // The same decoration has already been seen before, remove this one.
      modified = true;
      i = irContext->KillInst(i);
    }
  }

  return modified;
}

bool RemoveDuplicatesPass::AreTypesEqual(const Instruction& inst1,
                                         const Instruction& inst2,
                                         ir::IRContext* context) {
  if (inst1.opcode() != inst2.opcode()) return false;
  if (!ir::IsTypeInst(inst1.opcode())) return false;

  const analysis::Type* type1 =
      context->get_type_mgr()->GetType(inst1.result_id());
  const analysis::Type* type2 =
      context->get_type_mgr()->GetType(inst2.result_id());
  if (type1 && type2 && *type1 == *type2) return true;

  return false;
}

}  // namespace opt
}  // namespace spvtools
