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

#include "eliminate_dead_code_pass.h"

#include <algorithm>
#include <unordered_set>

#include "def_use_manager.h"
#include "instruction.h"
#include "log.h"
#include "operand.h"
#include "reflect.h"

namespace spvtools {
namespace opt {

namespace {

// Returns true if the given |inst| is a dead instruction:
// * generating result id, and
// * no uses or uses are all debug instructions, and
// * if is variable-defining instruction, having internal visibility.
bool IsDeadInst(const analysis::DefUseManager& def_use_mgr,
                const ir::Instruction& inst) {
  // Instructions not generating result ids are used for their side effects.
  if (inst.result_id() == 0) return false;
  // For instructions with result ids and uses, we need to check their uses.
  if (auto* uses = def_use_mgr.GetUses(inst.result_id())) {
    // Annotations can change the semantics of the annotated instructions. We
    // treat all instructions with annotations as non-dead, because we don't
    // know whether there will be new annotations introduced in the future
    // having effect of making instructions from deletable to non-deletable.
    // Just for future proof.
    // TODO(antiagainst): But some known decorations can be handled, e.g., an
    // instruction annotated with only RelaxedPrecision.
    return std::all_of(uses->cbegin(), uses->cend(),
                       [](const analysis::Use& use) {
                         return ir::IsDebugInst(use.inst->opcode());
                       });
  }

  // Instructions with result ids but no uses.

  if (inst.opcode() != SpvOpVariable) return true;
  // Lots of storage classes mean external visibility. For safety, do not
  // consider them as dead instructions even without uses.
  switch (inst.GetSingleWordInOperand(0)) {
    case SpvStorageClassPrivate:
    case SpvStorageClassFunction:
    case SpvStorageClassGeneric:
    case SpvStorageClassAtomicCounter:
    case SpvStorageClassImage:
      return true;
    default:
      break;
  }
  return false;
}

}  // anonymous namespace

Pass::Status EliminateDeadCodePass::Process(ir::Module* module) {
  SPIRV_DEBUG(consumer(), "starting eliminate-dead-code processing");

  bool changed = false;
  analysis::DefUseManager def_use_mgr(consumer(), module);
  std::unordered_set<ir::Instruction*> candidates;

  // If |inst| is a dead instruction, pushes all its id operands to
  // |candidates|, kills |inst| and returns true. Otherwise, does
  // nothing and returns false.
  auto handle_dead_inst_candiate =
      [this, &def_use_mgr, &candidates](ir::Instruction* inst) -> bool {
    if (!IsDeadInst(def_use_mgr, *inst)) return false;
    // Still have uses for this dead code: they are just debug or annotation
    // instructions.
    if (auto* uses = def_use_mgr.GetUses(inst->result_id())) {
      for (auto& use : *uses) use.inst->ToNop();
    }
    // All id operands should be considered as dead instruction candidates.
    for (auto& operand : *inst) {
      if (spvIsIdType(operand.type)) {
        candidates.insert(def_use_mgr.GetDef(operand.words.front()));
      }
    }
    SPIRV_DEBUG(consumer(), "killed dead instruction [id = %u]",
                inst->result_id());
    def_use_mgr.KillInst(inst);
    return true;
  };

  for (auto& import : module->ext_inst_imports()) {
    changed |= handle_dead_inst_candiate(&import);
  }
  for (auto& tv : module->types_values()) {
    changed |= handle_dead_inst_candiate(&tv);
  }
  // Process all instructions inside basic blocks.
  for (auto& function : *module) {
    for (auto& block : function) {
      for (auto& inst : block) changed |= handle_dead_inst_candiate(&inst);
    }
  }

  // Recursively process dead instruction candidates.
  while (!candidates.empty()) {
    ir::Instruction* inst = *candidates.begin();
    changed |= handle_dead_inst_candiate(inst);
    candidates.erase(inst);
  }

  SPIRV_DEBUG(consumer(), "finishing eliminate-dead-code processing");
  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
