// Copyright (c) 2017 Google Inc.
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

#include "dead_variable_elimination.h"

#include "reflect.h"

namespace spvtools {
namespace opt {

Pass::Status DeadVariableElimination::Process(spvtools::ir::Module* module) {
  bool modified = false;
  module_ = module;
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));
  FindNamedOrDecoratedIds();
  decoration_manager_ = std::unique_ptr<analysis::DecorationManager>(
      new analysis::DecorationManager(module));

  std::vector<uint32_t> ids_to_remove;

  // Get the reference count for all of the global OpVariable instructions.
  for (auto& inst : module->types_values()) {
    if (inst.opcode() == SpvOp::SpvOpVariable) {
      size_t count = 0;
      uint32_t result_id = inst.result_id();

      // Check the linkage.  If it is exported, it could be reference somewhere
      // else, so we must keep the variable around.
      const ir::Instruction* linkage_instruction =
          decoration_manager_->GetDecoration(result_id,
                                             SpvDecorationLinkageAttributes);
      if (linkage_instruction != NULL) {
        uint32_t last_operand = linkage_instruction->NumOperands() - 1;
        if (linkage_instruction->GetSingleWordOperand(last_operand) ==
            SpvLinkageTypeExport) {
          count = MUST_KEEP;
        }
      }

      if (count != MUST_KEEP) {
        // If we don't have to keep the instruction for other reasons, then look
        // at the uses and count the number of real references.
        if (analysis::UseList* uses = def_use_mgr_->GetUses(result_id)) {
          count = std::count_if(
              uses->begin(), uses->end(), [](const analysis::Use& u) {
                return (!ir::IsAnnotationInst(u.inst->opcode()) &&
                    u.inst->opcode() != SpvOpName);
              });
        }
      }
      reference_count_[result_id] = count;
      if (count == 0) {
        ids_to_remove.push_back(result_id);
      }
    }
  }

  // Remove all of the variables that have a reference count of 0.
  if (!ids_to_remove.empty()) {
    modified = true;
    for (auto result_id : ids_to_remove) {
      DeleteVariable(result_id);
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

void DeadVariableElimination::DeleteVariable(uint32_t result_id) {
  ir::Instruction* inst = def_use_mgr_->GetDef(result_id);
  assert(inst->opcode() == SpvOpVariable &&
      "Should not be trying to delete anything other than an OpVariable.");

  // Look for an initializer that references another variable.  We need to know
  // if that variable can be deleted after the reference is removed.
  if (inst->NumOperands() == 4) {
    ir::Instruction* initializer =
        def_use_mgr_->GetDef(inst->GetSingleWordOperand(3));
    if (initializer->opcode() == SpvOpVariable) {
      uint32_t initializer_id = initializer->result_id();
      size_t& count = reference_count_[initializer_id];
      if (count != MUST_KEEP) {
        --count;
      }

      if (count == 0) {
        DeleteVariable(initializer_id);
      }
    }
  }
  this->KillNamesAndDecorates(result_id);
  def_use_mgr_->KillDef(result_id);
}
}  // namespace opt
}  // namespace spvtools
