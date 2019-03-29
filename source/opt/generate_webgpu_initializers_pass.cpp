// Copyright (c) 2019 Google Inc.
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

#include "source/opt/generate_webgpu_initializers_pass.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

using inst_iterator = InstructionList::iterator;

namespace {

bool NeedsWebGPUInitializer(Instruction* inst) {
  if (inst->opcode() != SpvOpVariable) return false;

  auto storage_class = inst->GetSingleWordOperand(2);
  if (storage_class != SpvStorageClassOutput &&
      storage_class != SpvStorageClassPrivate &&
      storage_class != SpvStorageClassFunction) {
    return false;
  }

  if (inst->NumOperands() > 3) return false;

  return true;
}

}  // namespace

Pass::Status GenerateWebGPUInitializersPass::Process() {
  auto constant_mgr = context()->get_constant_mgr();
  auto* module = context()->module();
  bool changed = false;

  // Handle global/module scoped variables
  for (auto iter = module->types_values_begin();
       iter != module->types_values_end(); ++iter) {
    Instruction* inst = &(*iter);

    if (inst->opcode() == SpvOpConstantNull) {
      auto* type = constant_mgr->GetType(inst);
      null_constant_type_map_[type] = inst;
      seen_null_constants_.insert(inst);
      continue;
    }

    if (!NeedsWebGPUInitializer(inst)) continue;

    changed = true;

    auto* constant_inst = GetNullConstantForVariable(inst);
    if (seen_null_constants_.find(constant_inst) ==
        seen_null_constants_.end()) {
      constant_inst->RemoveFromList();
      constant_inst->InsertBefore(inst);
      auto* type = constant_mgr->GetType(constant_inst);
      null_constant_type_map_[type] = inst;
      seen_null_constants_.insert(inst);
    }
    AddNullInitializerToVariable(constant_inst, inst);
  }

  // Handle local/function scoped variables
  for (auto func = module->begin(); func != module->end(); ++func) {
    auto block = func->entry().get();
    for (auto iter = block->begin();
         iter != block->end() && iter->opcode() == SpvOpVariable; ++iter) {
      Instruction* inst = &(*iter);
      if (!NeedsWebGPUInitializer(inst)) continue;

      changed = true;
      auto* constant_inst = GetNullConstantForVariable(inst);
      AddNullInitializerToVariable(constant_inst, inst);
    }
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Instruction* GenerateWebGPUInitializersPass::GetNullConstantForVariable(
    Instruction* variable_inst) {
  auto constant_mgr = context()->get_constant_mgr();
  auto* constant_type = context()
                            ->get_constant_mgr()
                            ->GetType(variable_inst)
                            ->AsPointer()
                            ->pointee_type();

  if (null_constant_type_map_.find(constant_type) ==
      null_constant_type_map_.end()) {
    auto* constant = constant_mgr->GetConstant(constant_type, {});
    return constant_mgr->GetDefiningInstruction(constant);
  } else {
    return null_constant_type_map_[constant_type];
  }
}

void GenerateWebGPUInitializersPass::AddNullInitializerToVariable(
    Instruction* constant_inst, Instruction* variable_inst) {
  auto constant_id = constant_inst->result_id();
  variable_inst->AddOperand(Operand(SPV_OPERAND_TYPE_ID, {constant_id}));
  context()->UpdateDefUse(variable_inst);
  context()->UpdateDefUse(constant_inst);
}

}  // namespace opt
}  // namespace spvtools
