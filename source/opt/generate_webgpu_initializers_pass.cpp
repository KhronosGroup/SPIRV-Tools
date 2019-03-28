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

bool NeedsWebGPUInitializer(const inst_iterator inst) {
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

// Attempts to add a null initializer to the instruction. If the constant for
// the null initializer is defined after the instruction, it will be
// moved to be earlier and then the the null initialer will be added.
// Caller is responsible for ensuring the instruction needs to have an
// initializer added.
void AddNullInitializer(IRContext* context, inst_iterator inst,
                        void(ensure_order)(Module*, uint32_t, inst_iterator) =
                            [](Module*, uint32_t, InstructionList::iterator)
                            -> void { }) {
  auto constant_mgr = context->get_constant_mgr();

  auto* constant_type =
      constant_mgr->GetType(&(*inst))->AsPointer()->pointee_type();
  auto* constant = constant_mgr->GetConstant(constant_type, {});
  auto* constant_inst = constant_mgr->GetDefiningInstruction(constant);
  auto constant_id = constant_inst->result_id();

  ensure_order(context->module(), constant_id, inst);

  inst->AddOperand(Operand(SPV_OPERAND_TYPE_ID, {constant_id}));
  context->UpdateDefUse(&(*inst));
  context->UpdateDefUse(constant_inst);
}

}  // namespace

Pass::Status GenerateWebGPUInitializersPass::Process() {
  auto* module = context()->module();
  bool changed = false;

  // Handle global/module scoped variables
  for (auto inst = module->types_values_begin();
       inst != module->types_values_end(); ++inst) {
    if (!NeedsWebGPUInitializer(inst)) continue;

    changed = true;
    AddNullInitializer(context(), inst,
                       [](Module* inner_module, uint32_t constant_id,
                          InstructionList::iterator inner_inst) {
                         inner_module->EnsureIdDefinedBeforeInstruction(
                             constant_id, inner_inst);
                       });
  }

  // Handle local/function scoped variables
  for (auto func = module->begin(); func != module->end(); ++func) {
    for (auto block = func->begin(); block != func->end(); ++block) {
      for (auto inst = block->begin(); inst != block->end(); ++inst) {
        if (!NeedsWebGPUInitializer(inst)) continue;

        changed = true;
        AddNullInitializer(context(), inst);
      }
    }
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
