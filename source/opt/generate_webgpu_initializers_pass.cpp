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
// the null initializer is defined after the instruction, it will instead be
// moved to be earlier and an early exit will occur.
// Reordering instructions potentially breaks the iterators, so cannot complete
// operation.
// It is the responsibility of the caller to check the return value and re-run
// iteration if a reordering has occured.
// Caller is responsible for ensuring the instruction needs to have an
// initializer added.
//
// Returns true, if the instruction has had the null initializer added.
// Returns false, if the instructions were instead reordered, so iteration needs
// to be restarted.
//
// |ensure_order| returns true if changes were needed to ensure the order,
// otherwise false.

bool AddNullInitializer(
    IRContext* context, inst_iterator inst,
    bool(ensure_order)(Module*, uint32_t, inst_iterator) =
        [](Module*, uint32_t, InstructionList::iterator) -> bool {
      return false;
    }) {
  auto constant_mgr = context->get_constant_mgr();

  auto* constant_type =
      constant_mgr->GetType(&(*inst))->AsPointer()->pointee_type();
  auto* constant = constant_mgr->GetConstant(constant_type, {});
  auto* constant_inst = constant_mgr->GetDefiningInstruction(constant);
  auto constant_id = constant_inst->result_id();

  if (ensure_order(context->module(), constant_id, inst)) {
    return false;
  }

  inst->AddOperand(Operand(SPV_OPERAND_TYPE_ID, {constant_id}));
  context->UpdateDefUse(&(*inst));
  context->UpdateDefUse(constant_inst);

  return true;
}

}  // namespace

Pass::Status GenerateWebGPUInitializersPass::Process() {
  auto* module = context()->module();
  bool changed = false;
  bool instructions_stable;

  // Handle global/module scoped variables
  do {
    instructions_stable = true;
    for (auto inst = module->types_values_begin();
         inst != module->types_values_end(); ++inst) {
      if (!NeedsWebGPUInitializer(inst)) continue;

      changed = true;
      instructions_stable = AddNullInitializer(
          context(), inst,
          [](Module* inner_module, uint32_t constant_id,
             InstructionList::iterator inner_inst) {
            return inner_module->EnsureIdDefinedBeforeInstruction(constant_id,
                                                                  inner_inst);
          });
      if (!instructions_stable) break;
    }
  } while (!instructions_stable);

  // Handle local/function scoped variables
  for (auto func = module->begin(); func != module->end(); ++func) {
    for (auto block = func->begin(); block != func->end(); ++block) {
      for (auto inst = block->begin(); inst != block->end(); ++inst) {
        if (!NeedsWebGPUInitializer(inst)) continue;

        changed = true;
        // Do not need to check the return value, because the
        AddNullInitializer(context(), inst);
      }
    }
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
