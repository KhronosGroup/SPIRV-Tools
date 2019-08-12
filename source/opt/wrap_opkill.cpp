// Copyright (c) 2019 Google LLC
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

#include "source/opt/wrap_opkill.h"

#include "ir_builder.h"

namespace spvtools {
namespace opt {

Pass::Status WrapOpKill::Process() {
  bool modified = false;

  for (auto& func : *get_module()) {
    func.ForEachInst([this, &modified](Instruction* inst) {
      if (inst->opcode() == SpvOpKill) {
        modified = true;
        ReplaceWithFunctionCall(inst);
      }
    });
  }

  if (opkill_function_ != nullptr) {
    assert(modified &&
           "The function should only be generated if something was modified.");
    context()->AddFunction(std::move(opkill_function_));
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

void WrapOpKill::ReplaceWithFunctionCall(Instruction* inst) {
  assert(inst->opcode() == SpvOpKill &&
         "|inst| must be an OpKill instruction.");
  InstructionBuilder ir_builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  ir_builder.AddFunctionCall(GetVoidTypeId(), GetOpKillFuncId(), {});
  ir_builder.AddUnreachable();
  context()->KillInst(inst);
}

uint32_t WrapOpKill::GetVoidTypeId() {
  if (void_type_id_ != 0) {
    return void_type_id_;
  }

  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Void void_type;
  void_type_id_ = type_mgr->GetTypeInstruction(&void_type);
  return void_type_id_;
}

uint32_t WrapOpKill::GetVoidFunctionTypeId() {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Void void_type;
  const analysis::Type* registered_void_type =
      type_mgr->GetRegisteredType(&void_type);

  analysis::Function func_type(registered_void_type, {});
  return type_mgr->GetTypeInstruction(&func_type);
}

uint32_t WrapOpKill::GetOpKillFuncId() {
  if (opkill_function_ != nullptr) {
    return opkill_function_->result_id();
  }

  uint32_t opkill_func_id = TakeNextId();

  // Generate the function start instruction
  std::unique_ptr<Instruction> func_start(new Instruction(
      context(), SpvOpFunction, GetVoidTypeId(), opkill_func_id, {}));
  func_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  func_start->AddOperand({SPV_OPERAND_TYPE_ID, {GetVoidFunctionTypeId()}});
  opkill_function_.reset(new Function(std::move(func_start)));

  // Generate the function end instruction
  std::unique_ptr<Instruction> func_end(
      new Instruction(context(), SpvOpFunctionEnd, 0, 0, {}));
  opkill_function_->SetFunctionEnd(std::move(func_end));

  // Create the one basic block for the function.
  std::unique_ptr<Instruction> label_inst(
      new Instruction(context(), SpvOpLabel, 0, TakeNextId(), {}));
  std::unique_ptr<BasicBlock> bb(new BasicBlock(std::move(label_inst)));

  // Add the OpKill to the basic block
  std::unique_ptr<Instruction> kill_inst(
      new Instruction(context(), SpvOpKill, 0, 0, {}));
  bb->AddInstruction(std::move(kill_inst));

  // Add the bb to the function
  opkill_function_->AddBasicBlock(std::move(bb));

  // Add the function to the module.
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    opkill_function_->ForEachInst(
        [this](Instruction* inst) { context()->AnalyzeDefUse(inst); });
  }

  if (context()->AreAnalysesValid(IRContext::kAnalysisInstrToBlockMapping)) {
    for (BasicBlock& basic_block : *opkill_function_) {
      context()->set_instr_block(basic_block.GetLabelInst(), &basic_block);
      for (Instruction& inst : basic_block) {
        context()->set_instr_block(&inst, &basic_block);
      }
    }
  }

  return opkill_function_->result_id();
}

}  // namespace opt
}  // namespace spvtools
