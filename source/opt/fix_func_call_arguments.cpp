// Copyright (c) 2022 AMD LLC
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

#include "fix_func_call_arguments.h"

#include "ir_builder.h"

using namespace spvtools;
using namespace opt;

bool FixFuncCallArgumentsPass::HasNoFunctionToCall() {
  auto funcsNum = get_module()->end() - get_module()->begin();
  return funcsNum == 1;
}

Pass::Status FixFuncCallArgumentsPass::Process() {
  bool modified = false;
  if (HasNoFunctionToCall()) return Status::SuccessWithoutChange;
  for (auto& func : *get_module()) {
    // Get Variable insertion point
    Instruction* varInsertPt = &*(func.begin()->begin());
    func.ForEachInst([this, &modified, &varInsertPt](Instruction* inst) {
      if (inst->opcode() == SpvOpFunctionCall) {
        modified |= FixFuncCallArguments(inst, varInsertPt);
      }
    });
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool FixFuncCallArgumentsPass::FixFuncCallArguments(Instruction* func_call_inst,
                                                    Instruction* var_insertPt) {
  bool modified = false;
  Instruction* nextInst = func_call_inst->NextNode();
  for (uint32_t i = 0; i < func_call_inst->NumInOperands(); ++i) {
    Operand& op = func_call_inst->GetInOperand(i);
    if (op.type != SPV_OPERAND_TYPE_ID) continue;
    Instruction* operand_inst = get_def_use_mgr()->GetDef(op.AsId());
    if (operand_inst->opcode() == SpvOpAccessChain) {
      ReplaceAccessChainFuncCallArguments(func_call_inst, &op, operand_inst,
                                          nextInst, var_insertPt, i);
      modified = true;
    }
  }
  if (modified) {
    context()->UpdateDefUse(func_call_inst);
  }
  return modified;
}

void FixFuncCallArgumentsPass::ReplaceAccessChainFuncCallArguments(
    Instruction* func_call_inst, Operand* operand, Instruction* operand_inst,
    Instruction* next_insertPt, Instruction* var_insertPt,
    unsigned operand_index) {
  InstructionBuilder builder(
      context(), func_call_inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  Instruction* op_ptr_type = get_def_use_mgr()->GetDef(operand_inst->type_id());
  Instruction* op_type =
      get_def_use_mgr()->GetDef(op_ptr_type->GetSingleWordInOperand(1));
  uint32_t varType = context()->get_type_mgr()->FindPointerToType(
      op_type->result_id(), SpvStorageClassFunction);
  // Create new variable
  builder.SetInsertPoint(var_insertPt);
  Instruction* var = builder.AddVariable(varType, SpvStorageClassFunction);
  // Load access chain to the new variable before function call
  builder.SetInsertPoint(func_call_inst);
  Instruction* load = builder.AddLoad(op_type->result_id(), operand->AsId());
  builder.AddStore(var->result_id(), load->result_id());
  // Load return value to the acesschain after function call
  builder.SetInsertPoint(next_insertPt);
  load = builder.AddLoad(op_type->result_id(), var->result_id());
  builder.AddStore(operand->AsId(), load->result_id());

  // Replace AccessChain with new create variable
  func_call_inst->SetInOperand(operand_index, {var->result_id()});
}
