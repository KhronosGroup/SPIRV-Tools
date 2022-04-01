// Copyright (c) 2022 Google LLC
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

#include "source/opt/var_for_func_call_pass.h"

#include "ir_builder.h"

using namespace spvtools;
using namespace opt;

Pass::Status VarsForFunctionCallPass::Process() {
  auto result = Status::SuccessWithoutChange;
  auto funcsNum = get_module()->end() - get_module()->begin();
  if (funcsNum == 1) return result;
  for (auto& func : *get_module()) {
    // Get Variable insertion point
    Instruction* varInsertPt = &*(func.begin()->begin());
    func.ForEachInst([this, &result, &varInsertPt](Instruction* inst) {
      if (inst->opcode() == SpvOpFunctionCall) {
        Instruction* nextInst = inst->NextNode();
        for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
          auto& op = inst->GetInOperand(i);
          if (op.type != SPV_OPERAND_TYPE_ID) continue;
          Instruction* op_inst = get_def_use_mgr()->GetDef(op.AsId());
          if (op_inst->opcode() == SpvOpAccessChain) {
            InstructionBuilder builder(
                context(), inst,
                IRContext::kAnalysisDefUse |
                    IRContext::kAnalysisInstrToBlockMapping);

            Instruction* op_ptr_type =
                get_def_use_mgr()->GetDef(op_inst->type_id());
            Instruction* op_type = get_def_use_mgr()->GetDef(
                op_ptr_type->GetSingleWordInOperand(1));
            auto varType = context()->get_type_mgr()->FindPointerToType(
                op_type->result_id(), SpvStorageClassFunction);
            // Create new variable
            builder.SetInsertPoint(varInsertPt);
            Instruction* var =
                builder.AddVariable(varType, SpvStorageClassFunction);
            // Load access chain to the new variable before function call
            builder.SetInsertPoint(inst);
            Instruction* load =
                builder.AddLoad(op_type->result_id(), op.AsId());
            builder.AddStore(var->result_id(), load->result_id());
            // Load return value to the acesschain after function call
            builder.SetInsertPoint(nextInst);
            load = builder.AddLoad(op_type->result_id(), var->result_id());
            builder.AddStore(op.AsId(), load->result_id());

            // Replace AccessChain with new create variable
            inst->SetInOperand(i, {var->result_id()});
            result = Status::SuccessWithChange;
          }
        }
        if (result == Status::SuccessWithChange) {
          context()->UpdateDefUse(inst);
        }
      }
    });
  }
  return result;
}