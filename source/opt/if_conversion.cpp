// Copyright (c) 2018 Google LLC
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

#include "if_conversion.h"

#include <iostream>
namespace spvtools {
namespace opt {

Pass::Status IfConversion::Process(ir::IRContext* c) {
  InitializeProcessing(c);

  std::vector<ir::Instruction*> to_kill;
  for (auto& func : *get_module()) {
    DominatorAnalysis* dominators =
        context()->GetDominatorAnalysis(&func, *context()->cfg());
    for (auto& block : func) {
      // Get an insertion point.
      auto iter = block.begin();
      while (iter != block.end() && iter->opcode() == SpvOpPhi) {
        ++iter;
      }

      block.ForEachPhiInst([this, &to_kill, dominators, &block,
                            &iter](ir::Instruction* phi) {
        std::cerr << "Examining " << *phi << std::endl;
        // TODO(alan-baker): Extend to more than two predecessors
        if (phi->NumInOperands() != 4u) return;

        if (!CheckType(phi->type_id())) return;

        ir::BasicBlock* inc0 = GetIncomingBlock(phi, 0u);
        if (dominators->Dominates(&block, inc0)) return;
        // if (!dominators->IsReachable(inc0)) return;

        ir::BasicBlock* inc1 = GetIncomingBlock(phi, 1u);
        if (dominators->Dominates(&block, inc1)) return;
        // if (!dominators->IsReachable(inc1)) return;

        ir::BasicBlock* common = CommonDominator(inc0, inc1, *dominators);
        if (!common) return;
        ir::Instruction* branch = common->terminator();
        if (branch->opcode() != SpvOpBranchConditional) return;

        uint32_t condition = branch->GetSingleWordInOperand(0u);
        ir::BasicBlock* then_block =
            GetBlock(branch->GetSingleWordInOperand(1u));
        ir::BasicBlock* else_block =
            GetBlock(branch->GetSingleWordInOperand(2u));
        std::cerr << " inc0 = " << inc0->id() << std::endl;
        std::cerr << " inc1 = " << inc1->id() << std::endl;
        std::cerr << " common = " << common->id() << std::endl;
        std::cerr << " then = " << then_block->id() << std::endl;
        std::cerr << " else = " << else_block->id() << std::endl;
        ir::Instruction* true_value = nullptr;
        ir::Instruction* false_value = nullptr;
        if ((then_block == &block && inc0 == common) ||
            dominators->Dominates(then_block, inc0)) {
          true_value = GetIncomingValue(phi, 0u);
          false_value = GetIncomingValue(phi, 1u);
        } else {
          true_value = GetIncomingValue(phi, 1u);
          false_value = GetIncomingValue(phi, 0u);
        }

        ir::BasicBlock* true_def_block = context()->get_instr_block(true_value);
        if (true_def_block && !dominators->Dominates(true_def_block, &block))
          return;

        ir::BasicBlock* false_def_block =
            context()->get_instr_block(false_value);
        if (false_def_block && !dominators->Dominates(false_def_block, &block))
          return;

        std::cerr << "Replacing " << *phi << std::endl;
        // TODO(alan-baker): re-use |phi|'s result id.
        std::unique_ptr<ir::Instruction> select(new ir::Instruction(
            context(), SpvOpSelect, phi->type_id(), TakeNextId(),
            std::initializer_list<ir::Operand>{
                {SPV_OPERAND_TYPE_ID, {condition}},
                {SPV_OPERAND_TYPE_ID, {true_value->result_id()}},
                {SPV_OPERAND_TYPE_ID, {false_value->result_id()}}}));
        get_def_use_mgr()->AnalyzeInstDefUse(select.get());
        context()->set_instr_block(select.get(), &block);
        context()->ReplaceAllUsesWith(phi->result_id(), select->result_id());
        iter.InsertBefore(std::move(select));
        to_kill.push_back(phi);
      });
    }
  }

  for (auto inst : to_kill) {
    context()->KillInst(inst);
  }

  return Status::SuccessWithoutChange;
}

bool IfConversion::CheckType(uint32_t id) {
  ir::Instruction* type = get_def_use_mgr()->GetDef(id);
  SpvOp op = type->opcode();
  if (spvOpcodeIsScalarType(op) || op == SpvOpTypePointer ||
      op == SpvOpTypeVector)
    return true;
  return false;
}

ir::BasicBlock* IfConversion::GetBlock(uint32_t id) {
  return context()->get_instr_block(get_def_use_mgr()->GetDef(id));
}

ir::BasicBlock* IfConversion::GetIncomingBlock(ir::Instruction* phi,
                                               uint32_t predecessor) {
  uint32_t in_index = 2 * predecessor + 1;
  return GetBlock(phi->GetSingleWordInOperand(in_index));
}

ir::Instruction* IfConversion::GetIncomingValue(ir::Instruction* phi,
                                                uint32_t predecessor) {
  uint32_t in_index = 2 * predecessor;
  return get_def_use_mgr()->GetDef(phi->GetSingleWordInOperand(in_index));
}

ir::BasicBlock* IfConversion::CommonDominator(
    ir::BasicBlock* inc0, ir::BasicBlock* inc1,
    const DominatorAnalysis& dominators) {
  std::unordered_set<ir::BasicBlock*> seen;
  ir::BasicBlock* block = inc0;
  while (block && seen.insert(block).second) {
    block = dominators.ImmediateDominator(block);
  }

  block = inc1;
  while (block && seen.insert(block).second) {
    block = dominators.ImmediateDominator(block);
  }

  return block;
}

}  // namespace opt
}  // namespace spvtools
