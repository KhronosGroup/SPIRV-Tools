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

namespace spvtools {
namespace opt {

Pass::Status IfConversion::Process(ir::IRContext* c) {
  InitializeProcessing(c);

  bool modified = false;
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

      ir::BasicBlock* common = nullptr;
      block.WhileEachPhiInst([this, &modified, &common, &to_kill, dominators,
                              &block, &iter](ir::Instruction* phi) {
        // A false return from this function stops iteration through subsequent
        // phis. If any of the cfg aspects of the phi are incompatible (e.g.
        // bad domination of incoming block), there is no point checking other
        // phi nodes. If the function returns true, then this particular phi is
        // incompatible, but subsequent phis may be compatible.

        // TODO(alan-baker): Extend to more than two predecessors
        if (phi->NumInOperands() != 4u) return false;

        // This phi is not compatible, but subsequent phis might be.
        if (!CheckType(phi->type_id())) return true;

        ir::BasicBlock* inc0 = GetIncomingBlock(phi, 0u);
        if (dominators->Dominates(&block, inc0)) return false;

        ir::BasicBlock* inc1 = GetIncomingBlock(phi, 1u);
        if (dominators->Dominates(&block, inc1)) return false;

        // All phis will have the same common dominator, so cache the result
        // for this block. If there is no common dominator, we can stop
        // traversing.
        if (!common) common = CommonDominator(inc0, inc1, *dominators);
        if (!common) return false;
        ir::Instruction* branch = common->terminator();
        if (branch->opcode() != SpvOpBranchConditional) return false;

        // We cannot transform cases where the phi is used by another phi in the
        // same block due to instruction ordering restrictions.
        // TODO(alan-baker): If all inappropriate uses could also be
        // transformed, we could still remove this phi.
        if (!CheckPhiUsers(phi, &block)) return true;

        // Identify the incoming values associated with the true and false
        // branches. If |then_block| dominates |inc0| or if the true edge
        // branches straight to this block and |common| is |inc0|, then |inc0|
        // is on the true branch. Otherwise the |inc1| is on the true branch.
        uint32_t condition = branch->GetSingleWordInOperand(0u);
        ir::BasicBlock* then_block =
            GetBlock(branch->GetSingleWordInOperand(1u));
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

        // If either incoming value is defined is a block that does not dominate
        // this phi, then we cannot eliminate the phi with a select.
        // TODO(alan-baker): Perform code motion where it makes sense to enable
        // the transform in this case.
        ir::BasicBlock* true_def_block = context()->get_instr_block(true_value);
        if (true_def_block && !dominators->Dominates(true_def_block, &block))
          return true;

        ir::BasicBlock* false_def_block =
            context()->get_instr_block(false_value);
        if (false_def_block && !dominators->Dominates(false_def_block, &block))
          return true;

        analysis::Type* data_ty =
            context()->get_type_mgr()->GetType(true_value->type_id());
        if (analysis::Vector* vec_data_ty = data_ty->AsVector()) {
          condition = SplatCondition(vec_data_ty, &block, iter, condition);
        }

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

        return true;
      });
    }
  }

  for (auto inst : to_kill) {
    context()->KillInst(inst);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool IfConversion::CheckPhiUsers(ir::Instruction* phi, ir::BasicBlock* block) {
  return get_def_use_mgr()->WhileEachUser(phi, [block,
                                                this](ir::Instruction* user) {
    if (user->opcode() == SpvOpPhi && context()->get_instr_block(user) == block)
      return false;
    return true;
  });
}

uint32_t IfConversion::SplatCondition(analysis::Vector* vec_data_ty,
                                      ir::BasicBlock* block,
                                      ir::BasicBlock::iterator where,
                                      uint32_t cond) {
  // If the data inputs to OpSelect are vectors, the condition for
  // OpSelect must be a boolean vector with the same number of
  // components. So splat the condition for the branch into a vector
  // type.
  analysis::Bool bool_ty;
  analysis::Vector bool_vec_ty(&bool_ty, vec_data_ty->element_count());
  uint32_t bool_vec_id =
      context()->get_type_mgr()->GetTypeInstruction(&bool_vec_ty);
  std::vector<ir::Operand> ops;
  for (size_t i = 0; i != vec_data_ty->element_count(); ++i) {
    ops.emplace_back(SPV_OPERAND_TYPE_ID,
                     std::initializer_list<uint32_t>{cond});
  }
  uint32_t splat_id = TakeNextId();
  std::unique_ptr<ir::Instruction> splat(new ir::Instruction(
      context(), SpvOpCompositeConstruct, bool_vec_id, splat_id, ops));
  context()->set_instr_block(splat.get(), block);
  get_def_use_mgr()->AnalyzeInstDefUse(splat.get());
  where.InsertBefore(std::move(splat));
  return splat_id;
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
