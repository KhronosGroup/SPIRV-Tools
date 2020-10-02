// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_replace_branch_from_dead_block_with_exit.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceBranchFromDeadBlockWithExit::TransformationReplaceBranchFromDeadBlockWithExit(
    const spvtools::fuzz::protobufs::TransformationReplaceBranchFromDeadBlockWithExit& message)
    : message_(message) {}

TransformationReplaceBranchFromDeadBlockWithExit::TransformationReplaceBranchFromDeadBlockWithExit(uint32_t block_id,
                                                                                                   SpvOp opcode,
                                                                                                   uint32_t return_value_id) {
  message_.set_block_id(block_id);
  message_.set_opcode(opcode);
  message_.set_return_value_id(return_value_id);
}

bool TransformationReplaceBranchFromDeadBlockWithExit::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  auto block = ir_context->get_instr_block(message_.block_id());
  if (!block) {
    return false;
  }
  if (!transformation_context.GetFactManager()->BlockIsDead(block->id())) {
    return false;
  }
  if (block->terminator()->opcode() != SpvOpBranch) {
    return false;
  }
  if (ir_context->GetStructuredCFGAnalysis()->IsInContinueConstruct(block->id())) {
    return false;
  }
  auto successor = ir_context->get_instr_block(block->terminator()->GetSingleWordInOperand(0));
  auto predecessors_of_successor = ir_context->cfg()->preds(successor->id());
  std::set<uint32_t> unique_predecessors_of_successor(predecessors_of_successor.begin(), predecessors_of_successor.end());
  if (unique_predecessors_of_successor.size() < 2) {
    return false;
  }
  return true;
}

void TransformationReplaceBranchFromDeadBlockWithExit::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto terminator = ir_context->get_instr_block(message_.block_id())->terminator();
  terminator->SetOpcode(static_cast<SpvOp>(message_.opcode()));
  opt::Instruction::OperandList operands;
  if (message_.opcode() == SpvOpReturnValue) {
    operands.push_back({ SPV_OPERAND_TYPE_TYPE_ID, { message_.return_value_id()}});
  }
  terminator->SetInOperands(std::move(operands));
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

std::unordered_set<uint32_t> TransformationReplaceBranchFromDeadBlockWithExit::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

protobufs::Transformation TransformationReplaceBranchFromDeadBlockWithExit::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_branch_from_dead_block_with_exit() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
