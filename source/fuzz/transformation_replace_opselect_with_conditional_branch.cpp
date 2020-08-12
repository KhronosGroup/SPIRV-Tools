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

#include "source/fuzz/transformation_replace_opselect_with_conditional_branch.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {
TransformationReplaceOpselectWithConditionalBranch::
    TransformationReplaceOpselectWithConditionalBranch(
        const spvtools::fuzz::protobufs::
            TransformationReplaceOpselectWithConditionalBranch& message)
    : message_(message) {}

TransformationReplaceOpselectWithConditionalBranch::
    TransformationReplaceOpselectWithConditionalBranch(
        uint32_t select_id, std::pair<uint32_t, uint32_t> new_block_ids) {
  message_.set_select_id(select_id);
  protobufs::UInt32Pair pair;
  pair.set_first(new_block_ids.first);
  pair.set_second(new_block_ids.second);
  *message_.mutable_new_block_ids() = pair;
}

bool TransformationReplaceOpselectWithConditionalBranch::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.select_id());

  // The instruction must exist and it must be an OpSelect instruction.
  if (!instruction || instruction->opcode() != SpvOpSelect) {
    return false;
  }

  auto block = ir_context->get_instr_block(instruction);

  // Check that the new block ids are fresh and distinct.
  std::set<uint32_t> used_ids;
  for (uint32_t id :
       {message_.new_block_ids().first(), message_.new_block_ids().second()}) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(id, ir_context,
                                                      &used_ids)) {
      return false;
    }
  }

  // The block cannot be a loop header, since splitting it would make back edges
  // invalid.
  if (block->IsLoopHeader()) {
    return false;
  }

  // The block must be split around the OpSelect instruction. This means that
  // there cannot be an OpSampledImage instruction before OpSelect that is used
  // after it, because they are required to be in the same basic block.
  return !fuzzerutil::
      SplitBeforeInstructionSeparatesOpSampledImageDefinitionFromUse(
          block, instruction);
}

void TransformationReplaceOpselectWithConditionalBranch::Apply(
    opt::IRContext* ir_context, TransformationContext* /* unused */) const {
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.select_id());

  auto block = ir_context->get_instr_block(instruction);

  // Update the module id bound with the new ids being used.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.new_block_ids().first());
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  message_.new_block_ids().second());

  // Split the block before the OpSelect instruction to get what will be the
  // merge block.
  auto merge_block = block->SplitBasicBlock(
      ir_context, message_.new_block_ids().second(),
      fuzzerutil::GetIteratorForInstruction(block, instruction));

  // Create a new empty block.
  auto new_block = MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
      ir_context, SpvOpLabel, 0, message_.new_block_ids().first(),
      opt::Instruction::OperandList{}));

  // Add an unconditional branch from the new block to the merge block.
  new_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpBranch, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {merge_block->id()}}}));

  // Add an OpSelectionMerge instruction to the original block.
  block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpSelectionMerge, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {merge_block->id()}},
          {SPV_OPERAND_TYPE_SELECTION_CONTROL, {0}}}));

  // Add a conditional branching instruction to the original block, using the
  // same conditional as OpSelect and branching to the new block if the
  // condition is true, to the merge block otherwise.
  block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpBranchConditional, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {instruction->GetSingleWordInOperand(0)}},
          {SPV_OPERAND_TYPE_ID, {new_block->id()}},
          {SPV_OPERAND_TYPE_ID, {merge_block->id()}}}));

  // Replace the OpSelect instruction in the merge block with an OpPhi.
  // This:          OpSelect %type %cond %if %else
  // will become:   OpPhi %type %if %new_block_id %else %block_id
  instruction->SetOpcode(SpvOpPhi);
  std::vector<opt::Operand> operands;

  operands.emplace_back(instruction->GetInOperand(1));
  operands.emplace_back(opt::Operand{SPV_OPERAND_TYPE_ID, {new_block->id()}});

  operands.emplace_back(instruction->GetInOperand(2));
  operands.emplace_back(opt::Operand{SPV_OPERAND_TYPE_ID, {block->id()}});

  instruction->SetInOperands(std::move(operands));

  // Insert the new block before the merge block.
  block->GetParent()->InsertBasicBlockBefore(std::move(new_block), merge_block);
}

protobufs::Transformation
TransformationReplaceOpselectWithConditionalBranch::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_opselect_with_conditional_branch() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
