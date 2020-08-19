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
TransformationReplaceOpSelectWithConditionalBranch::
    TransformationReplaceOpSelectWithConditionalBranch(
        const spvtools::fuzz::protobufs::
            TransformationReplaceOpSelectWithConditionalBranch& message)
    : message_(message) {}

TransformationReplaceOpSelectWithConditionalBranch::
    TransformationReplaceOpSelectWithConditionalBranch(
        uint32_t select_id, uint32_t true_block_fresh_id,
        uint32_t merge_block_fresh_id) {
  message_.set_select_id(select_id);
  message_.set_true_block_fresh_id(true_block_fresh_id);
  message_.set_merge_block_fresh_id(merge_block_fresh_id);
}

bool TransformationReplaceOpSelectWithConditionalBranch::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.select_id());

  // The instruction must exist and it must be an OpSelect instruction.
  if (!instruction || instruction->opcode() != SpvOpSelect) {
    return false;
  }

  // Check that the condition is a scalar boolean.
  auto condition = ir_context->get_def_use_mgr()->GetDef(
      instruction->GetSingleWordInOperand(0));
  if (!condition) {
    return false;
  }
  auto condition_type =
      ir_context->get_type_mgr()->GetType(condition->type_id());
  if (!condition_type || !condition_type->AsBool()) {
    return false;
  }

  auto block = ir_context->get_instr_block(instruction);
  assert(block && "The block containing the instruction must be found");

  // Check that the new block ids are fresh and distinct.
  std::set<uint32_t> used_ids;
  for (uint32_t id :
       {message_.true_block_fresh_id(), message_.merge_block_fresh_id()}) {
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

void TransformationReplaceOpSelectWithConditionalBranch::Apply(
    opt::IRContext* ir_context, TransformationContext* /* unused */) const {
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.select_id());

  auto block = ir_context->get_instr_block(instruction);

  // Update the module id bound with the new ids being used.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.true_block_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.merge_block_fresh_id());

  // Split the block before the OpSelect instruction to get what will be the
  // merge block.
  auto merge_block = block->SplitBasicBlock(
      ir_context, message_.merge_block_fresh_id(),
      fuzzerutil::GetIteratorForInstruction(block, instruction));

  // Create a new empty block.
  auto new_block = MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
      ir_context, SpvOpLabel, 0, message_.true_block_fresh_id(),
      opt::Instruction::OperandList{}));

  // Add an unconditional branch from the new block to the merge block.
  new_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpBranch, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {merge_block->id()}}}));

  // Add an OpSelectionMerge instruction to the original block.
  block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpSelectionMerge, 0, 0,
      opt::Instruction::OperandList{{SPV_OPERAND_TYPE_ID, {merge_block->id()}},
                                    {SPV_OPERAND_TYPE_SELECTION_CONTROL,
                                     {SpvSelectionControlMaskNone}}}));

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
TransformationReplaceOpSelectWithConditionalBranch::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_opselect_with_conditional_branch() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
