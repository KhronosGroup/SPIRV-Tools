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

#include "source/fuzz/transformation_split_loop.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationSplitLoop::TransformationSplitLoop(
    const spvtools::fuzz::protobufs::TransformationSplitLoop& message)
    : message_(message) {}

TransformationSplitLoop::TransformationSplitLoop(
    uint32_t loop_header_id, uint32_t variable_counter_fresh_id,
    uint32_t variable_run_second_fresh_id, uint32_t constant_limit_id,
    uint32_t load_counter_fresh_id, uint32_t increment_counter_fresh_id,
    uint32_t condition_counter_fresh_id, uint32_t new_body_entry_block_fresh_id,
    uint32_t conditional_block_fresh_id, uint32_t load_run_second_fresh_id,
    uint32_t selection_merge_block_fresh_id,
    const std::vector<uint32_t>& logical_not_fresh_ids,
    const std::map<uint32_t, uint32_t>& original_label_to_duplicate_label,
    const std::map<uint32_t, uint32_t>& original_id_to_duplicate_id) {
  message_.set_loop_header_id(loop_header_id);
  message_.set_variable_counter_fresh_id(variable_counter_fresh_id);
  message_.set_variable_run_second_fresh_id(variable_run_second_fresh_id);
  message_.set_constant_limit_id(constant_limit_id);
  message_.set_load_counter_fresh_id(load_counter_fresh_id);
  message_.set_increment_counter_fresh_id(increment_counter_fresh_id);
  message_.set_condition_counter_fresh_id(condition_counter_fresh_id);
  message_.set_new_body_entry_block_fresh_id(new_body_entry_block_fresh_id);
  message_.set_conditional_block_fresh_id(conditional_block_fresh_id);
  message_.set_load_run_second_fresh_id(load_run_second_fresh_id);
  message_.set_selection_merge_block_fresh_id(selection_merge_block_fresh_id);
  for (auto logical_not_fresh_id : logical_not_fresh_ids) {
    message_.add_logical_not_fresh_ids(logical_not_fresh_id);
  }
  *message_.mutable_original_label_to_duplicate_label() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_label_to_duplicate_label);
  *message_.mutable_original_id_to_duplicate_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_id_to_duplicate_id);
}

bool TransformationSplitLoop::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Various ids used by this transformation must be fresh and distinct.
  std::set<uint32_t> ids_used_by_this_transformation;
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.variable_counter_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.variable_run_second_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.load_counter_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.increment_counter_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.condition_counter_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_body_entry_block_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.conditional_block_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.load_run_second_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.selection_merge_block_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  // |constant_limit_id| must refer to a value of type integer.
  auto constant_limit_instr =
      ir_context->get_def_use_mgr()->GetDef(message_.constant_limit_id());
  if (constant_limit_instr == nullptr || !constant_limit_instr->type_id()) {
    return false;
  }
  if (!ir_context->get_type_mgr()
           ->GetType(constant_limit_instr->type_id())
           ->AsInteger()) {
    return false;
  }

  // |loop_header_block| must refer to the loop header.
  auto loop_header_instr =
      ir_context->get_def_use_mgr()->GetDef(message_.loop_header_id());
  if (!loop_header_instr || loop_header_instr->opcode() != SpvOpLabel) {
    return false;
  }
  auto loop_header_block =
      fuzzerutil::MaybeFindBlock(ir_context, message_.loop_header_id());
  if (!loop_header_block || !loop_header_block->IsLoopHeader()) {
    return false;
  }

  // The header of the loop cannot be the first block.
  auto enclosing_function = loop_header_block->GetParent();
  assert(&*enclosing_function->begin() != loop_header_block &&
         "A loop header cannot be an entry block of a function.");

  auto continue_id = loop_header_block->ContinueBlockId();
  auto entry_block_pred_ids = ir_context->cfg()->preds(loop_header_block->id());
  std::sort(entry_block_pred_ids.begin(), entry_block_pred_ids.end());
  entry_block_pred_ids.erase(
      std::unique(entry_block_pred_ids.begin(), entry_block_pred_ids.end()),
      entry_block_pred_ids.end());

  // We remove a back edge from the continue block.
  entry_block_pred_ids.erase(
      std::remove(entry_block_pred_ids.begin(), entry_block_pred_ids.end(),
                  continue_id),
      entry_block_pred_ids.end());

  // Because the duplicated loop will have only one predecessor, to make
  // resolving OpPhi instructions easier, we require that the entry block has
  // only one predecessor.
  if (entry_block_pred_ids.size() > 1) {
    return false;
  }

  // We expect the merge block to end with OpBranch, OpReturn, OpReturnValue, so
  // that the region is single-exit.
  auto merge_block =
      ir_context->cfg()->block(loop_header_block->MergeBlockId());
  switch (merge_block->terminator()->opcode()) {
    case SpvOpBranch:
    case SpvOpReturn:
    case SpvOpReturnValue:
      break;
    default:
      return false;
  }
  // We don't allow OpPhi instructions in merge block, since the
  // |new_body_entry_block| will branch to the merge block and resolving these
  // OpPhi instructions can be complicated.
  for (const auto& instr : *merge_block) {
    if (instr.opcode() == SpvOpPhi) {
      return false;
    }
  }

  std::map<uint32_t, uint32_t> original_label_to_duplicate_label =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_label_to_duplicate_label());

  std::map<uint32_t, uint32_t> original_id_to_duplicate_id =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_id_to_duplicate_id());

  std::vector<uint32_t> logical_not_fresh_ids =
      fuzzerutil::RepeatedFieldToVector(message_.logical_not_fresh_ids());

  for (auto logical_not_fresh_id : logical_not_fresh_ids) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            logical_not_fresh_id, ir_context,
            &ids_used_by_this_transformation)) {
      return false;
    }
  }

  auto region_set = TransformationSplitLoop::GetRegionBlocks(
      ir_context, loop_header_block, merge_block);

  uint32_t needed_logical_not_fresh_ids = 0;
  for (auto block : region_set) {
    // If we have a terminator of form: "OpBranchConditional %cond %merge
    // %other" then we need a fresh id for OpLogicalNot instruction.
    if (block->terminator()->opcode() == SpvOpBranchConditional &&
        block->terminator()->GetSingleWordInOperand(1) == merge_block->id()) {
      needed_logical_not_fresh_ids++;
    }
    auto label =
        ir_context->get_def_use_mgr()->GetDef(block->id())->result_id();
    // The label of every block in the region must be present in the map
    // |original_label_to_duplicate_label|.
    if (original_label_to_duplicate_label.count(label) == 0) {
      return false;
    }
    auto duplicate_label = original_label_to_duplicate_label[label];
    // Each id assigned to labels in the region must be distinct and fresh.
    if (!duplicate_label ||
        !CheckIdIsFreshAndNotUsedByThisTransformation(
            duplicate_label, ir_context, &ids_used_by_this_transformation)) {
      return false;
    }
    for (const auto& instr : *block) {
      if (!instr.HasResultId()) {
        continue;
      }
      // Every instruction with a result id in the region must be present in the
      // map |original_id_to_duplicate_id|.
      if (original_id_to_duplicate_id.count(instr.result_id()) == 0) {
        return false;
      }
      auto duplicate_id = original_id_to_duplicate_id[instr.result_id()];
      // Id assigned to this result id in the region must be distinct and fresh.
      if (!duplicate_id ||
          !CheckIdIsFreshAndNotUsedByThisTransformation(
              duplicate_id, ir_context, &ids_used_by_this_transformation)) {
        return false;
      }
    }
  }
  // There is not enough fresh ids provided in the protobuf.
  if (needed_logical_not_fresh_ids > logical_not_fresh_ids.size()) {
    return false;
  }
  // Check if all required constants are present.
  auto const_zero_id = fuzzerutil::MaybeGetIntegerConstant(
      ir_context, transformation_context, {0}, 32, false, false);
  auto const_one_id = fuzzerutil::MaybeGetIntegerConstant(
      ir_context, transformation_context, {1}, 32, false, false);
  auto const_false_id = fuzzerutil::MaybeGetBoolConstant(
      ir_context, transformation_context, false, false);
  auto const_true_id = fuzzerutil::MaybeGetBoolConstant(
      ir_context, transformation_context, true, false);
  if (!const_zero_id || !const_one_id || !const_false_id || !const_true_id) {
    return false;
  }

  return true;
}

void TransformationSplitLoop::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto const_zero_id = fuzzerutil::MaybeGetIntegerConstant(
      ir_context, *transformation_context, {0}, 32, false, false);
  auto const_one_id = fuzzerutil::MaybeGetIntegerConstant(
      ir_context, *transformation_context, {1}, 32, false, false);
  auto const_false_id = fuzzerutil::MaybeGetBoolConstant(
      ir_context, *transformation_context, false, false);
  auto const_true_id = fuzzerutil::MaybeGetBoolConstant(
      ir_context, *transformation_context, true, false);

  auto loop_header_block = ir_context->cfg()->block(message_.loop_header_id());
  auto enclosing_function = loop_header_block->GetParent();
  auto merge_block =
      ir_context->cfg()->block(loop_header_block->MergeBlockId());

  // Create local variables: counter and run_second.
  auto run_second_pointer_type_id = fuzzerutil::MaybeGetPointerType(
      ir_context, fuzzerutil::MaybeGetBoolType(ir_context),
      SpvStorageClassFunction);
  auto run_second_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, run_second_pointer_type_id);
  fuzzerutil::AddLocalVariable(ir_context,
                               message_.variable_run_second_fresh_id(),
                               run_second_pointer_type_id,
                               enclosing_function->result_id(), const_true_id);

  auto counter_pointer_type_id = fuzzerutil::MaybeGetPointerType(
      ir_context, fuzzerutil::MaybeGetIntegerType(ir_context, 32, false),
      SpvStorageClassFunction);
  auto counter_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, counter_pointer_type_id);
  fuzzerutil::AddLocalVariable(ir_context, message_.variable_counter_fresh_id(),
                               counter_pointer_type_id,
                               enclosing_function->result_id(), const_zero_id);
  for (uint32_t id :
       {message_.load_counter_fresh_id(), message_.increment_counter_fresh_id(),
        message_.condition_counter_fresh_id(),
        message_.new_body_entry_block_fresh_id(),
        message_.conditional_block_fresh_id(),
        message_.load_run_second_fresh_id(),
        message_.selection_merge_block_fresh_id()}) {
    fuzzerutil::UpdateModuleIdBound(ir_context, id);
  }

  // Get the data from repeated field in the protobuf.
  std::map<uint32_t, uint32_t> original_label_to_duplicate_label =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_label_to_duplicate_label());

  std::map<uint32_t, uint32_t> original_id_to_duplicate_id =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_id_to_duplicate_id());

  std::vector<uint32_t> logical_not_fresh_ids =
      fuzzerutil::RepeatedFieldToVector(message_.logical_not_fresh_ids());

  // Create three new blocks |selection_merge_block|, |new_body_entry_block|,
  // |conditional_block|. They will be inserted at the end of this method.
  std::unique_ptr<opt::BasicBlock> selection_merge_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, SpvOpLabel, 0, message_.selection_merge_block_fresh_id(),
          opt::Instruction::OperandList{}));

  std::unique_ptr<opt::BasicBlock> new_body_entry_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, SpvOpLabel, 0, message_.new_body_entry_block_fresh_id(),
          opt::Instruction::OperandList{}));

  std::unique_ptr<opt::BasicBlock> conditional_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, SpvOpLabel, 0, message_.conditional_block_fresh_id(),
          opt::Instruction::OperandList{}));

  // In the terminator of |loop_header| take first branch which is not a merge
  // block and set it to |new_body_entry|, which will be the first block of the
  // body of the loop.
  auto loop_header_terminator = loop_header_block->terminator();
  auto loop_header_merge_instr = loop_header_block->GetLoopMergeInst();
  uint32_t next_not_merge_id;
  if (loop_header_terminator->opcode() == SpvOpBranch) {
    next_not_merge_id = loop_header_terminator->GetSingleWordInOperand(0);
  } else /*SpvOpBranchConditional*/ {
    assert(loop_header_terminator->opcode() == SpvOpBranchConditional &&
           "At this point the terminator of the loop header must be "
           "OpBranchConditional");
    uint32_t first_id = loop_header_terminator->GetSingleWordInOperand(1);
    uint32_t second_id = loop_header_terminator->GetSingleWordInOperand(2);
    if (first_id != merge_block->id()) {
      next_not_merge_id = first_id;
    } else {
      next_not_merge_id = second_id;
    }
  }

  // We know that the execution of the transformed region will end in
  // |selection_merge_block|. Hence, we need to resolve OpPhi instructions:
  // change all occurrences of the label id of the |merge_block| to the label id
  // of the |selection_merge_block|.
  merge_block->ForEachSuccessorLabel(
      [this, &merge_block, ir_context](uint32_t label_id) {
        auto block = ir_context->cfg()->block(label_id);
        for (auto& instr : *block) {
          if (instr.opcode() == SpvOpPhi) {
            instr.ForEachId([this, &merge_block](uint32_t* id) {
              if (*id == merge_block->id()) {
                *id = message_.selection_merge_block_fresh_id();
              }
            });
          }
        }
      });

  std::vector<opt::BasicBlock*> blocks;
  for (auto& block : *enclosing_function) {
    blocks.push_back(&block);
  }
  std::set<opt::BasicBlock*> loop_blocks =
      GetRegionBlocks(ir_context, loop_header_block, merge_block);

  opt::BasicBlock* previous_block = nullptr;
  opt::BasicBlock* duplicated_merge_block = nullptr;
  // Duplicate the loop.
  for (auto& block : blocks) {
    if (loop_blocks.count(block) == 0) {
      continue;
    }
    fuzzerutil::UpdateModuleIdBound(
        ir_context, original_label_to_duplicate_label[block->id()]);

    // Create the duplicated block.
    std::unique_ptr<opt::BasicBlock> duplicated_block =
        MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
            ir_context, SpvOpLabel, 0,
            original_label_to_duplicate_label[block->id()],
            opt::Instruction::OperandList()));

    std::vector<opt::Instruction*> instructions_to_move_block;
    for (auto& instr : *block) {
      // The terminator of |merge_block| will be moved to
      // |selection_merge_block|. It won't be duplicated, so we skip it here.
      if (block == merge_block && (instr.opcode() == SpvOpReturn ||
                                   instr.opcode() == SpvOpReturnValue ||
                                   instr.opcode() == SpvOpBranch)) {
        continue;
      }
      if (instr.result_id()) {
        fuzzerutil::UpdateModuleIdBound(
            ir_context, original_id_to_duplicate_id[instr.result_id()]);
      }
      auto cloned_instr = instr.Clone(ir_context);
      duplicated_block->AddInstruction(
          std::unique_ptr<opt::Instruction>(cloned_instr));
      // If an id from the original region was used in this instruction,
      // replace it with the value from |original_id_to_duplicate_id|.
      // If a label from the original region was used in this instruction,
      // replace it with the value from |original_label_to_duplicate_label|.
      cloned_instr->ForEachId(
          [original_id_to_duplicate_id,
           original_label_to_duplicate_label](uint32_t* op) {
            if (original_id_to_duplicate_id.count(*op) != 0) {
              *op = original_id_to_duplicate_id.at(*op);
            }
            if (original_label_to_duplicate_label.count(*op) != 0) {
              *op = original_label_to_duplicate_label.at(*op);
            }
          });
      // Resolve OpPhi instruction in the first duplicated block. If there was a
      // branch from outside of the loop, change the id to the id of
      // |conditional_block|.
      if (!previous_block && cloned_instr->opcode() == SpvOpPhi) {
        for (uint32_t i = 1; i < cloned_instr->NumInOperands(); i += 2) {
          auto duplicated_continue_id =
              original_label_to_duplicate_label[loop_header_block
                                                    ->ContinueBlockId()];
          if (cloned_instr->GetSingleWordInOperand(i) !=
              duplicated_continue_id) {
            cloned_instr->SetInOperand(i, {conditional_block->id()});
          }
        }
      }
    }

    // If the block is the first duplicated block, insert it after the exit
    // block of the original region. Otherwise, insert it after the preceding
    // one.
    auto duplicated_block_ptr = duplicated_block.get();
    if (previous_block) {
      enclosing_function->InsertBasicBlockAfter(std::move(duplicated_block),
                                                previous_block);
    } else {
      enclosing_function->InsertBasicBlockAfter(std::move(duplicated_block),
                                                merge_block);
    }
    previous_block = duplicated_block_ptr;
    // After execution of the loop, this variable stores a pointer to the last
    // duplicated block.
    if (block == merge_block) {
      duplicated_merge_block = duplicated_block_ptr;
    }
  }

  // Move the terminator of |merge_block| to the end of
  // |selection_merge_block|
  {
    auto instr = merge_block->terminator();
    auto cloned_instr = instr->Clone(ir_context);
    selection_merge_block->AddInstruction(
        std::unique_ptr<opt::Instruction>(cloned_instr));
    ir_context->KillInst(instr);
  }

  // |merge_block| of the original loop will now branch to
  // |conditional_block|.
  merge_block->AddInstruction(MakeUnique<opt::Instruction>(opt::Instruction(
      ir_context, SpvOpBranch, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.conditional_block_fresh_id()}}}))));

  // Add instructions to |conditional_block|. If |run_second| is true then go to
  // the duplicated loop header. Else, go to |selection_merge_block|.
  conditional_block->AddInstruction(MakeUnique<opt::Instruction>(
      opt::Instruction(ir_context, SpvOpLoad, run_second_type_id,
                       message_.load_run_second_fresh_id(),
                       opt::Instruction::OperandList(
                           {{SPV_OPERAND_TYPE_ID,
                             {message_.variable_run_second_fresh_id()}}}))));

  conditional_block->AddInstruction(MakeUnique<opt::Instruction>(
      opt::Instruction(ir_context, SpvOpSelectionMerge, 0, 0,
                       {{SPV_OPERAND_TYPE_ID, {selection_merge_block->id()}},
                        {SPV_OPERAND_TYPE_SELECTION_CONTROL,
                         {SpvSelectionControlMaskNone}}})));

  conditional_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpBranchConditional, 0, 0,
          {{SPV_OPERAND_TYPE_ID, {message_.load_run_second_fresh_id()}},
           {SPV_OPERAND_TYPE_ID,
            {original_label_to_duplicate_label[loop_header_block->id()]}},
           {SPV_OPERAND_TYPE_ID, {selection_merge_block->id()}}})));

  // |duplicated_merge_block| of the duplicated loop will now branch to
  // |selection_merge_block|.
  duplicated_merge_block->AddInstruction(MakeUnique<opt::Instruction>(
      opt::Instruction(ir_context, SpvOpBranch, 0, 0,
                       opt::Instruction::OperandList(
                           {{SPV_OPERAND_TYPE_ID,
                             {message_.selection_merge_block_fresh_id()}}}))));

  // Move all instructions except OpLabel, OpLoopMerge and OpPhi to the
  // |new_body_entry_block|. The terminator, which is OpBranch or
  // OpBranchConditional is also moved. If there any of these instructions that
  // cause side effects (like OpStore), avoid executing them incorrect number of
  // times due to |counter| reaching its iteration limit.
  std::vector<opt::Instruction> instructions_in_block;

  for (auto iter = loop_header_block->begin();
       iter != loop_header_block->end();) {
    if (iter->opcode() == SpvOpLabel || iter->opcode() == SpvOpLoopMerge ||
        iter->opcode() == SpvOpPhi) {
      ++iter;
    } else {
      new_body_entry_block->AddInstruction(
          std::unique_ptr<opt::Instruction>(iter->Clone(ir_context)));
      auto new_iter = ir_context->KillInst(&*iter);
      if (!new_iter) break;
      iter = new_iter;
    }
  }

  // Add incrementation instructions to |loop_header_block|. At this point, the
  // terminator is an OpLoopMerge instruction.
  loop_header_block->terminator()->InsertBefore(MakeUnique<opt::Instruction>(
      opt::Instruction(ir_context, SpvOpLoad, counter_type_id,
                       message_.load_counter_fresh_id(),
                       opt::Instruction::OperandList(
                           {{SPV_OPERAND_TYPE_ID,
                             {message_.variable_counter_fresh_id()}}}))));

  loop_header_block->terminator()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpIAdd, counter_type_id,
          message_.increment_counter_fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.load_counter_fresh_id()}},
               {SPV_OPERAND_TYPE_ID, {const_one_id}}}))));

  loop_header_block->terminator()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_counter_fresh_id()}},
               {SPV_OPERAND_TYPE_ID,
                {message_.increment_counter_fresh_id()}}}))));

  auto bool_type_id = fuzzerutil::MaybeGetBoolType(ir_context);
  assert(bool_type_id && "The bool type must be present.");

  loop_header_block->terminator()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpULessThan, bool_type_id,
          message_.condition_counter_fresh_id(),
          {{SPV_OPERAND_TYPE_ID, {message_.increment_counter_fresh_id()}},
           {SPV_OPERAND_TYPE_ID, {message_.constant_limit_id()}}})));

  // Insert conditional branch at the end of |loop_header|. If the number of
  // iterations is smaller than the limit, go to |new_body_entry|. Otherwise,
  // we have finished the iteration in the first loop, therefore go to
  // |merge_block|.
  loop_header_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpBranchConditional, 0, 0,
          {{SPV_OPERAND_TYPE_ID, {message_.condition_counter_fresh_id()}},
           {SPV_OPERAND_TYPE_ID, {message_.new_body_entry_block_fresh_id()}},
           {SPV_OPERAND_TYPE_ID, {merge_block->id()}}})));

  // Set |run_second| for every branch to the merge block.
  auto merge_block_instr = merge_block->GetLabelInst();
  ir_context->get_def_use_mgr()->ForEachUse(
      merge_block_instr,
      [this, const_false_id, ir_context, loop_blocks, &logical_not_fresh_ids,
       bool_type_id](opt::Instruction* user, uint32_t operand) {
        auto user_block = ir_context->get_instr_block(user);
        if (loop_blocks.find(user_block) == loop_blocks.end()) {
          return;
        }
        // If we branch unconditionally to the merge block, set |run_second|
        // to false.
        if (user->opcode() == SpvOpBranch) {
          user->InsertBefore(MakeUnique<opt::Instruction>(opt::Instruction(
              ir_context, SpvOpStore, 0, 0,
              opt::Instruction::OperandList(
                  {{SPV_OPERAND_TYPE_ID,
                    {message_.variable_run_second_fresh_id()}},
                   {SPV_OPERAND_TYPE_ID, {const_false_id}}}))));
        } else if (user->opcode() == SpvOpBranchConditional) {
          auto instr_to_insert_before = user;
          // In the header block, insert before the OpLoopMerge instruction.
          if (user_block->IsLoopHeader()) {
            instr_to_insert_before = user->PreviousNode();
          }
          if (operand == 2) {
            // If we have an instruction of form: "OpBranchConditional %cond
            // %other %merge" then set |run_second| to the value of the
            // boolean condition %cond
            instr_to_insert_before->InsertBefore(
                MakeUnique<opt::Instruction>(opt::Instruction(
                    ir_context, SpvOpStore, 0, 0,
                    opt::Instruction::OperandList(
                        {{SPV_OPERAND_TYPE_ID,
                          {message_.variable_run_second_fresh_id()}},
                         {SPV_OPERAND_TYPE_ID,
                          {user->GetSingleWordInOperand(0)}}}))));
          } else if (operand == 1) {
            // If we have an instruction of form: "OpBranchConditional %cond
            // %merge %other" then set |run_second| to the negation of the
            // value of the boolean condition %cond.
            auto result_id = logical_not_fresh_ids.back();
            logical_not_fresh_ids.pop_back();
            instr_to_insert_before->InsertBefore(
                MakeUnique<opt::Instruction>(opt::Instruction(
                    ir_context, SpvOpLogicalNot, bool_type_id, result_id,
                    opt::Instruction::OperandList(
                        {{SPV_OPERAND_TYPE_ID,
                          {user->GetSingleWordInOperand(0)}}}))));
            instr_to_insert_before->InsertBefore(
                MakeUnique<opt::Instruction>(opt::Instruction(
                    ir_context, SpvOpStore, 0, 0,
                    opt::Instruction::OperandList(
                        {{SPV_OPERAND_TYPE_ID,
                          {message_.variable_run_second_fresh_id()}},
                         {SPV_OPERAND_TYPE_ID, {result_id}}}))));
          }
        }
      });

  // Insert newly created blocks.
  enclosing_function->InsertBasicBlockAfter(std::move(new_body_entry_block),
                                            loop_header_block);
  enclosing_function->InsertBasicBlockAfter(std::move(selection_merge_block),
                                            duplicated_merge_block);
  enclosing_function->InsertBasicBlockAfter(std::move(conditional_block),
                                            merge_block);

  // If the |loop_header_id| was used in the OpLoopMerge instruction as a
  // continue target, change it to |new_body_entry_block_id|.
  auto loop_header_merge_instr_continue_target =
      loop_header_merge_instr->GetSingleWordInOperand(1);
  if (loop_header_merge_instr_continue_target == message_.loop_header_id()) {
    loop_header_merge_instr->SetInOperand(
        1, {message_.new_body_entry_block_fresh_id()});
  }

  // In the original loop, if there are OpPhi instructions referring to the
  // loop header, change these ids to |new_body_entry_block_id|, since
  // |new_body_entry_block| was inserted after the loop header.
  {
    auto block = ir_context->cfg()->block(next_not_merge_id);
    for (auto& instr : *block) {
      if (instr.opcode() == SpvOpPhi) {
        for (uint32_t i = 1; i < instr.NumInOperands(); i += 2) {
          if (instr.GetSingleWordInOperand(i) == message_.loop_header_id()) {
            instr.SetInOperand(i, {message_.new_body_entry_block_fresh_id()});
          }
        }
      }
    }
  }
  // Since we have changed the module, the analyses are now invalid.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationSplitLoop::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_split_loop() = message_;
  return result;
}

// TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3785):
//     The following code has been copied from TransformationOutlineFunction.
//     Consider refactoring to avoid duplication.

std::set<opt::BasicBlock*> TransformationSplitLoop::GetRegionBlocks(
    opt::IRContext* ir_context, opt::BasicBlock* entry_block,
    opt::BasicBlock* exit_block) {
  auto enclosing_function = entry_block->GetParent();
  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(enclosing_function);
  auto postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(enclosing_function);

  // A block belongs to a region between the entry block and the exit block if
  // and only if it is dominated by the entry block and post-dominated by the
  // exit block.
  std::set<opt::BasicBlock*> result;
  for (auto& block : *enclosing_function) {
    if (dominator_analysis->Dominates(entry_block, &block) &&
        postdominator_analysis->Dominates(exit_block, &block)) {
      result.insert(&block);
    }
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
