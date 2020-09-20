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
    uint32_t loop_header_id, uint32_t variable_counter_id,
    uint32_t variable_run_second_id, uint32_t constant_limit_id,
    uint32_t load_counter_fresh_id, uint32_t increment_counter_fresh_id,
    uint32_t condition_counter_fresh_id, uint32_t new_body_entry_block_fresh_id,
    uint32_t conditional_block_fresh_id, uint32_t load_run_second_fresh_id,
    uint32_t selection_merge_block_fresh_id,
    const std::vector<uint32_t>& logical_not_fresh_ids,
    const std::map<uint32_t, uint32_t>& original_label_to_duplicate_label,
    const std::map<uint32_t, uint32_t>& original_id_to_duplicate_id) {
  message_.set_loop_header_id(loop_header_id);
  message_.set_variable_counter_id(variable_counter_id);
  message_.set_variable_run_second_id(variable_run_second_id);
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
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  std::set<uint32_t> ids_used_by_this_transformation;

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
          message_.new_body_entry_block_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  auto loop_header_block = ir_context->cfg()->block(message_.loop_header_id());
  if (!loop_header_block->IsLoopHeader()) {
    return false;
  }

  auto continue_id = loop_header_block->ContinueBlockId();
  auto entry_block_pred_ids = ir_context->cfg()->preds(loop_header_block->id());
  std::sort(entry_block_pred_ids.begin(), entry_block_pred_ids.end());

  entry_block_pred_ids.erase(
      unique(entry_block_pred_ids.begin(), entry_block_pred_ids.end()),
      entry_block_pred_ids.end());

  entry_block_pred_ids.erase(
      std::remove(entry_block_pred_ids.begin(), entry_block_pred_ids.end(),
                  continue_id),
      entry_block_pred_ids.end());
  if (entry_block_pred_ids.size() > 1) {
    return false;
  }

  auto merge_block =
      ir_context->cfg()->block(loop_header_block->MergeBlockId());
  switch (merge_block->terminator()->opcode()) {
    case SpvOpBranch:
    case SpvOpReturn:
    case SpvOpReturnValue:
      return true;
    default:
      return false;
  }
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

  for (uint32_t id :
       {message_.load_counter_fresh_id(), message_.increment_counter_fresh_id(),
        message_.condition_counter_fresh_id(),
        message_.new_body_entry_block_fresh_id(),
        message_.conditional_block_fresh_id(),
        message_.load_run_second_fresh_id(),
        message_.selection_merge_block_fresh_id()}) {
    fuzzerutil::UpdateModuleIdBound(ir_context, id);
  }

  auto loop_header_block = ir_context->cfg()->block(message_.loop_header_id());
  auto merge_block =
      ir_context->cfg()->block(loop_header_block->MergeBlockId());
  auto enclosing_function = loop_header_block->GetParent();

  // .......................................................
  // 1. Duplicate the loop
  // .......................................................

  std::map<uint32_t, uint32_t> original_label_to_duplicate_label =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_label_to_duplicate_label());

  std::map<uint32_t, uint32_t> original_id_to_duplicate_id =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_id_to_duplicate_id());

  std::vector<uint32_t> logical_not_fresh_ids =
      fuzzerutil::RepeatedFieldToVector(message_.logical_not_fresh_ids());

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

  // We know that the execution of the transformed region will end in
  // |selection_merge_block|. Hence, we need to change all occurrences of the
  // label id of the |merge_block| to the label id of the
  // |selection_merge_block|.
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

  auto exited_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, ir_context->get_def_use_mgr()
                      ->GetDef(message_.variable_run_second_id())
                      ->type_id());

  std::vector<opt::BasicBlock*> blocks;
  for (auto& block : *enclosing_function) {
    blocks.push_back(&block);
  }
  std::set<opt::BasicBlock*> loop_blocks =
      GetRegionBlocks(ir_context, loop_header_block, merge_block);

  std::vector<opt::Instruction*> instructions_to_move;
  opt::BasicBlock* previous_block = nullptr;
  opt::BasicBlock* duplicated_merge_block = nullptr;
  for (auto& block : blocks) {
    if (loop_blocks.count(block) == 0) {
      continue;
    }
    fuzzerutil::UpdateModuleIdBound(
        ir_context, original_label_to_duplicate_label[block->id()]);

    std::unique_ptr<opt::BasicBlock> duplicated_block =
        MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
            ir_context, SpvOpLabel, 0,
            original_label_to_duplicate_label[block->id()],
            opt::Instruction::OperandList()));

    std::vector<opt::Instruction*> instructions_to_move_block;
    for (auto& instr : *block) {
      if (block == merge_block && (instr.opcode() == SpvOpReturn ||
                                   instr.opcode() == SpvOpReturnValue ||
                                   instr.opcode() == SpvOpBranch)) {
        instructions_to_move.push_back(&instr);
        continue;
      }
      if (instr.result_id()) {
        fuzzerutil::UpdateModuleIdBound(
            ir_context, original_id_to_duplicate_id[instr.result_id()]);
      }
      auto cloned_instr = instr.Clone(ir_context);
      duplicated_block->AddInstruction(
          std::unique_ptr<opt::Instruction>(cloned_instr));
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

    // Duplicate the instruction.
    // auto cloned_instr = instr.Clone(ir_context);
    // duplicated_block->AddInstruction(
    //    std::unique_ptr<opt::Instruction>(cloned_instr));

    // If an id from the original region was used in this instruction,
    // replace it with the value from |original_id_to_duplicate_id|.
    // If a label from the original region was used in this instruction,
    // replace it with the value from |original_label_to_duplicate_label|.

    // If the block is the first duplicated block, insert it before the exit
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

  for (auto instr : instructions_to_move) {
    auto cloned_instr = instr->Clone(ir_context);
    selection_merge_block->AddInstruction(
        std::unique_ptr<opt::Instruction>(cloned_instr));
    ir_context->KillInst(instr);
  }

  merge_block->AddInstruction(MakeUnique<opt::Instruction>(opt::Instruction(
      ir_context, SpvOpBranch, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.conditional_block_fresh_id()}}}))));

  conditional_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpLoad, exited_type_id,
          message_.load_run_second_fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_run_second_id()}}}))));

  conditional_block->AddInstruction(MakeUnique<opt::Instruction>(
      opt::Instruction(ir_context, SpvOpSelectionMerge, 0, 0,
                       {{SPV_OPERAND_TYPE_ID, {selection_merge_block->id()}},
                        {SPV_OPERAND_TYPE_SELECTION_CONTROL, {0}}})));

  conditional_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpBranchConditional, 0, 0,
          {{SPV_OPERAND_TYPE_ID, {message_.load_run_second_fresh_id()}},
           {SPV_OPERAND_TYPE_ID,
            {original_label_to_duplicate_label[loop_header_block->id()]}},
           {SPV_OPERAND_TYPE_ID, {selection_merge_block->id()}}})));

  duplicated_merge_block->AddInstruction(MakeUnique<opt::Instruction>(
      opt::Instruction(ir_context, SpvOpBranch, 0, 0,
                       opt::Instruction::OperandList(
                           {{SPV_OPERAND_TYPE_ID,
                             {message_.selection_merge_block_fresh_id()}}}))));

  // .......................................................
  // 2. Insert some specific instructions in the first loop.
  // .......................................................

  auto block_terminator = enclosing_function->entry()->terminator();

  enclosing_function->entry()->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_counter_id()}},
               {SPV_OPERAND_TYPE_ID, {const_zero_id}}}))));

  enclosing_function->entry()->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_run_second_id()}},
               {SPV_OPERAND_TYPE_ID, {const_true_id}}}))));

  {
    auto cloned_instr = block_terminator->Clone(ir_context);
    enclosing_function->entry()->AddInstruction(
        std::unique_ptr<opt::Instruction>(cloned_instr));
    ir_context->KillInst(block_terminator);
  }

  auto counter_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, ir_context->get_def_use_mgr()
                      ->GetDef(message_.variable_counter_id())
                      ->type_id());

  new_body_entry_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpLoad, counter_type_id,
          message_.load_counter_fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_counter_id()}}}))));

  new_body_entry_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpIAdd, counter_type_id,
          message_.increment_counter_fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.load_counter_fresh_id()}},
               {SPV_OPERAND_TYPE_ID, {const_one_id}}}))));

  new_body_entry_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_counter_id()}},
               {SPV_OPERAND_TYPE_ID,
                {message_.increment_counter_fresh_id()}}}))));

  auto bool_type_id = fuzzerutil::MaybeGetBoolType(ir_context);
  assert(bool_type_id && "The bool type must be present.");

  new_body_entry_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpULessThan, bool_type_id,
          message_.condition_counter_fresh_id(),
          {{SPV_OPERAND_TYPE_ID, {message_.increment_counter_fresh_id()}},
           {SPV_OPERAND_TYPE_ID, {message_.constant_limit_id()}}})));

  // Set |run_second| to false before every break from the loop.

  auto merge_block_instr = merge_block->GetLabelInst();
  ir_context->get_def_use_mgr()->ForEachUse(
      merge_block_instr,
      [this, const_false_id, ir_context, loop_blocks, &logical_not_fresh_ids,
       bool_type_id](opt::Instruction* user, uint32_t operand) {
        auto user_block = ir_context->get_instr_block(user);
        if (loop_blocks.find(user_block) == loop_blocks.end()) {
          return;
        }
        // If we branch unconditionally to the merge block, set |run_second| to
        // false.
        if (user->opcode() == SpvOpBranch) {
          user->InsertBefore(MakeUnique<opt::Instruction>(opt::Instruction(
              ir_context, SpvOpStore, 0, 0,
              opt::Instruction::OperandList(
                  {{SPV_OPERAND_TYPE_ID, {message_.variable_run_second_id()}},
                   {SPV_OPERAND_TYPE_ID, {const_false_id}}}))));
        } else if (user->opcode() == SpvOpBranchConditional) {
          // If we have an instruction of form: OpBranchConditional %cond %other
          // %merge then set |run_second| to the value of the boolean condition
          // %cond
          auto instr_to_insert_before = user;
          if (user_block->IsLoopHeader()) {
            instr_to_insert_before = user->PreviousNode();
          }
          if (operand == 2) {
            instr_to_insert_before->InsertBefore(MakeUnique<opt::Instruction>(
                opt::Instruction(ir_context, SpvOpStore, 0, 0,
                                 opt::Instruction::OperandList(
                                     {{SPV_OPERAND_TYPE_ID,
                                       {message_.variable_run_second_id()}},
                                      {SPV_OPERAND_TYPE_ID,
                                       {user->GetSingleWordInOperand(0)}}}))));
          } else if (operand == 1) {
            // If we have an instruction of form: OpBranchConditional %cond
            // %merge %other then set |run_second| to the negation of the value
            // of the boolean condition %cond.
            auto result_id = logical_not_fresh_ids.back();
            logical_not_fresh_ids.pop_back();
            instr_to_insert_before->InsertBefore(
                MakeUnique<opt::Instruction>(opt::Instruction(
                    ir_context, SpvOpLogicalNot, bool_type_id, result_id,
                    opt::Instruction::OperandList(
                        {{SPV_OPERAND_TYPE_ID,
                          {user->GetSingleWordInOperand(0)}}}))));
            instr_to_insert_before->InsertBefore(MakeUnique<opt::Instruction>(
                opt::Instruction(ir_context, SpvOpStore, 0, 0,
                                 opt::Instruction::OperandList(
                                     {{SPV_OPERAND_TYPE_ID,
                                       {message_.variable_run_second_id()}},
                                      {SPV_OPERAND_TYPE_ID, {result_id}}}))));
          }
        }
      });

  // In the terminator of |loop_header| take first branch which is not a merge
  // block and set it to |new_body_entry|
  auto loop_header_terminator = loop_header_block->terminator();
  uint32_t next_not_merge_id;
  uint32_t next_not_merge_pos;
  if (loop_header_terminator->opcode() == SpvOpBranch) {
    next_not_merge_id = loop_header_terminator->GetSingleWordInOperand(0);
    next_not_merge_pos = 0;
  } else  // SpvOpBranchConditional
  {
    uint32_t first_id = loop_header_terminator->GetSingleWordInOperand(1);
    uint32_t second_id = loop_header_terminator->GetSingleWordInOperand(2);
    if (first_id != merge_block->id()) {
      next_not_merge_id = first_id;
      next_not_merge_pos = 1;
    } else {
      next_not_merge_id = second_id;
      next_not_merge_pos = 2;
    }
  }

  loop_header_terminator->SetInOperand(next_not_merge_pos,
                                       {new_body_entry_block->id()});

  loop_header_block->GetLoopMergeInst()->ForEachInOperand([this](uint32_t* id) {
    if (*id == message_.loop_header_id()) {
      *id = message_.new_body_entry_block_fresh_id();
    }
  });

  // Insert conditional branch at the end of |new_body_entry|. If the number of
  // iterations is smaller than the limit, go to |next_not_merge_id|. Otherwise,
  // we have finished the iteration in the first loop, therefore go to
  // |merge_block->id()|
  new_body_entry_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpBranchConditional, 0, 0,
          {{SPV_OPERAND_TYPE_ID, {message_.condition_counter_fresh_id()}},
           {SPV_OPERAND_TYPE_ID, {next_not_merge_id}},
           {SPV_OPERAND_TYPE_ID, {merge_block->id()}}})));

  // auto new_body_entry_block_instr = new_body_entry_block->GetLabelInst();

  enclosing_function->InsertBasicBlockAfter(std::move(new_body_entry_block),
                                            loop_header_block);
  enclosing_function->InsertBasicBlockAfter(std::move(selection_merge_block),
                                            duplicated_merge_block);
  enclosing_function->InsertBasicBlockAfter(std::move(conditional_block),
                                            merge_block);

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

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

}  // namespace fuzz

protobufs::Transformation TransformationSplitLoop::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_split_loop() = message_;
  return result;
}

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
