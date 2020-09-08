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
    uint32_t variable_run_second_id, uint32_t constant_one_id,
    uint32_t constant_zero_id, uint32_t constant_limit_id,
    uint32_t load_counter_fresh_id, uint32_t increment_counter_fresh_id,
    uint32_t condition_counter_fresh_id, uint32_t pred_merge_block_fresh_id,
    uint32_t constant_true_id, uint32_t constant_false_id,
    uint32_t load_run_second_fresh_id, uint32_t selection_merge_block_fresh_id,
    std::map<uint32_t, uint32_t> original_label_to_duplicate_label,
    std::map<uint32_t, uint32_t> original_id_to_duplicate_id) {
  message_.set_loop_header_id(loop_header_id);
  message_.set_variable_counter_id(variable_counter_id);
  message_.set_variable_run_second_id(variable_run_second_id);
  message_.set_constant_one_id(constant_one_id);
  message_.set_constant_zero_id(constant_zero_id);
  message_.set_constant_limit_id(constant_limit_id);
  message_.set_load_counter_fresh_id(load_counter_fresh_id);
  message_.set_increment_counter_fresh_id(increment_counter_fresh_id);
  message_.set_condition_counter_fresh_id(condition_counter_fresh_id);
  message_.set_pred_merge_block_fresh_id(pred_merge_block_fresh_id);
  message_.set_constant_true_id(constant_true_id);
  message_.set_constant_false_id(constant_false_id);
  message_.set_load_run_second_fresh_id(load_run_second_fresh_id);
  message_.set_selection_merge_block_fresh_id(selection_merge_block_fresh_id);
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
          message_.pred_merge_block_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  auto entry_block = ir_context->cfg()->block(message_.loop_header_id());
  if (!entry_block->IsLoopHeader()) {
    return false;
  }
  return true;
}

void TransformationSplitLoop::Apply(opt::IRContext* ir_context,
                                    TransformationContext* /*unused*/) const {
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.load_counter_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  message_.increment_counter_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  message_.condition_counter_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  message_.pred_merge_block_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.load_counter_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  message_.selection_merge_block_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  message_.load_run_second_fresh_id());

  std::unique_ptr<opt::BasicBlock> pred_merge_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, SpvOpLabel, 0, message_.pred_merge_block_fresh_id(),
          opt::Instruction::OperandList()));

  auto loop_header_block = ir_context->cfg()->block(message_.loop_header_id());
  auto merge_block =
      ir_context->cfg()->block(loop_header_block->MergeBlockId());
  auto enclosing_function = loop_header_block->GetParent();
  pred_merge_block->SetParent(enclosing_function);

  // .......................................................
  // 1. Duplicate the loop
  // .......................................................

  std::map<uint32_t, uint32_t> original_label_to_duplicate_label =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_label_to_duplicate_label());

  std::map<uint32_t, uint32_t> original_id_to_duplicate_id =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_id_to_duplicate_id());

  std::unique_ptr<opt::BasicBlock> selection_merge_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, SpvOpLabel, 0, message_.selection_merge_block_fresh_id(),
          opt::Instruction::OperandList()));
  selection_merge_block->SetParent(enclosing_function);

  auto exited_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, ir_context->get_def_use_mgr()
                      ->GetDef(message_.variable_run_second_id())
                      ->type_id());

  std::vector<opt::BasicBlock*> blocks;
  for (auto block_it = enclosing_function->begin();
       block_it != enclosing_function->end(); block_it++) {
    blocks.push_back(&*block_it);
  }
  std::set<opt::BasicBlock*> loop_blocks =
      GetRegionBlocks(ir_context, loop_header_block, merge_block);

  std::vector<opt::Instruction*> instructions_to_move;
  opt::BasicBlock* previous_block = nullptr;

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
    duplicated_block->SetParent(enclosing_function);
    auto duplicated_block_ptr = duplicated_block.get();
    if (previous_block) {
      enclosing_function->InsertBasicBlockAfter(std::move(duplicated_block),
                                                previous_block);
    } else {
      enclosing_function->InsertBasicBlockAfter(std::move(duplicated_block),
                                                merge_block);
    }
    previous_block = duplicated_block_ptr;
  }

  merge_block->terminator()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpLoad, exited_type_id,
          message_.load_run_second_fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_run_second_id()}}}))));

  merge_block->terminator()->InsertBefore(MakeUnique<opt::Instruction>(
      opt::Instruction(ir_context, SpvOpSelectionMerge, 0, 0,
                       {{SPV_OPERAND_TYPE_ID, {selection_merge_block->id()}},
                        {SPV_OPERAND_TYPE_SELECTION_CONTROL, {0}}})));

  merge_block->terminator()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpBranchConditional, 0, 0,
          {{SPV_OPERAND_TYPE_ID, {message_.load_run_second_fresh_id()}},
           {SPV_OPERAND_TYPE_ID,
            {original_label_to_duplicate_label[loop_header_block->id()]}},
           {SPV_OPERAND_TYPE_ID, {selection_merge_block->id()}}})));
  // After execution of the loop, this variable stores a pointer to the last
  // duplicated block.
  auto duplicated_merge_block = previous_block;

  for (auto instr : instructions_to_move) {
    auto cloned_instr = instr->Clone(ir_context);
    selection_merge_block->AddInstruction(
        std::unique_ptr<opt::Instruction>(cloned_instr));
    ir_context->KillInst(instr);
  }

  duplicated_merge_block->AddInstruction(MakeUnique<opt::Instruction>(
      opt::Instruction(ir_context, SpvOpBranch, 0, 0,
                       opt::Instruction::OperandList(
                           {{SPV_OPERAND_TYPE_ID,
                             {message_.selection_merge_block_fresh_id()}}}))));

  // .......................................................
  // 2. Insert some specific instructions in the first loop.
  // .......................................................

  enclosing_function->begin()->terminator()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_counter_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.constant_zero_id()}}}))));

  enclosing_function->begin()->terminator()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_run_second_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.constant_true_id()}}}))));

  pred_merge_block->AddInstruction(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_run_second_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.constant_false_id()}}}))));

  pred_merge_block->AddInstruction(MakeUnique<opt::Instruction>(
      opt::Instruction(ir_context, SpvOpBranch, 0, 0,
                       opt::Instruction::OperandList(
                           {{SPV_OPERAND_TYPE_ID, {merge_block->id()}}}))));

  auto condition_block = ir_context->cfg()->block(
      loop_header_block->terminator()->GetSingleWordOperand(0));
  // Replace all uses of merge label with the |pred_merge_block| label, except
  // for the block directly after the loop header.
  ir_context->get_def_use_mgr()->ForEachUse(
      merge_block->id(),
      [ir_context, this, loop_blocks, &loop_header_block, condition_block](
          opt::Instruction* user, uint32_t operand_index) {
        auto user_block = ir_context->get_instr_block(user);
        // We don't change the merge block of the function.
        if (user->opcode() == SpvOpLoopMerge) {
          return;
        }
        // If the condition of the loop is false, we don't set |exited| to true,
        // so we progress directly to the merge block.
        if (user->opcode() == SpvOpBranchConditional &&
            user_block == condition_block) {
          return;
        }
        if ((loop_blocks.find(user_block) == loop_blocks.end())) {
          return;
        }
        user->SetOperand(operand_index, {message_.pred_merge_block_fresh_id()});
      });

  auto counter_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, ir_context->get_def_use_mgr()
                      ->GetDef(message_.variable_counter_id())
                      ->type_id());

  loop_header_block->terminator()->PreviousNode()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpLoad, counter_type_id,
          message_.load_counter_fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_counter_id()}}}))));

  loop_header_block->terminator()->PreviousNode()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpIAdd, counter_type_id,
          message_.increment_counter_fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.load_counter_fresh_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.constant_one_id()}}}))));

  loop_header_block->terminator()->PreviousNode()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_counter_id()}},
               {SPV_OPERAND_TYPE_ID,
                {message_.increment_counter_fresh_id()}}}))));

  auto bool_type_id = fuzzerutil::MaybeGetBoolType(ir_context);
  assert(bool_type_id && "The bool type must be present.");
  loop_header_block->terminator()->PreviousNode()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpSLessThan, bool_type_id,
          message_.condition_counter_fresh_id(),
          {{SPV_OPERAND_TYPE_ID, {message_.increment_counter_fresh_id()}},
           {SPV_OPERAND_TYPE_ID, {message_.constant_limit_id()}}})));

  loop_header_block->terminator()->InsertBefore(
      MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpBranchConditional, 0, 0,
          {{SPV_OPERAND_TYPE_ID, {message_.condition_counter_fresh_id()}},
           {SPV_OPERAND_TYPE_ID, {condition_block->id()}},
           {SPV_OPERAND_TYPE_ID, {merge_block->id()}}})));

  // Remove the OpBranch instruction.
  ir_context->KillInst(loop_header_block->terminator());

  enclosing_function->InsertBasicBlockBefore(std::move(pred_merge_block),
                                             merge_block);
  enclosing_function->InsertBasicBlockAfter(std::move(selection_merge_block),
                                            duplicated_merge_block);

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
