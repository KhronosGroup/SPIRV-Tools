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

#include "source/fuzz/transformation_flatten_conditional_branch.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationFlattenConditionalBranch::TransformationFlattenConditionalBranch(
    const protobufs::TransformationFlattenConditionalBranch& message)
    : message_(message) {}

TransformationFlattenConditionalBranch::TransformationFlattenConditionalBranch(
    uint32_t header_block_id,
    std::vector<
        std::pair<protobufs::InstructionDescriptor, IdsForEnclosingInst>>
        instructions_to_ids_for_enclosing) {
  message_.set_header_block_id(header_block_id);
  for (auto const& pair : instructions_to_ids_for_enclosing) {
    protobufs::InstToIdsForEnclosing inst_to_ids;
    *inst_to_ids.mutable_instruction() = pair.first;
    inst_to_ids.set_merge_block_id(pair.second.merge_block_id);
    inst_to_ids.set_execute_block_id(pair.second.execute_block_id);
    inst_to_ids.set_actual_result_id(pair.second.actual_result_id);
    inst_to_ids.set_alternative_block_id(pair.second.alternative_block_id);
    inst_to_ids.set_placeholder_result_id(pair.second.placeholder_result_id);
    *message_.add_inst_to_ids_for_enclosing() = inst_to_ids;
  }
}

bool TransformationFlattenConditionalBranch::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  uint32_t header_block_id = message_.header_block_id();
  auto header_block = fuzzerutil::MaybeFindBlock(ir_context, header_block_id);

  // The block must have been found and it must be a selection header.
  if (!header_block || !header_block->GetMergeInst() ||
      header_block->GetMergeInst()->opcode() != SpvOpSelectionMerge) {
    return false;
  }

  // The header block must end with an OpBranchConditional instruction.
  if (header_block->terminator()->opcode() != SpvOpBranchConditional) {
    return false;
  }

  // Use a set to keep track of the instructions that require fresh ids.
  std::set<opt::Instruction*> instructions_that_need_ids;

  // Check that, if there are enough ids, the conditional can be flattened and,
  // if so, add all the problematic instructions that need to be enclosed inside
  // conditionals to |instructions_that_need_ids|.
  if (!GetProblematicInstructionsIfConditionalCanBeFlattened(
          ir_context, header_block, &instructions_that_need_ids)) {
    return false;
  }

  // Get the mapping from instructions to the fresh ids needed to enclose them
  // inside conditionals.
  auto instructions_to_ids = GetInstructionsToIdsForEnclosing(ir_context);

  {
    // Check that all the ids given are fresh and distinct.

    std::set<uint32_t> used_fresh_ids;

    // Check the ids in the map.
    for (const auto& inst_to_ids : instructions_to_ids) {
      // Check the ids needed for all of the instructions that need to be
      // enclosed inside a conditional.
      for (uint32_t id : {inst_to_ids.second.merge_block_id,
                          inst_to_ids.second.execute_block_id}) {
        if (!id || !CheckIdIsFreshAndNotUsedByThisTransformation(
                       id, ir_context, &used_fresh_ids)) {
          return false;
        }
      }

      // Check the other ids needed, if the instruction needs a placeholder.
      if (InstructionNeedsPlaceholder(ir_context, *inst_to_ids.first)) {
        for (uint32_t id : {inst_to_ids.second.actual_result_id,
                            inst_to_ids.second.alternative_block_id,
                            inst_to_ids.second.placeholder_result_id}) {
          if (!id || !CheckIdIsFreshAndNotUsedByThisTransformation(
                         id, ir_context, &used_fresh_ids)) {
            return false;
          }
        }
      }
    }
  }

  // If some instructions that require ids are not in the map, the
  // transformation needs overflow ids to be applicable.
  for (auto instruction : instructions_that_need_ids) {
    if (instructions_to_ids.count(instruction) == 0 &&
        !transformation_context.GetOverflowIdSource()->HasOverflowIds()) {
      return false;
    }
  }

  // All checks were passed.
  return true;
}

void TransformationFlattenConditionalBranch::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  uint32_t header_block_id = message_.header_block_id();
  auto header_block = ir_context->cfg()->block(header_block_id);

  // Find the first block where flow converges (it is not necessarily the merge
  // block).
  uint32_t convergence_block_id = header_block->MergeBlockId();
  while (ir_context->cfg()->preds(convergence_block_id).size() == 1) {
    convergence_block_id = ir_context->cfg()->preds(convergence_block_id)[0];
  }

  // Get the mapping from instructions to fresh ids.
  auto instructions_to_fresh_ids = GetInstructionsToIdsForEnclosing(ir_context);

  auto branch_instruction = header_block->terminator();

  // Get a reference to the last block in the true branch, before flow
  // converges (if there is a true branch).
  opt::BasicBlock* last_true_block = nullptr;

  // Adjust the conditional branches by enclosing problematic instructions
  // within conditionals and get references to the last block in each branch.
  for (int branch = 2; branch >= 1; branch--) {
    // branch = 1 corresponds to the true branch, branch = 2 corresponds to the
    // false branch. Consider the false branch first so that the true branch is
    // laid out right after the header.

    auto current_block = header_block;
    // Get the id of the first block in this branch.
    uint32_t next_block_id = branch_instruction->GetSingleWordInOperand(branch);

    // Consider all blocks in the branch until the convergence block is reached.
    while (next_block_id != convergence_block_id) {
      // Move the next block to right after the current one.
      current_block->GetParent()->MoveBasicBlockToAfter(next_block_id,
                                                        current_block);

      // Move forward in the branch.
      current_block = ir_context->cfg()->block(next_block_id);

      // Find all the instructions in the current block which need to be
      // enclosed inside conditionals.
      std::vector<opt::Instruction*> problematic_instructions;

      current_block->ForEachInst(
          [&problematic_instructions](opt::Instruction* instruction) {
            if (instruction->opcode() != SpvOpLabel &&
                instruction->opcode() != SpvOpBranch &&
                !fuzzerutil::InstructionHasNoSideEffects(*instruction)) {
              problematic_instructions.push_back(instruction);
            }
          });

      uint32_t condition_id =
          header_block->terminator()->GetSingleWordInOperand(0);

      // Enclose all of the problematic instructions in conditionals, with the
      // same condition as the selection construct being flattened.
      for (auto instruction : problematic_instructions) {
        // Get the fresh_ids needed by this instructions
        IdsForEnclosingInst fresh_ids;

        if (instructions_to_fresh_ids.count(instruction) != 0) {
          // Get the fresh ids from the map, if present.
          fresh_ids = instructions_to_fresh_ids[instruction];
        } else {
          // If we could not get it from the map, use overflow ids.
          fresh_ids.merge_block_id =
              transformation_context->GetOverflowIdSource()
                  ->GetNextOverflowId();
          fresh_ids.execute_block_id =
              transformation_context->GetOverflowIdSource()
                  ->GetNextOverflowId();

          if (InstructionNeedsPlaceholder(ir_context, *instruction)) {
            fresh_ids.actual_result_id =
                transformation_context->GetOverflowIdSource()
                    ->GetNextOverflowId();
            fresh_ids.alternative_block_id =
                transformation_context->GetOverflowIdSource()
                    ->GetNextOverflowId();
            fresh_ids.placeholder_result_id =
                transformation_context->GetOverflowIdSource()
                    ->GetNextOverflowId();
          }
        }

        // Enclose the instruction in a conditional and get the merge block
        // generated by this operation (this is where all the following instructions
        // will be).
        current_block = EncloseInstructionInConditional(
            ir_context, transformation_context, current_block, instruction,
            fresh_ids, condition_id, branch == 1);
      }

      next_block_id = current_block->terminator()->GetSingleWordInOperand(0);

      // If the next block is the convergence block and this is the true branch,
      // record this as the last block in the true branch.
      if (next_block_id == convergence_block_id && branch == 1) {
        last_true_block = current_block;
      }
    }
  }

  // Get the condition operand and the ids of the first blocks of the true and
  // false branches.
  auto condition_operand = branch_instruction->GetInOperand(0);
  uint32_t first_true_block_id = branch_instruction->GetSingleWordInOperand(1);
  uint32_t first_false_block_id = branch_instruction->GetSingleWordInOperand(2);

  // The current header should unconditionally branch to the first block in the
  // true branch, if there exists a true branch, and to the first block in the
  // false branch if there is no true branch.
  uint32_t after_header = first_true_block_id != convergence_block_id
                              ? first_true_block_id
                              : first_false_block_id;

  // Kill the merge instruction and the branch instruction in the current
  // header.
  auto merge_inst = header_block->GetMergeInst();
  ir_context->KillInst(branch_instruction);
  ir_context->KillInst(merge_inst);

  // Add a new, unconditional, branch instruction from the current header to
  // |after_header|.
  header_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpBranch, 0, 0,
      opt::Instruction::OperandList{{SPV_OPERAND_TYPE_ID, {after_header}}}));

  // If there is a true branch, change the branch instruction so that the last
  // block in the true branch unconditionally branches to the first block in the
  // false branch (or the convergence block if there is no false branch).
  if (last_true_block) {
    last_true_block->terminator()->SetInOperand(0, {first_false_block_id});
  }

  // Replace all of the current OpPhi instructions in the convergence block with
  // OpSelect.
  ir_context->get_instr_block(convergence_block_id)
      ->ForEachPhiInst([&condition_operand](opt::Instruction* phi_inst) {
        phi_inst->SetOpcode(SpvOpSelect);
        std::vector<opt::Operand> operands;
        operands.emplace_back(condition_operand);
        // Only consider the operands referring to the instructions ids, as the
        // block labels are not necessary anymore.
        for (uint32_t i = 0; i < phi_inst->NumInOperands(); i += 2) {
          operands.emplace_back(phi_inst->GetInOperand(i));
        }
        phi_inst->SetInOperands(std::move(operands));
      });

  // Invalidate all analyses
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationFlattenConditionalBranch::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_flatten_conditional_branch() = message_;
  return result;
}

bool TransformationFlattenConditionalBranch::
    GetProblematicInstructionsIfConditionalCanBeFlattened(
        opt::IRContext* ir_context, opt::BasicBlock* header,
        std::set<opt::Instruction*>* instructions_that_need_ids) {
  uint32_t merge_block_id = header->MergeBlockIdIfAny();
  assert(merge_block_id &&
         header->GetMergeInst()->opcode() == SpvOpSelectionMerge &&
         header->terminator()->opcode() == SpvOpBranchConditional &&
         "|header| must be the header of a conditional.");

  // Find the first block where flow converges (it is not necessarily the merge
  // block).
  uint32_t convergence_block_id = merge_block_id;
  while (ir_context->cfg()->preds(convergence_block_id).size() == 1) {
    if (convergence_block_id == header->id()) {
      // There is a chain of blocks with one predecessor from the header block
      // to the merge block. This means that the region is not single-entry,
      // single-exit (because the merge block is only reached by one of the two
      // branches).
      return false;
    }
    convergence_block_id = ir_context->cfg()->preds(convergence_block_id)[0];
  }

  auto enclosing_function = header->GetParent();
  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(enclosing_function);
  auto postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(enclosing_function);

  // Check that this is a single-entry, single-exit region, by checking that the
  // header dominates the convergence block and that the convergence block
  // post-dominates the header.
  if (!dominator_analysis->Dominates(header->id(), convergence_block_id) ||
      !postdominator_analysis->Dominates(convergence_block_id, header->id())) {
    return false;
  }

  // Traverse the CFG starting from the header and check that, for all the
  // blocks that can be reached by the header before reaching the convergence
  // block:
  //  - they don't contain merge, barrier or OpSampledImage instructions
  //  - they branch unconditionally to another block
  //  Add any side-effecting instruction, requiring fresh ids, to
  //  |instructions_that_need_ids|
  std::list<uint32_t> to_check;
  header->ForEachSuccessorLabel(
      [&to_check](uint32_t label) { to_check.push_back(label); });

  while (!to_check.empty()) {
    uint32_t block_id = to_check.front();
    to_check.pop_front();

    if (block_id == convergence_block_id) {
      // We have reached the convergence block, we don't need to consider its
      // successors.
      continue;
    }

    auto block = ir_context->cfg()->block(block_id);

    // The block must not have a merge instruction, because inner constructs are
    // not allowed.
    if (block->GetMergeInst()) {
      return false;
    }

    // Check all of the instructions in the block.
    bool all_instructions_compatible =
        block->WhileEachInst([ir_context, instructions_that_need_ids](
                                 opt::Instruction* instruction) {
          // We can ignore OpLabel instructions.
          if (instruction->opcode() == SpvOpLabel) {
            return true;
          }

          // If the instruction is a branch, it must be an unconditional branch.
          if (instruction->IsBranch()) {
            return instruction->opcode() == SpvOpBranch;
          }

          // We cannot go ahead if we encounter an instruction that cannot be
          // handled.
          if (!InstructionCanBeHandled(ir_context, *instruction)) {
            return false;
          }

          // If the instruction has side effects, add it to the
          // |instructions_that_need_ids| set.
          if (!fuzzerutil::InstructionHasNoSideEffects(*instruction)) {
            instructions_that_need_ids->emplace(instruction);
          }

          return true;
        });

    if (!all_instructions_compatible) {
      return false;
    }

    // Add the successor of this block to the list of blocks that need to be
    // checked.
    to_check.push_back(block->terminator()->GetSingleWordInOperand(0));
  }

  // All the blocks are compatible with the transformation and this is indeed a
  // single-entry, single-exit region.
  return true;
}

bool TransformationFlattenConditionalBranch::InstructionNeedsPlaceholder(
    opt::IRContext* ir_context, const opt::Instruction& instruction) {
  assert(!fuzzerutil::InstructionHasNoSideEffects(instruction) &&
         InstructionCanBeHandled(ir_context, instruction) &&
         "The instruction must have side effects and it must be possible to "
         "enclose it inside a conditional.");

  if (instruction.HasResultId()) {
    // We need a placeholder iff the type is not Void.
    auto type = ir_context->get_type_mgr()->GetType(instruction.type_id());
    return type && !type->AsVoid();
  }

  return false;
}

std::unordered_map<opt::Instruction*, IdsForEnclosingInst>
TransformationFlattenConditionalBranch::GetInstructionsToIdsForEnclosing(
    opt::IRContext* ir_context) const {
  std::unordered_map<opt::Instruction*, IdsForEnclosingInst>
      instructions_to_ids;
  for (const auto& inst_to_ids : message_.inst_to_ids_for_enclosing()) {
    IdsForEnclosingInst ids = {
        inst_to_ids.merge_block_id(), inst_to_ids.execute_block_id(),
        inst_to_ids.actual_result_id(), inst_to_ids.alternative_block_id(),
        inst_to_ids.placeholder_result_id()};

    auto instruction = FindInstruction(inst_to_ids.instruction(), ir_context);
    if (instruction) {
      instructions_to_ids.emplace(instruction, std::move(ids));
    }
  }

  return instructions_to_ids;
}

opt::BasicBlock*
TransformationFlattenConditionalBranch::EncloseInstructionInConditional(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    opt::BasicBlock* block, opt::Instruction* instruction,
    const IdsForEnclosingInst& fresh_ids, uint32_t condition_id,
    bool exec_if_cond_true) const {
  // Get the next instruction (it will be useful for splitting).
  auto next_instruction = instruction->NextNode();

  // Update the module id bound.
  for (uint32_t id : {fresh_ids.merge_block_id, fresh_ids.execute_block_id}) {
    fuzzerutil::UpdateModuleIdBound(ir_context, id);
  }

  // Create the block where the instruction is executed by splitting the
  // original block.
  auto execute_block = block->SplitBasicBlock(
      ir_context, fresh_ids.execute_block_id,
      fuzzerutil::GetIteratorForInstruction(block, instruction));

  // Create the merge block for the conditional that we are about to create by
  // splitting execute_block (this will leave |instruction| as the only
  // instruction in |execute_block|).
  auto merge_block = execute_block->SplitBasicBlock(
      ir_context, fresh_ids.merge_block_id,
      fuzzerutil::GetIteratorForInstruction(execute_block, next_instruction));

  // Propagate the fact that the block is dead to the newly-created blocks.
  if (transformation_context->GetFactManager()->BlockIsDead(block->id())) {
    transformation_context->GetFactManager()->AddFactBlockIsDead(
        execute_block->id());
    transformation_context->GetFactManager()->AddFactBlockIsDead(
        merge_block->id());
  }

  // Initially, consider the merge block as the alternative block to branch to
  // if the instruction should not be executed.
  auto alternative_block = merge_block;

  // Add an unconditional branch from |execute_block| to |merge_block|.
  execute_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpBranch, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {merge_block->id()}}}));

  // If the instruction requires a placeholder, it means that it has a result id
  // and its result needs to be used later on, and we need to:
  // - add an additional block |fresh_ids.alternative_block_id| where a
  // placeholder
  //   result id (using fresh id |fresh_ids.placeholder_result_id|) is obtained
  //   by using the OpUndef instruction
  // - change the result id of the instruction to a fresh id
  //   (|fresh_ids.actual_result_id|).
  // - add an OpPhi instruction, which will have the original result id of the
  //   instruction, in the merge block.
  if (InstructionNeedsPlaceholder(ir_context, *instruction)) {
    // Update the module id bound with the additional ids.
    for (uint32_t id :
         {fresh_ids.actual_result_id, fresh_ids.alternative_block_id,
          fresh_ids.placeholder_result_id}) {
      fuzzerutil::UpdateModuleIdBound(ir_context, id);
    }

    // Create a new block using |fresh_ids.alternative_block_id| for its label.
    auto alternative_block_temp =
        MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
            ir_context, SpvOpLabel, 0, fresh_ids.alternative_block_id,
            opt::Instruction::OperandList{}));

    // Keep the original result id of the instruction in a variable.
    uint32_t original_result_id = instruction->result_id();

    // Set the result id of the instruction to be |fresh_ids.actual_result_id|.
    instruction->SetResultId(fresh_ids.actual_result_id);

    // Add an OpUndef instruction, with the same type as the original
    // instruction and id |fresh_ids.placeholder_result_id|, to the new block.
    alternative_block_temp->AddInstruction(MakeUnique<opt::Instruction>(
        ir_context, SpvOpUndef, instruction->type_id(),
        fresh_ids.placeholder_result_id, opt::Instruction::OperandList{}));

    // Add an unconditional branch from the new block to the merge block.
    alternative_block_temp->AddInstruction(MakeUnique<opt::Instruction>(
        ir_context, SpvOpBranch, 0, 0,
        opt::Instruction::OperandList{
            {SPV_OPERAND_TYPE_ID, {merge_block->id()}}}));

    // Insert the block before the merge block.
    alternative_block = block->GetParent()->InsertBasicBlockBefore(
        std::move(alternative_block_temp), merge_block);

    // Using the original instruction result id, add an OpPhi instruction to the
    // merge block, which will either take the value of the result of the
    // instruction or the placeholder value defined in the alternative block.
    merge_block->begin().InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpPhi, instruction->type_id(), original_result_id,
        opt::Instruction::OperandList{
            {SPV_OPERAND_TYPE_ID, {instruction->result_id()}},
            {SPV_OPERAND_TYPE_ID, {execute_block->id()}},
            {SPV_OPERAND_TYPE_ID, {fresh_ids.placeholder_result_id}},
            {SPV_OPERAND_TYPE_ID, {alternative_block->id()}}}));

    // Propagate the fact that the block is dead to the new block.
    if (transformation_context->GetFactManager()->BlockIsDead(block->id())) {
      transformation_context->GetFactManager()->AddFactBlockIsDead(
          alternative_block->id());
    }
  }

  // Depending on whether the instruction should be executed in the if branch or
  // in the else branch, get the corresponding ids.
  auto if_block_id = (exec_if_cond_true ? execute_block : alternative_block)
                         ->GetLabel()
                         ->result_id();
  auto else_block_id = (exec_if_cond_true ? alternative_block : execute_block)
                           ->GetLabel()
                           ->result_id();

  // Add an OpSelectionMerge instruction to the block.
  block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpSelectionMerge, 0, 0,
      opt::Instruction::OperandList{{SPV_OPERAND_TYPE_ID, {merge_block->id()}},
                                    {SPV_OPERAND_TYPE_SELECTION_CONTROL,
                                     {SpvSelectionControlMaskNone}}}));

  // Add an OpBranchConditional, to the block, using |condition_id| as the
  // condition and branching to |if_block_id| if the condition is true and to
  // |else_block_id| if the condition is false.
  block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpBranchConditional, 0, 0,
      opt::Instruction::OperandList{{SPV_OPERAND_TYPE_ID, {condition_id}},
                                    {SPV_OPERAND_TYPE_ID, {if_block_id}},
                                    {SPV_OPERAND_TYPE_ID, {else_block_id}}}));

  return merge_block;
}

bool TransformationFlattenConditionalBranch::InstructionCanBeHandled(
    opt::IRContext* ir_context, const opt::Instruction& instruction) {
  // We can handle all instructions with no side effects.
  if (fuzzerutil::InstructionHasNoSideEffects(instruction)) {
    return true;
  }

  // We cannot handle barrier instructions, while we should be able to handle
  // all other instructions by enclosing them inside a conditional.
  if (instruction.opcode() == SpvOpControlBarrier ||
      instruction.opcode() == SpvOpMemoryBarrier ||
      instruction.opcode() == SpvOpNamedBarrierInitialize ||
      instruction.opcode() == SpvOpMemoryNamedBarrier ||
      instruction.opcode() == SpvOpTypeNamedBarrier) {
    return false;
  }

  // We cannot handle OpSampledImage instructions, as they need to be in the
  // same block as their use.
  if (instruction.opcode() == SpvOpSampledImage) {
    return false;
  }

  // We cannot handle instructions with an id which return a void type, if the
  // result id is used in the module (e.g. a function call to a function that
  // returns nothing).
  if (instruction.HasResultId()) {
    auto type = ir_context->get_type_mgr()->GetType(instruction.type_id());
    assert(type && "The type should be found in the module");

    if (type->AsVoid() &&
        !ir_context->get_def_use_mgr()->WhileEachUse(
            instruction.result_id(),
            [](opt::Instruction* use_inst, uint32_t use_index) {
              // Return false if the id is used as an input operand.
              return use_index <
                     use_inst->NumOperands() - use_inst->NumInOperands();
            })) {
      return false;
    }
  }

  return true;
}

}  // namespace fuzz
}  // namespace spvtools
