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
        std::pair<protobufs::InstructionDescriptor, std::vector<uint32_t>>>
        instructions_to_fresh_ids,
    std::vector<uint32_t> overflow_ids) {
  message_.set_header_block_id(header_block_id);
  for (auto const& pair : instructions_to_fresh_ids) {
    protobufs::InstructionUint32ListPair mapping;
    *mapping.mutable_instruction_descriptor() = pair.first;
    for (auto id : pair.second) {
      mapping.add_id(id);
    }
    *message_.add_instruction_to_fresh_ids() = mapping;
  }
  for (auto id : overflow_ids) {
    message_.add_overflow_id(id);
  }
}

bool TransformationFlattenConditionalBranch::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  uint32_t header_block_id = message_.header_block_id();

  // |header_block_id| must refer to a block label.
  {
    auto label = ir_context->get_def_use_mgr()->GetDef(header_block_id);
    if (!label || label->opcode() != SpvOpLabel) {
      return false;
    }
  }

  auto header_block = ir_context->cfg()->block(header_block_id);

  // |header_block| must be a selection header.
  uint32_t merge_block_id = header_block->MergeBlockIdIfAny();
  if (!merge_block_id ||
      header_block->GetMergeInst()->opcode() != SpvOpSelectionMerge) {
    return false;
  }

  // The header block must end with an OpBranchConditional instruction.
  if (header_block->terminator()->opcode() != SpvOpBranchConditional) {
    return false;
  }

  // Find the first block where flow converges (it is not necessarily the merge
  // block).
  uint32_t convergence_block_id = merge_block_id;
  while (ir_context->cfg()->preds(convergence_block_id).size() == 1) {
    if (convergence_block_id == header_block_id) {
      // There is a chain of blocks with one predecessor from the header block
      // to the merge block. This means that the region is not single-entry,
      // single-exit (because the merge block is only reached by one of the two
      // branches).
      return false;
    }
    convergence_block_id = ir_context->cfg()->preds(convergence_block_id)[0];
  }

  // Get all the blocks reachable by the header block before reaching the
  // convergence block and check that, for each of these blocks, that:
  //  - the header dominates it and the convergence block postdominates it (so
  //    that the header and merge block form a single-entry, single-exit
  //    region)
  //  - it does not contain merge instructions
  //  - it branches unconditionally to another block
  //  - it does not contain atomic or barrier instructions
  auto enclosing_function = header_block->GetParent();
  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(enclosing_function);
  auto postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(enclosing_function);

  // Get the mapping from instructions to the fresh ids available for them.
  auto instructions_to_fresh_ids = GetInstructionsToFreshIdsMapping(ir_context);

  {
    // Check that all the ids given are fresh and distinct.

    std::set<uint32_t> used_fresh_ids;

    // Check the overflow ids.
    for (uint32_t id : message_.overflow_id()) {
      if (!CheckIdIsFreshAndNotUsedByThisTransformation(id, ir_context,
                                                        &used_fresh_ids)) {
        return false;
      }
    }

    // Check the ids in the map.
    for (auto pair : instructions_to_fresh_ids) {
      for (uint32_t id : pair.second) {
        if (!CheckIdIsFreshAndNotUsedByThisTransformation(id, ir_context,
                                                          &used_fresh_ids)) {
          return false;
        }
      }
    }
  }

  // Keep track of the number of overflow ids still available in the overflow
  // pool, as we go through the instructions.
  int remaining_overflow_ids = message_.overflow_id_size();

  // Perform a BST to find and check all the blocks that can be reached by the
  // header before reaching the convergence block.
  std::list<uint32_t> to_check;
  header_block->ForEachSuccessorLabel(
      [&to_check](uint32_t label) { to_check.push_back(label); });

  while (!to_check.empty()) {
    uint32_t block_id = to_check.front();
    to_check.pop_front();

    if (block_id == convergence_block_id) {
      // We have reached the convergence block, we don't need to consider its
      // successors.
      continue;
    }

    // If the block is not dominated by the header or it is not postdominated by
    // the convergence_block, this is not a single-entry, single-exit region.
    if (!dominator_analysis->Dominates(header_block_id, block_id) ||
        !postdominator_analysis->Dominates(convergence_block_id, block_id)) {
      return false;
    }

    auto block = ir_context->cfg()->block(block_id);

    // The block must not have a merge instruction, because inner constructs are
    // not allowed.
    if (block->GetMergeInst()) {
      return false;
    }

    // We need to make sure that OpSampledImage instructions will not be
    // separated from their use, as they need to be in the same block.

    // All result ids of an OpSampledImage instruction occurring before the last
    // point where the block will need to be split.
    std::set<uint32_t> sampled_image_result_ids_before_split;

    // All result ids of an OpSampledImage instruction occurring after the last
    // point where the block will need to be split. They can still be used.
    std::set<uint32_t> sampled_image_result_ids_after_split;

    // Check all of the instructions in the block.
    bool all_instructions_compatible = block->WhileEachInst(
        [this, &instructions_to_fresh_ids, &remaining_overflow_ids,
         &sampled_image_result_ids_before_split,
         &sampled_image_result_ids_after_split](opt::Instruction* instruction) {
          // The instruction cannot be an atomic or barrier instruction
          if (instruction->IsAtomicOp() ||
              instruction->opcode() == SpvOpControlBarrier ||
              instruction->opcode() == SpvOpMemoryBarrier ||
              instruction->opcode() == SpvOpNamedBarrierInitialize ||
              instruction->opcode() == SpvOpMemoryNamedBarrier ||
              instruction->opcode() == SpvOpTypeNamedBarrier) {
            return false;
          }

          // If the instruction is OpSampledImage, add the result id to
          // |sampled_image_result_ids_after_split|.
          if (instruction->opcode() == SpvOpSampledImage) {
            sampled_image_result_ids_after_split.emplace(
                instruction->result_id());
          }

          // If the instruction uses an OpSampledImage that appeared before a
          // point where we need to split the block (before a load, store or
          // function call), then the transformation is not applicable.
          if (!instruction->WhileEachInId(
                  [&sampled_image_result_ids_before_split](
                      uint32_t* id) -> bool {
                    return !sampled_image_result_ids_before_split.count(*id);
                  })) {
            return false;
          }

          // If the instruction is a load, store or function call, there must
          // be a mapping from the corresponding instruction descriptor to a
          // list of fresh ids or there must be enough overflow ids.
          if (instruction->opcode() == SpvOpLoad ||
              instruction->opcode() == SpvOpStore ||
              instruction->opcode() == SpvOpFunctionCall) {
            // The number of ids needed depends on the id of the instruction.
            uint32_t ids_needed_by_this_instruction =
                NumOfFreshIdsNeededByOpcode(instruction->opcode());

            if (instructions_to_fresh_ids.count(instruction) != 0) {
              // If there is a mapping from this instruction to a list of fresh
              // ids, the list must have enough ids.

              if (instructions_to_fresh_ids[instruction].size() <
                  ids_needed_by_this_instruction) {
                return false;
              }
            } else {
              // If there is no mapping, we need to rely on the pool of
              // overflow ids, where there must be enough remaining ids.

              remaining_overflow_ids -= ids_needed_by_this_instruction;

              if (remaining_overflow_ids < 0) {
                return false;
              }
            }

            // All OpSampledImage ids defined before this point should not be
            // used anymore. We need to move all ids in
            // |sampled_image_result_ids_after_split| to
            // |sampled_image_result_ids_before_split| so that this can be
            // checked for the following instructions.
            for (auto id : sampled_image_result_ids_after_split) {
              sampled_image_result_ids_before_split.emplace(id);
            }
            sampled_image_result_ids_after_split.clear();
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
  auto instructions_to_fresh_ids = GetInstructionsToFreshIdsMapping(ir_context);

  // Keep track of the number of overflow ids used.
  uint32_t overflow_ids_used = 0;

  auto branch_instruction = header_block->terminator();

  opt::BasicBlock* last_true_block = nullptr;

  // Adjust the conditional branches by enclosing problematic instructions
  // within conditionals and get references to the last block in each branch.
  for (int branch = 2; branch >= 1; branch--) {
    // branch = 1 corresponds to the true branch, branch = 2 corresponds to the
    // false branch. Consider the false branch first so that the true branch is
    // laid out right after the false branch.

    auto block = header_block;
    // Get the id of the first block in this branch.
    uint32_t block_id = branch_instruction->GetSingleWordInOperand(branch);

    // Consider all blocks in the branch until the convergence block is reached.
    while (block_id != convergence_block_id) {
      // Move the block to right after the previous one.
      block->GetParent()->MoveBasicBlockToAfter(block_id, block);

      block = ir_context->cfg()->block(block_id);
      block_id = block->terminator()->GetSingleWordInOperand(0);

      // Find all the problematic instructions in the block (OpStore, OpLoad,
      // OpFunctionCall).
      std::vector<opt::Instruction*> problematic_instructions;

      block->ForEachInst(
          [&problematic_instructions](opt::Instruction* instruction) {
            switch (instruction->opcode()) {
              case SpvOpStore:
              case SpvOpLoad:
              case SpvOpFunctionCall:
                problematic_instructions.push_back(instruction);
                break;
              default:
                break;
            }
          });

      uint32_t condition_id =
          header_block->terminator()->GetSingleWordInOperand(0);

      // Enclose all of the problematic instructions in conditionals, with the
      // same condition as the selection construct being flattened.
      for (auto instruction : problematic_instructions) {
        // Collect the fresh ids needed by this instructions
        uint32_t ids_needed =
            NumOfFreshIdsNeededByOpcode(instruction->opcode());
        std::vector<uint32_t> fresh_ids;

        // Get them from the map.
        if (instructions_to_fresh_ids.count(instruction) != 0) {
          fresh_ids = instructions_to_fresh_ids[instruction];
        }

        // Get the ones still needed from the overflow ids.
        for (int still_needed = ids_needed - (int)fresh_ids.size();
             still_needed > 0; still_needed--) {
          fresh_ids.push_back(message_.overflow_id(overflow_ids_used++));
        }

        // Enclose the instruction in a conditional and get the merge block
        // generated by this operation (this is where all the next instructions
        // will be).
        block = EncloseInstructionInConditional(
            ir_context, transformation_context, block, instruction, fresh_ids,
            condition_id, branch == 1);
      }

      // If the next block is the convergence block and this is the true branch,
      // record this as the last block in the true branch.
      if (block_id == convergence_block_id && branch == 1) {
        last_true_block = block;
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

std::unordered_map<opt::Instruction*, std::vector<uint32_t>>
TransformationFlattenConditionalBranch::GetInstructionsToFreshIdsMapping(
    opt::IRContext* ir_context) const {
  std::unordered_map<opt::Instruction*, std::vector<uint32_t>>
      instructions_to_fresh_ids;
  for (auto pair : message_.instruction_to_fresh_ids()) {
    std::vector<uint32_t> fresh_ids;
    for (uint32_t id : pair.id()) {
      fresh_ids.push_back(id);
    }

    auto instruction =
        FindInstruction(pair.instruction_descriptor(), ir_context);
    if (instruction) {
      instructions_to_fresh_ids.emplace(instruction, std::move(fresh_ids));
    }
  }

  return instructions_to_fresh_ids;
}

uint32_t TransformationFlattenConditionalBranch::NumOfFreshIdsNeededByOpcode(
    SpvOp opcode) const {
  switch (opcode) {
    case SpvOpStore:
      // 2 ids are needed for two new blocks, which will be used to enclose the
      // instruction within a conditional.
      return 2;
    case SpvOpLoad:
    case SpvOpFunctionCall:
      // These instructions return a result, so we need 3 fresh ids for new
      // blocks (the true block, the false block and the merge block), one for
      // the instruction itself and one for an instruction returning a dummy
      // value. The original result id of the instruction will be used for a new
      // OpPhi instruction.
      return 5;
    default:
      assert(false && "This line should never be reached");
      return 0;
  }
}

opt::BasicBlock*
TransformationFlattenConditionalBranch::EncloseInstructionInConditional(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    opt::BasicBlock* block, opt::Instruction* instruction,
    const std::vector<uint32_t>& fresh_ids, uint32_t condition_id,
    bool exec_if_cond_true) const {
  assert((instruction->opcode() == SpvOpStore ||
          instruction->opcode() == SpvOpLoad ||
          instruction->opcode() == SpvOpFunctionCall) &&
         "This function should only be called on OpStore, OpLoad or "
         "OpFunctionCall instructions.");

  // Get the next instruction (it will be useful for splitting).
  auto next_instruction = instruction->NextNode();

  // We need at least 2 fresh ids for two new blocks.
  assert(fresh_ids.size() >= 2 && "Not enough fresh ids.");

  // Update the module id bound
  for (auto id : fresh_ids) {
    fuzzerutil::UpdateModuleIdBound(ir_context, id);
  }

  // Create the block where the instruction is executed by splitting the
  // original block.
  auto execute_block = block->SplitBasicBlock(
      ir_context, fresh_ids[0],
      fuzzerutil::GetIteratorForInstruction(block, instruction));

  // Create the merge block for the conditional that we are about to create by
  // splitting execute_block (this will leave |instruction| as the only
  // instruction in |execute_block|).
  auto merge_block = execute_block->SplitBasicBlock(
      ir_context, fresh_ids[1],
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

  // If the instruction has a result id, we need to:
  // - add an additional block where a dummy result is obtained by using the
  // OpUndef instruction
  // - change the result id of the instruction to a fresh id
  // - add an OpPhi instruction, which will have the original result id of the
  //   instruction, in the merge block.
  if (instruction->HasResultId()) {
    // We need 3 more fresh ids for 1 additional block and 2 additional
    // instructions.
    assert(fresh_ids.size() >= 5 && "Not enough fresh ids.");

    // Create a new block using a fresh id for its label.
    auto alternative_block_temp = MakeUnique<opt::BasicBlock>(
        MakeUnique<opt::Instruction>(ir_context, SpvOpLabel, 0, fresh_ids[2],
                                     opt::Instruction::OperandList{}));

    // Keep the original result id of the instruction in a variable.
    uint32_t original_result_id = instruction->result_id();

    // Set the result id of the instruction to a fresh id.
    instruction->SetResultId(fresh_ids[3]);

    // Add an OpUndef instruction, with the same type as the original
    // instruction and a fresh id, to the new block.
    alternative_block_temp->AddInstruction(MakeUnique<opt::Instruction>(
        ir_context, SpvOpUndef, instruction->type_id(), fresh_ids[4],
        opt::Instruction::OperandList{}));

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
    // instruction or the dummy value defined in the alternative block.
    merge_block->begin().InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpPhi, instruction->type_id(), original_result_id,
        opt::Instruction::OperandList{
            {SPV_OPERAND_TYPE_ID, {instruction->result_id()}},
            {SPV_OPERAND_TYPE_ID, {execute_block->id()}},
            {SPV_OPERAND_TYPE_ID, {fresh_ids[4]}},
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
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {merge_block->id()}},
          {SPV_OPERAND_TYPE_SELECTION_CONTROL, {0}}}));

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

protobufs::Transformation TransformationFlattenConditionalBranch::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_flatten_conditional_branch() = message_;
  return result;
}
}  // namespace fuzz
}  // namespace spvtools
