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

#include "source/fuzz/transformation_duplicate_region_with_selection.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationDuplicateRegionWithSelection::
    TransformationDuplicateRegionWithSelection(
        const spvtools::fuzz::protobufs::
            TransformationDuplicateRegionWithSelection& message)
    : message_(message) {}

TransformationDuplicateRegionWithSelection::
    TransformationDuplicateRegionWithSelection(
        uint32_t new_entry_fresh_id, uint32_t condition_id,
        uint32_t merge_label_fresh_id, uint32_t entry_block_id,
        uint32_t exit_block_id,
        std::map<uint32_t, uint32_t> original_label_to_duplicate_label,
        std::map<uint32_t, uint32_t> original_id_to_duplicate_id,
        std::map<uint32_t, uint32_t> original_id_to_phi_id) {
  message_.set_new_entry_fresh_id(new_entry_fresh_id);
  message_.set_condition_id(condition_id);
  message_.set_merge_label_fresh_id(merge_label_fresh_id);
  message_.set_entry_block_id(entry_block_id);
  message_.set_exit_block_id(exit_block_id);
  *message_.mutable_original_label_to_duplicate_label() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_label_to_duplicate_label);
  *message_.mutable_original_id_to_duplicate_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_id_to_duplicate_id);
  *message_.mutable_original_id_to_phi_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_id_to_phi_id);
}

bool TransformationDuplicateRegionWithSelection::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Instruction with the id |condition_id| must exist and must be of a bool
  // type.
  auto bool_instr =
      ir_context->get_def_use_mgr()->GetDef(message_.condition_id());
  if (bool_instr == nullptr || !bool_instr->type_id()) {
    return false;
  }
  if (!ir_context->get_type_mgr()->GetType(bool_instr->type_id())->AsBool()) {
    return false;
  }

  // The |new_entry_fresh_id| must be fresh and distinct.
  std::set<uint32_t> ids_used_by_this_transformation;
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_entry_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  // The |merge_label_fresh_id| must be fresh and distinct.
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.merge_label_fresh_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  // The entry and exit block ids must refer to blocks.
  for (auto block_id : {message_.entry_block_id(), message_.exit_block_id()}) {
    auto block_label = ir_context->get_def_use_mgr()->GetDef(block_id);
    if (!block_label || block_label->opcode() != SpvOpLabel) {
      return false;
    }
  }
  auto entry_block = ir_context->cfg()->block(message_.entry_block_id());
  auto exit_block = ir_context->cfg()->block(message_.exit_block_id());

  // The |entry_block| and the |exit_block| must be in the same function.
  if (entry_block->GetParent() != exit_block->GetParent()) {
    return false;
  }

  // The |entry_block| must dominate the |exit_block|.
  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(entry_block->GetParent());
  if (!dominator_analysis->Dominates(entry_block, exit_block)) {
    return false;
  }

  // The |exit_block| must post-dominate the |entry_block|.
  auto postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(entry_block->GetParent());
  if (!postdominator_analysis->Dominates(exit_block, entry_block)) {
    return false;
  }

  auto enclosing_function = entry_block->GetParent();

  // Currently we avoid the case where |entry_block| is the first block of the
  // |enclosing_function|.
  if (&*enclosing_function->begin() == entry_block) {
    return false;
  }

  auto region_set = GetRegionBlocks(ir_context, entry_block, exit_block);

  // Check whether |region_set| really is a single-entry single-exit region, and
  // also check whether structured control flow constructs and their merge
  // and continue constructs are either wholly in or wholly out of the region -
  // e.g. avoid the situation where the region contains the head of a loop but
  // not the loop's continue construct.
  //
  // This is achieved by going through every block in the |enclosing_function|
  for (auto& block : *enclosing_function) {
    if (&block == exit_block) {
      // It is not OK for the exit block to head a loop construct or a
      // conditional construct.
      if (block.GetMergeInst()) {
        return false;
      }
      continue;
    }
    if (region_set.count(&block) != 0) {
      // The block is in the region and is not the region's exit block.  Let's
      // see whether all of the block's successors are in the region. If they
      // are not, the region is not single-entry single-exit.
      bool all_successors_in_region = true;
      block.WhileEachSuccessorLabel([&all_successors_in_region, ir_context,
                                     &region_set](uint32_t successor) -> bool {
        if (region_set.count(ir_context->cfg()->block(successor)) == 0) {
          all_successors_in_region = false;
          return false;
        }
        return true;
      });
      if (!all_successors_in_region) {
        return false;
      }
    }

    if (auto merge = block.GetMergeInst()) {
      // The block is a loop or selection header -- the header and its
      // associated merge block had better both be in the region or both be
      // outside the region.
      auto merge_block =
          ir_context->cfg()->block(merge->GetSingleWordOperand(0));
      if (region_set.count(&block) != region_set.count(merge_block)) {
        return false;
      }
    }

    if (auto loop_merge = block.GetLoopMergeInst()) {
      // Similar to the above, but for the continue target of a loop
      // |continue_target| and a loop header |loop_merge|.
      auto continue_target =
          ir_context->cfg()->block(loop_merge->GetSingleWordOperand(1));
      if (continue_target != exit_block &&
          region_set.count(&block) != region_set.count(continue_target)) {
        return false;
      }
    }
  }

  // Get the maps from the protobuf.
  std::map<uint32_t, uint32_t> original_label_to_duplicate_label =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_label_to_duplicate_label());

  std::map<uint32_t, uint32_t> original_id_to_duplicate_id =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_id_to_duplicate_id());

  std::map<uint32_t, uint32_t> original_id_to_phi_id =
      fuzzerutil::RepeatedUInt32PairToMap(message_.original_id_to_phi_id());

  for (auto block : region_set) {
    auto label =
        ir_context->get_def_use_mgr()->GetDef(block->id())->result_id();
    // The label of every block in the region must be present in the map
    // |original_label_to_duplicate_label|.
    if (original_label_to_duplicate_label.count(label) == 0) {
      return false;
    }
    auto duplicate_label = original_label_to_duplicate_label[label];
    // Each id assigned to labels in the region must be distinct and fresh.
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            duplicate_label, ir_context, &ids_used_by_this_transformation)) {
      return false;
    }
    for (auto instr : *block) {
      if (!instr.result_id()) {
        continue;
      }
      // Every instruction with a result id in the region must be present in the
      // map |original_id_to_duplicate_id|.
      if (original_id_to_duplicate_id.count(instr.result_id()) == 0) {
        return false;
      }
      auto duplicate_id = original_id_to_duplicate_id[instr.result_id()];
      // Id assigned to this result id in the region must be distinct and fresh.
      if (!CheckIdIsFreshAndNotUsedByThisTransformation(
              duplicate_id, ir_context, &ids_used_by_this_transformation)) {
        return false;
      }
      if (&instr == &*exit_block->tail() ||
          fuzzerutil::IdIsAvailableBeforeInstruction(
              ir_context, &*exit_block->tail(), instr.result_id())) {
        // Every instruction with a result id available at the end of the region
        // must be present in the map |original_id_to_phi_id|.
        if (original_id_to_phi_id.count(instr.result_id()) == 0) {
          return false;
        }
        auto phi_id = original_id_to_phi_id[instr.result_id()];
        // Id assigned to this result id in the region must be distinct and
        // fresh.
        if (!CheckIdIsFreshAndNotUsedByThisTransformation(
                phi_id, ir_context, &ids_used_by_this_transformation)) {
          return false;
        }
      }
    }
  }
  return true;
}

std::set<opt::BasicBlock*>
TransformationDuplicateRegionWithSelection::GetRegionBlocks(
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

void TransformationDuplicateRegionWithSelection::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.new_entry_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.merge_label_fresh_id());

  // Create the new entry block containing the main conditional instruction. Set
  // its parent to the parent of the original entry block, since it is located
  // in the same function.
  std::unique_ptr<opt::BasicBlock> new_entry_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, SpvOpLabel, 0, message_.new_entry_fresh_id(),
          opt::Instruction::OperandList()));
  auto entry_block = ir_context->cfg()->block(message_.entry_block_id());
  auto enclosing_function = entry_block->GetParent();
  new_entry_block->SetParent(enclosing_function);
  auto exit_block = ir_context->cfg()->block(message_.exit_block_id());

  // Get the blocks contained in the region.
  std::set<opt::BasicBlock*> region_blocks =
      GetRegionBlocks(ir_context, entry_block, exit_block);

  // Construct the merge block and set its parent.
  std::unique_ptr<opt::BasicBlock> merge_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(opt::Instruction(
          ir_context, SpvOpLabel, 0, message_.merge_label_fresh_id(),
          opt::Instruction::OperandList())));
  merge_block->SetParent(enclosing_function);

  // Get the maps from the protobuf.
  std::map<uint32_t, uint32_t> original_label_to_duplicate_label =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_label_to_duplicate_label());

  std::map<uint32_t, uint32_t> original_id_to_duplicate_id =
      fuzzerutil::RepeatedUInt32PairToMap(
          message_.original_id_to_duplicate_id());

  std::map<uint32_t, uint32_t> original_id_to_phi_id =
      fuzzerutil::RepeatedUInt32PairToMap(message_.original_id_to_phi_id());

  // Before adding duplicate blocks, we need to update the OpPhi instructions in
  // the successors of the |exit_block|. We know that the execution of the
  // transformed region will end in |merge_block|. Hence, we need to change all
  // occurrences of the label id of the |exit_block| to the label id of the
  // |merge_block|.
  exit_block->ForEachSuccessorLabel([this, ir_context](uint32_t label_id) {
    auto block = ir_context->cfg()->block(label_id);
    for (auto& instr : *block) {
      if (instr.opcode() == SpvOpPhi) {
        instr.ForEachId([this](uint32_t* id) {
          if (*id == message_.exit_block_id()) {
            *id = message_.merge_label_fresh_id();
          }
        });
      }
    }
  });

  // Duplication of blocks will invalidate iterators. Store all the blocks from
  // the enclosing function.
  std::vector<opt::BasicBlock*> blocks;
  for (auto block_it = enclosing_function->begin();
       block_it != enclosing_function->end(); block_it++) {
    blocks.push_back(&*block_it);
  }
  // Store instructions OpReturn, OpReturnValue, OpBranch from the last block to
  // be moved (cloned and removed).
  std::vector<opt::Instruction*> instructions_to_move;
  opt::BasicBlock* previous_block = nullptr;

  // Iterate over all blocks of the function to duplicate blocks of the original
  // region and their instructions.
  for (auto& block : blocks) {
    // The block must be contained in the region.
    if (region_blocks.count(block) == 0) {
      continue;
    }

    fuzzerutil::UpdateModuleIdBound(
        ir_context, original_label_to_duplicate_label[block->id()]);

    std::unique_ptr<opt::BasicBlock> duplicated_block =
        MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
            ir_context, SpvOpLabel, 0,
            original_label_to_duplicate_label[block->id()],
            opt::Instruction::OperandList()));

    for (auto& instr : *block) {
      // Cases where an instruction is an OpBranch
      // or an OpReturn or an OpReturnValue in the exit block are handled
      // separately, after the iteration.
      if (block == exit_block && (instr.opcode() == SpvOpReturn ||
                                  instr.opcode() == SpvOpReturnValue ||
                                  instr.opcode() == SpvOpBranch)) {
        instructions_to_move.push_back(&instr);
        continue;
      }
      // Duplicate the instruction.
      auto cloned_instr = instr.Clone(ir_context);
      duplicated_block->AddInstruction(
          std::unique_ptr<opt::Instruction>(cloned_instr));

      fuzzerutil::UpdateModuleIdBound(
          ir_context, original_id_to_duplicate_id[instr.result_id()]);

      // If an id from the original region was used in this instruction,
      // replace it with the value from |original_id_to_duplicate_id|.
      // If a label from the original region was used in this instruction,
      // replace it with the value from |original_label_to_duplicate_label|.
      cloned_instr->ForEachId([original_id_to_duplicate_id,
                               original_label_to_duplicate_label,
                               ir_context](uint32_t* op) {
        if (original_id_to_duplicate_id.count(*op) != 0) {
          *op = original_id_to_duplicate_id.at(*op);
        }
        if (original_label_to_duplicate_label.count(*op) != 0) {
          *op = original_label_to_duplicate_label.at(*op);
        }
      });
    }

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
                                                exit_block);
    }
    previous_block = duplicated_block_ptr;
  }

  // After execution of the loop, this variable stores a pointer to the last
  // duplicated block.
  auto duplicated_exit_block = previous_block;

  for (auto& block : region_blocks) {
    for (auto& instr : *block) {
      if (instr.result_id() != 0 &&
          (&instr == &*exit_block->tail() ||
           fuzzerutil::IdIsAvailableBeforeInstruction(
               ir_context, &*exit_block->tail(), instr.result_id()))) {
        // Add the OpPhi instruction for every result id that is available at
        // the end of the region (the last instruction of the |exit_block|)
        merge_block->AddInstruction(MakeUnique<opt::Instruction>(
            ir_context, SpvOpPhi, instr.type_id(),
            original_id_to_phi_id[instr.result_id()],
            opt::Instruction::OperandList({
                {SPV_OPERAND_TYPE_ID, {instr.result_id()}},
                {SPV_OPERAND_TYPE_ID, {exit_block->id()}},
                {SPV_OPERAND_TYPE_ID,
                 {original_id_to_duplicate_id[instr.result_id()]}},
                {SPV_OPERAND_TYPE_ID, {duplicated_exit_block->id()}},
            })));

        fuzzerutil::UpdateModuleIdBound(
            ir_context, original_id_to_phi_id[instr.result_id()]);

        // If the instruction has been remapped by an OpPhi, look for all its
        // uses outside of the region and outside of the merge block (to not
        // overwrite just added instructions in the merge block) and replace the
        // original instruction id with the id of the corresponding OpPhi
        // instruction.
        ir_context->get_def_use_mgr()->ForEachUse(
            &instr,
            [ir_context, &instr, region_blocks, original_id_to_phi_id,
             &merge_block](opt::Instruction* user, uint32_t operand_index) {
              auto user_block = ir_context->get_instr_block(user);
              if ((region_blocks.find(user_block) != region_blocks.end()) ||
                  user_block == merge_block.get()) {
                return;
              }
              user->SetOperand(operand_index,
                               {original_id_to_phi_id.at(instr.result_id())});
            });
      }
    }
  }

  // Construct a conditional instruction in the |new_entry_block|. If the
  // condition is true, the execution proceeds in the |entry_block| of the
  // original region. If the condition is false, the execution proceeds in the
  // first block of the duplicated region.
  new_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpSelectionMerge, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.merge_label_fresh_id()}},
           {SPV_OPERAND_TYPE_SELECTION_CONTROL, {0}}})));

  new_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, SpvOpBranchConditional, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.condition_id()}},
           {SPV_OPERAND_TYPE_ID, {message_.entry_block_id()}},
           {SPV_OPERAND_TYPE_ID,
            {original_label_to_duplicate_label[message_.entry_block_id()]}}})));

  // If the exit block of the region contained an OpReturn or OpReturnValue
  // instruction we know that this was the exit block of the function. Move this
  // instruction to the merge block, since it will be the new exit block of the
  // function.
  //
  // If the exit block of the region contained an OpBranch instruction, it
  // refers to an another block outside of the region. Move this instruction to
  // the merge block, since the execution after the transformed region will
  // proceed from there.
  for (auto instr : instructions_to_move) {
    auto cloned_instr = instr->Clone(ir_context);
    merge_block->AddInstruction(
        std::unique_ptr<opt::Instruction>(cloned_instr));
    ir_context->KillInst(instr);
  }

  // Add OpBranch instruction to the merge block at the end of |exit_block| and
  // at the end of |duplicated_exit_block|, so that the execution proceeds in
  // the |merge_block|.
  opt::Instruction merge_branch_instr = opt::Instruction(
      ir_context, SpvOpBranch, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.merge_label_fresh_id()}}}));
  exit_block->AddInstruction(MakeUnique<opt::Instruction>(merge_branch_instr));
  duplicated_exit_block->AddInstruction(
      MakeUnique<opt::Instruction>(merge_branch_instr));

  // Execution needs to start in the |new_entry_block|. Change all the uses of
  // |entry_block_label_instr| outside of the original region to
  // |message_.new_entry_fresh_id|.
  auto entry_block_label_instr =
      ir_context->get_def_use_mgr()->GetDef(message_.entry_block_id());
  ir_context->get_def_use_mgr()->ForEachUse(
      entry_block_label_instr,
      [this, ir_context, region_blocks](opt::Instruction* user,
                                        uint32_t operand_index) {
        auto user_block = ir_context->get_instr_block(user);
        if ((region_blocks.count(user_block) != 0)) {
          return;
        }
        switch (user->opcode()) {
          case SpvOpSwitch:
          case SpvOpBranch:
          case SpvOpBranchConditional:
          case SpvOpLoopMerge:
          case SpvOpSelectionMerge: {
            user->SetOperand(operand_index, {message_.new_entry_fresh_id()});
          } break;
          case SpvOpName:
            break;
          default:
            assert(false &&
                   "The label id cannot be used by instructions other than "
                   "OpSwitch, OpBranch, OpBranchConditional, OpLoopMerge, "
                   "OpSelectionMerge");
        }
      });

  // Insert the merge block after the |duplicated_exit_block| (the last
  // duplicated block).
  enclosing_function->InsertBasicBlockAfter(std::move(merge_block),
                                            duplicated_exit_block);

  // Insert the |new_entry_block| before the entry block of the original region.
  enclosing_function->InsertBasicBlockBefore(std::move(new_entry_block),
                                             entry_block);

  // Since we have changed the module, most of the analysis are now invalid. We
  // can invalidate analyses now after all of the blocks have been registered.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation
TransformationDuplicateRegionWithSelection::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_duplicate_region_with_selection() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
