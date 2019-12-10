// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_merge_blocks.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationMergeBlocks::TransformationMergeBlocks(
    const spvtools::fuzz::protobufs::TransformationMergeBlocks& message)
    : message_(message) {}

TransformationMergeBlocks::TransformationMergeBlocks(uint32_t block_id) {
  message_.set_block_id(block_id);
}

bool TransformationMergeBlocks::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  auto second_block = fuzzerutil::MaybeFindBlock(context, message_.block_id());
  // The given block must exist.
  if (!second_block) {
    return false;
  }
  // The block must have just one predecessor.
  auto predecessors = context->cfg()->preds(second_block->id());
  if (predecessors.size() != 1) {
    return false;
  }
  auto first_block = context->cfg()->block(predecessors.at(0));

  // The predecessor must have just one successor.
  if (first_block->terminator()->opcode() != SpvOpBranch) {
    return false;
  }
  assert(first_block->terminator()->GetSingleWordInOperand(0) ==
             second_block->id() &&
         "Sole successor of predecessor should yield the same block");

  // The block's successor must not be used as a merge block or continue target.
  bool used_as_merge_block_or_continue_target;
  context->get_def_use_mgr()->WhileEachUse(
      second_block->id(),
      [&used_as_merge_block_or_continue_target](
          const opt::Instruction* use_instruction,
          uint32_t /*unused*/) -> bool {
        switch (use_instruction->opcode()) {
          case SpvOpLoopMerge:
          case SpvOpSelectionMerge:
            used_as_merge_block_or_continue_target = true;
            return false;
          default:
            break;
        }
        return true;
      });
  if (used_as_merge_block_or_continue_target) {
    return false;
  }

  // The block's successor must not start with OpPhi.
  bool successor_starts_with_op_phi = false;
  second_block->WhileEachPhiInst(
      [&successor_starts_with_op_phi](const opt::Instruction *
                                      /*unused*/) -> bool {
        successor_starts_with_op_phi = true;
        return false;
      });
  if (successor_starts_with_op_phi) {
    return false;
  }
  return true;
}

void TransformationMergeBlocks::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* /*unused*/) const {
  auto second_block = fuzzerutil::MaybeFindBlock(context, message_.block_id());
  auto first_block =
      context->cfg()->block(context->cfg()->preds(second_block->id()).at(0));
  assert(first_block->terminator()->opcode() == SpvOpBranch &&
         "The blocks to be merged must be separated by OpBranch");

  // Erase the terminator of the first block.
  for (auto inst_it = first_block->begin();; ++inst_it) {
    if (&*inst_it == first_block->terminator()) {
      inst_it.Erase();
      break;
    }
  }

  // Add clones of all instructions from the second block to the first block,
  // erasing them from the second block in the process.
  for (auto inst_it = second_block->begin(); inst_it != second_block->end();) {
    first_block->AddInstruction(
        std::unique_ptr<opt::Instruction>(inst_it->Clone(context)));
    inst_it = inst_it.Erase();
  }

  // Erase the second block from the module (as it is now unreachable).
  for (auto block_it = first_block->GetParent()->begin();; ++block_it) {
    if (&*block_it == second_block) {
      block_it.Erase();
      break;
    }
  }

  // Invalidate all analyses, since we have changed the module significantly.
  context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationMergeBlocks::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_merge_blocks() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
