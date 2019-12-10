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
  auto first_block = fuzzerutil::MaybeFindBlock(context, message_.block_id());
  // The given block must exist.
  if (!first_block) {
    return false;
  }
  // The block must have just one successor.
  if (first_block->terminator()->opcode() != SpvOpBranch) {
    return false;
  }
  // The block's successor must have just one predecessor.
  auto successor = context->cfg()->block(first_block->terminator()
          ->GetSingleWordInOperand(0));
  if (context->cfg()->preds(successor->id()).size() != 1) {
    return false;
  }

  // The block's successor must not be used as a merge block or continue target.
  bool used_as_merge_block_or_continue_target;
  context->get_def_use_mgr()->WhileEachUse(successor->id(),
          [&used_as_merge_block_or_continue_target](const opt::Instruction*
          use_instruction, uint32_t /*unused*/) -> bool {
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
  successor->WhileEachPhiInst([&successor_starts_with_op_phi](const
  opt::Instruction* /*unused*/)
  ->
  bool {
    successor_starts_with_op_phi = true;
    return false;
  });
  if (successor_starts_with_op_phi) {
    return false;
  }
  return true;
}

void TransformationMergeBlocks::Apply(
    opt::IRContext* /*context*/, spvtools::fuzz::FactManager* /*unused*/) const {
  assert(false && "Not implemented yet");
}

protobufs::Transformation TransformationMergeBlocks::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_merge_blocks() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
