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
        uint32_t select_id, std::vector<uint32_t> new_block_ids) {
  message_.set_select_id(select_id);
  for (auto id : new_block_ids) {
    message_.add_new_block_id(id);
  }
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

  auto block = ir_context->get_instr_block(instruction);

  // If the block containing the instruction is a merge block, at least 3 fresh
  // ids are needed.
  if (fuzzerutil::IsMergeBlock(ir_context, block->id()) &&
      message_.new_block_id_size() < 3) {
    return false;
  }

  // In all cases, at least 2 fresh ids are needed.
  if (message_.new_block_id_size() < 2) {
    return false;
  }

  // Check that the new block ids are fresh and distinct.
  std::set<uint32_t> used_ids;
  for (uint32_t id : message_.new_block_id()) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(id, ir_context,
                                                      &used_ids)) {
      return false;
    }
  }

  // The block must be split around the OpSelect instruction. This means that
  // there cannot be an OpSampledImage instruction before OpSelect that is used
  // after it, because they are required to be in the same basic block.
  return !fuzzerutil::
      SplitBeforeInstructionSeparatesOpSampledImageDefinitionFromUse(
          block, instruction);
}

void TransformationReplaceOpSelectWithConditionalBranch::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation
TransformationReplaceOpSelectWithConditionalBranch::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_opselect_with_conditional_branch() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
