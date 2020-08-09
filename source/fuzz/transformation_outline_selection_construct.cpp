// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_outline_selection_construct.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationOutlineSelectionConstruct::
    TransformationOutlineSelectionConstruct(
        const protobufs::TransformationOutlineSelectionConstruct& message)
    : message_(message) {}

TransformationOutlineSelectionConstruct::
    TransformationOutlineSelectionConstruct(uint32_t new_header_block_id,
                                            uint32_t new_merge_block_id) {
  message_.set_new_header_block_id(new_header_block_id);
  message_.set_new_merge_block_id(new_merge_block_id);
}

bool TransformationOutlineSelectionConstruct::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Check that it is possible to outline a region of blocks without breaking
  // domination and structured control flow rules.
  if (!IsApplicableToBlockRange(ir_context, message_.new_header_block_id(),
                                message_.new_merge_block_id())) {
    return false;
  }

  // There must exist an irrelevant boolean constant to be used as a condition
  // in the OpBranchConditional instruction.
  return fuzzerutil::MaybeGetBoolConstant(ir_context, transformation_context,
                                          true, true) != 0;
}

void TransformationOutlineSelectionConstruct::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto* new_header_block =
      ir_context->cfg()->block(message_.new_header_block_id());
  assert(new_header_block->terminator()->opcode() == SpvOpBranch &&
         "Should condition should've been checked in the IsApplicable");

  const auto successor_id =
      new_header_block->terminator()->GetSingleWordInOperand(0);

  // Change |entry_block|'s terminator to |OpBranchConditional|.
  new_header_block->terminator()->SetOpcode(SpvOpBranchConditional);
  new_header_block->terminator()->SetInOperands(
      {{SPV_OPERAND_TYPE_ID,
        {fuzzerutil::MaybeGetBoolConstant(ir_context, *transformation_context,
                                          true, true)}},
       {SPV_OPERAND_TYPE_ID, {successor_id}},
       {SPV_OPERAND_TYPE_ID, {successor_id}}});

  // Insert OpSelectionMerge before the terminator.
  new_header_block->terminator()->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, SpvOpSelectionMerge, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.new_merge_block_id()}},
          {SPV_OPERAND_TYPE_SELECTION_CONTROL,
           {SpvSelectionControlMaskNone}}}));

  // We've change the module so we must invalidate analyses.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationOutlineSelectionConstruct::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_outline_selection_construct() = message_;
  return result;
}

bool TransformationOutlineSelectionConstruct::IsApplicableToBlockRange(
    opt::IRContext* ir_context, uint32_t header_block_candidate_id,
    uint32_t merge_block_candidate_id) {
  // Check that |header_block_candidate_id| and |merge_block_candidate_id| are
  // valid.
  for (auto block_id : {header_block_candidate_id, merge_block_candidate_id}) {
    const auto* label_inst = ir_context->get_def_use_mgr()->GetDef(block_id);
    if (!label_inst || label_inst->opcode() != SpvOpLabel) {
      return false;
    }
  }

  const auto* header_block_candidate =
      ir_context->cfg()->block(header_block_candidate_id);
  const auto* merge_block_candidate =
      ir_context->cfg()->block(merge_block_candidate_id);

  // |header_block_candidate| and |merge_block_candidate| must be from the same
  // function.
  if (header_block_candidate->GetParent() !=
      merge_block_candidate->GetParent()) {
    return false;
  }

  const auto* dominator_analysis =
      ir_context->GetDominatorAnalysis(header_block_candidate->GetParent());
  const auto* postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(header_block_candidate->GetParent());

  if (!dominator_analysis->Dominates(header_block_candidate,
                                     merge_block_candidate) ||
      !postdominator_analysis->Dominates(merge_block_candidate,
                                         header_block_candidate)) {
    return false;
  }

  // |header_block_candidate| can't be a header since we are about to make it
  // one.
  if (header_block_candidate->GetMergeInst()) {
    return false;
  }

  // |header_block_candidate| must have an OpBranch terminator.
  if (header_block_candidate->terminator()->opcode() != SpvOpBranch) {
    return false;
  }

  // Every header block must have a unique merge block. This rule will be
  // violated if we make |merge_block_candidate| a merge block of a newly
  // created header.
  if (ir_context->GetStructuredCFGAnalysis()->IsMergeBlock(
          merge_block_candidate_id)) {
    return false;
  }

  // Compute a set of blocks, dominated by |header_block_candidate| and
  // postdominated by |merge_block_candidate|.
  std::unordered_set<uint32_t> outlined_blocks;
  for (const auto& block : *header_block_candidate->GetParent()) {
    if (dominator_analysis->Dominates(header_block_candidate, &block) &&
        postdominator_analysis->Dominates(merge_block_candidate, &block)) {
      outlined_blocks.insert(block.id());
    }
  }

  // |header_block_candidate| can be a merge block of some construct.
  // |merge_block_candidate| can be a header block of some construct.
  outlined_blocks.erase(header_block_candidate_id);
  outlined_blocks.erase(merge_block_candidate_id);

  // Make sure that every construct is either completely included in the
  // |outlined_blocks| or completely excluded out of it.
  for (const auto& block : *header_block_candidate->GetParent()) {
    if (const auto* merge_inst = block.GetMergeInst()) {
      // |block| is a header block - make sure it and its merge block either are
      // both outlined or both not.
      auto merge_block_id = merge_inst->GetSingleWordInOperand(0);
      if (dominator_analysis->IsReachable(merge_block_id) &&
          outlined_blocks.count(merge_block_id) !=
              outlined_blocks.count(block.id())) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace fuzz
}  // namespace spvtools
