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

#include "source/fuzz/fuzzer_pass_outline_selection_construct.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_outline_selection_construct.h"
#include "source/fuzz/transformation_split_block.h"

namespace spvtools {
namespace fuzz {

FuzzerPassOutlineSelectionConstruct::FuzzerPassOutlineSelectionConstruct(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassOutlineSelectionConstruct::~FuzzerPassOutlineSelectionConstruct() =
    default;

void FuzzerPassOutlineSelectionConstruct::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfOutliningSelectionConstruct())) {
      continue;
    }

    // It is easier to select an element at random from a vector than from an
    // instruction list.
    std::vector<opt::BasicBlock*> header_block_candidates;
    for (auto& block : function) {
      header_block_candidates.push_back(&block);
    }

    if (header_block_candidates.empty()) {
      continue;
    }

    // Try to get a header block candidate that will increase the chances of the
    // transformation being applicable.
    auto* header_block_candidate = MaybeGetHeaderBlockCandidate(
        header_block_candidates[GetFuzzerContext()->RandomIndex(
            header_block_candidates)]);
    if (!header_block_candidate) {
      continue;
    }

    std::vector<opt::BasicBlock*> merge_block_candidates;
    for (auto& block : function) {
      if (GetIRContext()->GetDominatorAnalysis(&function)->Dominates(
              header_block_candidate, &block) &&
          GetIRContext()->GetPostDominatorAnalysis(&function)->Dominates(
              &block, header_block_candidate)) {
        merge_block_candidates.push_back(&block);
      }
    }

    if (merge_block_candidates.empty()) {
      continue;
    }

    // Try to get a merge block candidate that will increase the chances of the
    // transformation being applicable.
    auto* merge_block_candidate = MaybeGetMergeBlockCandidate(
        merge_block_candidates[GetFuzzerContext()->RandomIndex(
            merge_block_candidates)]);
    if (!merge_block_candidate) {
      continue;
    }

    if (!TransformationOutlineSelectionConstruct::IsApplicableToBlockRange(
            GetIRContext(), header_block_candidate->id(),
            merge_block_candidate->id())) {
      continue;
    }

    // This boolean constant will be used as a condition for the
    // OpBranchConditional instruction. We mark it as irrelevant to be able to
    // replace it with a more interesting value later.
    FindOrCreateBoolConstant(true, true);

    ApplyTransformation(TransformationOutlineSelectionConstruct(
        header_block_candidate->id(), merge_block_candidate->id()));
  }
}

opt::BasicBlock*
FuzzerPassOutlineSelectionConstruct::MaybeGetHeaderBlockCandidate(
    opt::BasicBlock* header_block_candidate) {
  // Try to create a preheader if |header_block_candidate| is a loop header.
  if (header_block_candidate->GetLoopMergeInst()) {
    // GetOrCreateSimpleLoopPreheader only supports reachable blocks.
    return GetIRContext()->cfg()->preds(header_block_candidate->id()).size() ==
                   1
               ? nullptr
               : GetOrCreateSimpleLoopPreheader(header_block_candidate->id());
  }

  // Try to split |header_block_candidate| if it's already a header block.
  if (header_block_candidate->GetMergeInst()) {
    auto* split_before_inst = &*header_block_candidate->begin();
    while (split_before_inst->opcode() == SpvOpVariable ||
           split_before_inst->opcode() == SpvOpPhi) {
      split_before_inst = split_before_inst->NextNode();
    }

    ApplyTransformation(TransformationSplitBlock(
        MakeInstructionDescriptor(GetIRContext(), split_before_inst),
        GetFuzzerContext()->GetFreshId()));
  }

  return header_block_candidate;
}

opt::BasicBlock*
FuzzerPassOutlineSelectionConstruct::MaybeGetMergeBlockCandidate(
    opt::BasicBlock* merge_block_candidate) {
  // If |merge_block_candidate| is a merge block of some construct, try to split
  // it and return a newly created block.
  if (GetIRContext()->GetStructuredCFGAnalysis()->IsMergeBlock(
          merge_block_candidate->id())) {
    auto* split_before_inst = &*merge_block_candidate->begin();
    while (split_before_inst->opcode() == SpvOpPhi) {
      split_before_inst = split_before_inst->NextNode();
    }

    auto new_block_id = GetFuzzerContext()->GetFreshId();
    ApplyTransformation(TransformationSplitBlock(
        MakeInstructionDescriptor(GetIRContext(), split_before_inst),
        new_block_id));
    return GetIRContext()->cfg()->block(new_block_id);
  }

  return merge_block_candidate;
}

}  // namespace fuzz
}  // namespace spvtools
