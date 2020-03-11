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

#include "source/fuzz/transformation_add_dead_continue.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddDeadContinue::TransformationAddDeadContinue(
    const spvtools::fuzz::protobufs::TransformationAddDeadContinue& message)
    : message_(message) {}

TransformationAddDeadContinue::TransformationAddDeadContinue(
    uint32_t from_block, bool continue_condition_value,
    std::vector<uint32_t> phi_id) {
  message_.set_from_block(from_block);
  message_.set_continue_condition_value(continue_condition_value);
  for (auto id : phi_id) {
    message_.add_phi_id(id);
  }
}

bool TransformationAddDeadContinue::IsApplicable(
    opt::IRContext* context, const FactManager& /*unused*/) const {
  // First, we check that a constant with the same value as
  // |message_.continue_condition_value| is present.
  if (!fuzzerutil::MaybeGetBoolConstantId(
          context, message_.continue_condition_value())) {
    // The required constant is not present, so the transformation cannot be
    // applied.
    return false;
  }

  // Check that |message_.from_block| really is a block id.
  opt::BasicBlock* bb_from =
      fuzzerutil::MaybeFindBlock(context, message_.from_block());
  if (bb_from == nullptr) {
    return false;
  }

  // Check that |message_.from_block| ends with an unconditional branch.
  if (bb_from->terminator()->opcode() != SpvOpBranch) {
    // The block associated with the id does not end with an unconditional
    // branch.
    return false;
  }

  // Get the header for the innermost loop containing |message_.from_block|.
  // Because the structured CFG analysis does not regard a loop header as part
  // of the loop it heads, we check first whether bb_from is a loop header
  // before using the structured CFG analysis.
  auto loop_header = bb_from->IsLoopHeader()
                         ? message_.from_block()
                         : context->GetStructuredCFGAnalysis()->ContainingLoop(
                               message_.from_block());
  if (!loop_header) {
    return false;
  }

  auto continue_block = context->cfg()->block(loop_header)->ContinueBlockId();

  if (!fuzzerutil::BlockIsReachableInItsFunction(
          context, context->cfg()->block(continue_block))) {
    // If the loop's continue block is unreachable, we conservatively do not
    // allow adding a dead continue, to avoid the compilations that arise due to
    // the lack of sensible dominance information for unreachable blocks.
    return false;
  }

  if (fuzzerutil::BlockIsInLoopContinueConstruct(context, message_.from_block(),
                                                 loop_header)) {
    // We cannot jump to the continue target from the continue construct.
    return false;
  }

  if (context->GetStructuredCFGAnalysis()->IsMergeBlock(continue_block)) {
    // A branch straight to the continue target that is also a merge block might
    // break the property that a construct header must dominate its merge block
    // (if the merge block is reachable).
    return false;
  }

  // Check whether the data passed to extend OpPhi instructions is appropriate.
  assert(fuzzerutil::PhiIdsOkForNewEdge(context, bb_from,
                                        context->cfg()->block(continue_block),
                                        message_.phi_id()) &&
         "Provided ids for OpPhi instruction are invalid");

  // Adding the dead break is only valid if SPIR-V rules related to dominance
  // hold.  Rather than checking these rules explicitly, we defer to the
  // validator.  We make a clone of the module, apply the transformation to the
  // clone, and check whether the transformed clone is valid.
  //
  // In principle some of the above checks could be removed, with more reliance
  // being placed on the validator.  This should be revisited if we are sure
  // the validator is complete with respect to checking structured control flow
  // rules.
  auto cloned_context = fuzzerutil::CloneIRContext(context);
  ApplyImpl(cloned_context.get());
  return fuzzerutil::IsValid(cloned_context.get());
}

void TransformationAddDeadContinue::Apply(opt::IRContext* context,
                                          FactManager* /*unused*/) const {
  ApplyImpl(context);
  // Invalidate all analyses
  context->InvalidateAnalysesExceptFor(opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationAddDeadContinue::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_dead_continue() = message_;
  return result;
}

void TransformationAddDeadContinue::ApplyImpl(
    spvtools::opt::IRContext* context) const {
  auto bb_from = context->cfg()->block(message_.from_block());
  auto continue_block =
      bb_from->IsLoopHeader()
          ? bb_from->ContinueBlockId()
          : context->GetStructuredCFGAnalysis()->LoopContinueBlock(
                message_.from_block());
  assert(continue_block && "message_.from_block must be in a loop.");
  fuzzerutil::AddUnreachableEdgeAndUpdateOpPhis(
      context, bb_from, context->cfg()->block(continue_block),
      message_.continue_condition_value(), message_.phi_id());
}

}  // namespace fuzz
}  // namespace spvtools
