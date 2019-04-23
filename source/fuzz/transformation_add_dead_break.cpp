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

#include "source/fuzz/transformation_add_dead_break.h"
#include "source/opt/basic_block.h"
#include "source/opt/ir_context.h"
#include "source/opt/struct_cfg_analysis.h"

namespace spvtools {
namespace fuzz {

using opt::BasicBlock;
using opt::IRContext;
using opt::Instruction;

BasicBlock* TransformationAddDeadBreak::MaybeFindBlock(
    IRContext* context, uint32_t maybe_block_id) {
  auto inst = context->get_def_use_mgr()->GetDef(maybe_block_id);
  if (inst == nullptr) {
    // No instruction defining this id was found.
    return nullptr;
  }
  if (inst->opcode() != SpvOpLabel) {
    // The instruction defining the id is not a label, so it cannot be a block
    // id.
    return nullptr;
  }
  return context->cfg()->block(maybe_block_id);
}

bool TransformationAddDeadBreak::PhiIdsOk(IRContext* context,
                                          BasicBlock* bb_from,
                                          BasicBlock* bb_to) {
  if (bb_from->IsSuccessor(bb_to)) {
    // There is already an edge from |from_block_| to |to_block_|, so there is
    // no need to extend OpPhi instructions.  Do not allow phi ids to be
    // present. This might turn out to be too strict; perhaps it would be OK
    // just to ignore the ids in this case.
    return phi_ids_.empty();
  }
  // The break would add a previously non-existent edge from |from_block_| to
  // |to_block_|, so we go through the given phi ids and check that they exactly
  // match the OpPhi instructions in |to_block_|.
  uint32_t phi_index = 0;
  // An explicit loop, rather than applying a lambda to each OpPhi in |bb_to|,
  // makes sense here because we need to increment |phi_index| for each OpPhi
  // instruction.
  for (auto& inst : *bb_to) {
    if (inst.opcode() != SpvOpPhi) {
      // The OpPhi instructions all occur at the start of the block; if we find
      // a non-OpPhi then we have seen them all.
      break;
    }
    if (phi_index == phi_ids_.size()) {
      // Not enough phi ids have been provided to account for the OpPhi
      // instructions.
      return false;
    }
    // Look for an instruction defining the next phi id.
    Instruction* phi_extension =
        context->get_def_use_mgr()->GetDef(phi_ids_[phi_index]);
    if (!phi_extension) {
      // The id given to extend this OpPhi does not exist.
      return false;
    }
    if (phi_extension->type_id() != inst.type_id()) {
      // The instruction given to extend this OpPhi either does not have a type
      // or its type does not match that of the OpPhi.
      return false;
    }
    // Check whether the given id corresponds to a constant.  If so, that's
    // fine.
    bool found = false;
    for (auto& type_value_inst : context->module()->types_values()) {
      if (&type_value_inst == phi_extension) {
        found = true;
      }
    }
    if (!found) {
      // The id did not correspond to a constant, so it must come from an
      // instruction in the function.  Check whether its definition dominates
      // the exit of |from_block_|.
      auto dominator_analysis =
          context->GetDominatorAnalysis(bb_from->GetParent());
      if (!dominator_analysis->Dominates(phi_extension,
                                         bb_from->terminator())) {
        // The given id is no good as its definition does not dominate the exit
        // of |from_block_|
        return false;
      }
    }
    phi_index++;
  }
  // Reject the transformation if not all of the ids for extending OpPhi
  // instructions are needed. This might turn out to be stricter than necessary;
  // perhaps it would be OK just to not use the ids in this case.
  return phi_index == phi_ids_.size();
}

bool TransformationAddDeadBreak::FromBlockIsInLoopContinueConstruct(
    IRContext* context, uint32_t maybe_loop_header) {
  // We deem a block to be part of a loop's continue construct if the loop's
  // continue target dominates the block.
  auto containing_construct_block = context->cfg()->block(maybe_loop_header);
  if (containing_construct_block->IsLoopHeader()) {
    auto continue_target = containing_construct_block->ContinueBlockId();
    if (context->GetDominatorAnalysis(containing_construct_block->GetParent())
            ->Dominates(continue_target, from_block_)) {
      return true;
    }
  }
  return false;
}

bool TransformationAddDeadBreak::AddingBreakRespectsStructuredControlFlow(
    IRContext* context, BasicBlock* bb_from) {
  // Look at the structured control flow associated with |from_block_| and
  // check whether it is contained in an appropriate construct with merge id
  // |to_block_| such that a break from |from_block_| to |to_block_| is legal.

  // There are three legal cases to consider:
  // (1) |from_block_| is a loop header and |to_block_| is its merge
  // (2) |from_block_| is a non-header node of a construct, and |to_block_|
  //     is the merge for that construct
  // (3) |from_block_| is a non-header node of a selection construct, and
  //     |to_block_| is the merge for the innermost loop containing
  //     |from_block_|
  //
  // The reason we need to distinguish between cases (1) and (2) is that the
  // structured CFG analysis does not deem a header to be part of the construct
  // that it heads.

  // Consider case (1)
  if (bb_from->IsLoopHeader()) {
    // Case (1) holds if |to_block_| is the merge block for the loop;
    // otherwise no case holds
    return bb_from->MergeBlockId() == to_block_;
  }

  // Both cases (2) and (3) require that |from_block_| is inside some
  // structured control flow construct.

  auto containing_construct =
      context->GetStructuredCFGAnalysis()->ContainingConstruct(from_block_);
  if (!containing_construct) {
    // |from_block_| is not in a construct from which we can break.
    return false;
  }

  // Consider case (2)
  if (to_block_ ==
      context->cfg()->block(containing_construct)->MergeBlockId()) {
    // This looks like an instance of case (2).
    // However, the structured CFG analysis regards the continue construct of a
    // loop as part of the loop, but it is not legal to jump from a loop's
    // continue construct to the loop's merge, so we need to check for this
    // case.
    return !FromBlockIsInLoopContinueConstruct(context, containing_construct);
  }

  // Case (3) holds if and only if |to_block_| is the merge block for this
  // innermost loop that contains |from_block_|
  auto containing_loop_header =
      context->GetStructuredCFGAnalysis()->ContainingLoop(from_block_);
  if (containing_loop_header &&
      to_block_ ==
          context->cfg()->block(containing_loop_header)->MergeBlockId()) {
    return !FromBlockIsInLoopContinueConstruct(context, containing_loop_header);
  }
  return false;
}

bool TransformationAddDeadBreak::IsApplicable(IRContext* context) {
  // First, we check that a constant with the same value as
  // |break_condition_value_| is present.
  opt::analysis::Bool bool_type;
  auto registered_bool_type =
      context->get_type_mgr()->GetRegisteredType(&bool_type);
  if (!registered_bool_type) {
    return false;
  }
  opt::analysis::BoolConstant bool_constant(registered_bool_type->AsBool(),
                                            break_condition_value_);
  if (!context->get_constant_mgr()->FindConstant(&bool_constant)) {
    // The required constant is not present, so the transformation cannot be
    // applied.
    return false;
  }

  // Check that |from_block_| and |to_block_| really are block ids
  BasicBlock* bb_from = MaybeFindBlock(context, from_block_);
  if (bb_from == nullptr) {
    return false;
  }
  BasicBlock* bb_to = MaybeFindBlock(context, to_block_);
  if (bb_to == nullptr) {
    return false;
  }

  // Check that |from_block_| ends with an unconditional branch.
  if (bb_from->terminator()->opcode() != SpvOpBranch) {
    // The block associated with the id does not end with an unconditional
    // branch.
    return false;
  }

  assert(bb_from != nullptr &&
         "We should have found a block if this line of code is reached.");
  assert(
      bb_from->id() == from_block_ &&
      "The id of the block we found should match the source id for the break.");
  assert(bb_to != nullptr &&
         "We should have found a block if this line of code is reached.");
  assert(
      bb_to->id() == to_block_ &&
      "The id of the block we found should match the target id for the break.");

  // Check whether the data passed to extend OpPhi instructions is appropriate.
  if (!PhiIdsOk(context, bb_from, bb_to)) {
    return false;
  }

  // Finally, check that adding the break would respect the rules of structured
  // control flow.
  return AddingBreakRespectsStructuredControlFlow(context, bb_from);
}

void TransformationAddDeadBreak::Apply(IRContext* context) {
  // Get the id of the boolean constant to be used as the break condition.
  opt::analysis::Bool bool_type;
  opt::analysis::BoolConstant bool_constant(
      context->get_type_mgr()->GetRegisteredType(&bool_type)->AsBool(),
      break_condition_value_);
  uint32_t bool_id = context->get_constant_mgr()->FindDeclaredConstant(
      &bool_constant, context->get_type_mgr()->GetId(&bool_type));

  auto bb_from = context->cfg()->block(from_block_);
  auto bb_to = context->cfg()->block(to_block_);
  const bool from_to_edge_already_exists = bb_from->IsSuccessor(bb_to);
  auto successor = bb_from->terminator()->GetSingleWordInOperand(0);
  assert(bb_from->terminator()->opcode() == SpvOpBranch &&
         "Precondition for the transformation requires that the source block "
         "ends with OpBranch");

  // Add the dead break, by turning OpBranch into OpBranchConditional, and
  // ordering the targets depending on whether the given boolean corresponds to
  // true or false.
  bb_from->terminator()->SetOpcode(SpvOpBranchConditional);
  bb_from->terminator()->SetInOperands(
      {{SPV_OPERAND_TYPE_ID, {bool_id}},
       {SPV_OPERAND_TYPE_ID, {break_condition_value_ ? successor : to_block_}},
       {SPV_OPERAND_TYPE_ID,
        {break_condition_value_ ? to_block_ : successor}}});

  // Update OpPhi instructions in the target block if this break adds a
  // previously non-existent edge from source to target.
  if (!from_to_edge_already_exists) {
    uint32_t phi_index = 0;
    for (auto& inst : *bb_to) {
      if (inst.opcode() != SpvOpPhi) {
        break;
      }
      assert(phi_index < phi_ids_.size() &&
             "There should be exactly one phi id per OpPhi instruction.");
      inst.AddOperand({SPV_OPERAND_TYPE_ID, {phi_ids_[phi_index]}});
      inst.AddOperand({SPV_OPERAND_TYPE_ID, {from_block_}});
      phi_index++;
    }
    assert(phi_index == phi_ids_.size() &&
           "There should be exactly one phi id per OpPhi instruction.");
  }

  // Invalidate all analyses
  context->InvalidateAnalysesExceptFor(IRContext::Analysis::kAnalysisNone);
}

TransformationAddDeadBreak::TransformationAddDeadBreak(
    const spvtools::fuzz::protobufs::TransformationAddDeadBreak& message)
    : from_block_(message.from_block()),
      to_block_(message.to_block()),
      break_condition_value_(message.break_condition_value()) {
  for (auto id : message.phi_ids()) {
    phi_ids_.push_back(id);
  }
}

protobufs::Transformation TransformationAddDeadBreak::ToMessage() {
  auto add_dead_break_message = new protobufs::TransformationAddDeadBreak;
  add_dead_break_message->set_break_condition_value(break_condition_value_);
  add_dead_break_message->set_from_block(from_block_);
  for (auto id : phi_ids_) {
    add_dead_break_message->add_phi_ids(id);
  }
  add_dead_break_message->set_to_block(to_block_);
  protobufs::Transformation result;
  result.set_allocated_add_dead_break(add_dead_break_message);
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
