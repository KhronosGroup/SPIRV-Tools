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

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

namespace fuzzerutil {

bool IsFreshId(opt::IRContext* context, uint32_t id) {
  return !context->get_def_use_mgr()->GetDef(id);
}

void UpdateModuleIdBound(opt::IRContext* context, uint32_t id) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2541) consider the
  //  case where the maximum id bound is reached.
  context->module()->SetIdBound(
      std::max(context->module()->id_bound(), id + 1));
}

opt::BasicBlock* MaybeFindBlock(opt::IRContext* context,
                                uint32_t maybe_block_id) {
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

bool PhiIdsOkForNewEdge(
    opt::IRContext* context, opt::BasicBlock* bb_from, opt::BasicBlock* bb_to,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& phi_ids) {
  if (bb_from->IsSuccessor(bb_to)) {
    // There is already an edge from |from_block| to |to_block|, so there is
    // no need to extend OpPhi instructions.  Do not allow phi ids to be
    // present. This might turn out to be too strict; perhaps it would be OK
    // just to ignore the ids in this case.
    return phi_ids.empty();
  }
  // The edge would add a previously non-existent edge from |from_block| to
  // |to_block|, so we go through the given phi ids and check that they exactly
  // match the OpPhi instructions in |to_block|.
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
    if (phi_index == static_cast<uint32_t>(phi_ids.size())) {
      // Not enough phi ids have been provided to account for the OpPhi
      // instructions.
      return false;
    }
    // Look for an instruction defining the next phi id.
    opt::Instruction* phi_extension =
        context->get_def_use_mgr()->GetDef(phi_ids[phi_index]);
    if (!phi_extension) {
      // The id given to extend this OpPhi does not exist.
      return false;
    }
    if (phi_extension->type_id() != inst.type_id()) {
      // The instruction given to extend this OpPhi either does not have a type
      // or its type does not match that of the OpPhi.
      return false;
    }

    if (context->get_instr_block(phi_extension)) {
      // The instruction defining the phi id has an associated block (i.e., it
      // is not a global value).  Check whether its definition dominates the
      // exit of |from_block|.
      auto dominator_analysis =
          context->GetDominatorAnalysis(bb_from->GetParent());
      if (!dominator_analysis->Dominates(phi_extension,
                                         bb_from->terminator())) {
        // The given id is no good as its definition does not dominate the exit
        // of |from_block|
        return false;
      }
    }
    phi_index++;
  }
  // Return false if not all of the ids for extending OpPhi instructions are
  // needed. This might turn out to be stricter than necessary; perhaps it would
  // be OK just to not use the ids in this case.
  return phi_index == static_cast<uint32_t>(phi_ids.size());
}

void AddUnreachableEdgeAndUpdateOpPhis(
    opt::IRContext* context, opt::BasicBlock* bb_from, opt::BasicBlock* bb_to,
    bool condition_value,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& phi_ids) {
  assert(PhiIdsOkForNewEdge(context, bb_from, bb_to, phi_ids) &&
         "Precondition on phi_ids is not satisfied");
  assert(bb_from->terminator()->opcode() == SpvOpBranch &&
         "Precondition on terminator of bb_from is not satisfied");

  // Get the id of the boolean constant to be used as the condition.
  opt::analysis::Bool bool_type;
  opt::analysis::BoolConstant bool_constant(
      context->get_type_mgr()->GetRegisteredType(&bool_type)->AsBool(),
      condition_value);
  uint32_t bool_id = context->get_constant_mgr()->FindDeclaredConstant(
      &bool_constant, context->get_type_mgr()->GetId(&bool_type));

  const bool from_to_edge_already_exists = bb_from->IsSuccessor(bb_to);
  auto successor = bb_from->terminator()->GetSingleWordInOperand(0);

  // Add the dead branch, by turning OpBranch into OpBranchConditional, and
  // ordering the targets depending on whether the given boolean corresponds to
  // true or false.
  bb_from->terminator()->SetOpcode(SpvOpBranchConditional);
  bb_from->terminator()->SetInOperands(
      {{SPV_OPERAND_TYPE_ID, {bool_id}},
       {SPV_OPERAND_TYPE_ID, {condition_value ? successor : bb_to->id()}},
       {SPV_OPERAND_TYPE_ID, {condition_value ? bb_to->id() : successor}}});

  // Update OpPhi instructions in the target block if this branch adds a
  // previously non-existent edge from source to target.
  if (!from_to_edge_already_exists) {
    uint32_t phi_index = 0;
    for (auto& inst : *bb_to) {
      if (inst.opcode() != SpvOpPhi) {
        break;
      }
      assert(phi_index < static_cast<uint32_t>(phi_ids.size()) &&
             "There should be exactly one phi id per OpPhi instruction.");
      inst.AddOperand({SPV_OPERAND_TYPE_ID, {phi_ids[phi_index]}});
      inst.AddOperand({SPV_OPERAND_TYPE_ID, {bb_from->id()}});
      phi_index++;
    }
    assert(phi_index == static_cast<uint32_t>(phi_ids.size()) &&
           "There should be exactly one phi id per OpPhi instruction.");
  }
}

bool BlockIsInLoopContinueConstruct(opt::IRContext* context, uint32_t block_id,
                                    uint32_t maybe_loop_header_id) {
  // We deem a block to be part of a loop's continue construct if the loop's
  // continue target dominates the block.
  auto containing_construct_block = context->cfg()->block(maybe_loop_header_id);
  if (containing_construct_block->IsLoopHeader()) {
    auto continue_target = containing_construct_block->ContinueBlockId();
    if (context->GetDominatorAnalysis(containing_construct_block->GetParent())
            ->Dominates(continue_target, block_id)) {
      return true;
    }
  }
  return false;
}

opt::BasicBlock::iterator GetIteratorForBaseInstructionAndOffset(
    opt::BasicBlock* block, const opt::Instruction* base_inst,
    uint32_t offset) {
  // The cases where |base_inst| is the block's label, vs. inside the block,
  // are dealt with separately.
  if (base_inst == block->GetLabelInst()) {
    // |base_inst| is the block's label.
    if (offset == 0) {
      // We cannot return an iterator to the block's label.
      return block->end();
    }
    // Conceptually, the first instruction in the block is [label + 1].
    // We thus start from 1 when applying the offset.
    auto inst_it = block->begin();
    for (uint32_t i = 1; i < offset && inst_it != block->end(); i++) {
      ++inst_it;
    }
    // This is either the desired instruction, or the end of the block.
    return inst_it;
  }
  // |base_inst| is inside the block.
  for (auto inst_it = block->begin(); inst_it != block->end(); ++inst_it) {
    if (base_inst == &*inst_it) {
      // We have found the base instruction; we now apply the offset.
      for (uint32_t i = 0; i < offset && inst_it != block->end(); i++) {
        ++inst_it;
      }
      // This is either the desired instruction, or the end of the block.
      return inst_it;
    }
  }
  assert(false && "The base instruction was not found.");
  return nullptr;
}

// Helper method to figure out whether there is a back-edge-free path from
// |source| to |dest|.
bool CanReachWithoutTraversingLoopBackEdge(opt::IRContext* context,
                                           opt::BasicBlock* source,
                                           opt::BasicBlock* dest) {
  // First, check that the destination block occurs after the source block in
  // the function.  If it does not, then it is certainly not possible for
  // |source| to reach |dest| without going through a loop back edge.
  bool dest_after_source = false;
  for (auto iterator = source->GetParent()->FindBlock(source->id());
       iterator != source->GetParent()->end(); ++iterator) {
    if (&*iterator == dest) {
      dest_after_source = true;
      break;
    }
  }

  if (!dest_after_source) {
    return false;
  }

  // Search through the blocks that can be reached from |source|, looking for
  // |dest|.
  // TODO(afd): consider writing a "WhileEach" version of
  //  "ForEachBlockInPostOrder" to allow for an early exit when a
  //  match is found.
  bool found = false;
  context->cfg()->ForEachBlockInPostOrder(
      source, [dest, &found](opt::BasicBlock* visited) {
        if (visited == dest) {
          found = true;
        }
      });
  assert((source != dest || found) &&
         "A block should be deemed to reach itself.");
  return found;
}

bool NewEdgeLeavingConstructBodyRespectsUseDefDominance(
    opt::IRContext* context, opt::BasicBlock* bb_from, opt::BasicBlock* bb_to,
    opt::BasicBlock* construct_merge_block) {
  auto enclosing_function = bb_from->GetParent();
  auto dominator_analysis = context->GetDominatorAnalysis(enclosing_function);

  // We are concerned about definitions occurring in the construct being
  // bypassed by an edge from |bb_from| to |bb_to|. We thus need to investigate
  // all block from after |bb_from| until |construct_merge_block|.

  auto merge_block_iterator =
      enclosing_function->FindBlock(construct_merge_block->id());
  auto block_it = enclosing_function->FindBlock(bb_from->id());
  assert(block_it != enclosing_function->end());
  ++block_it;

  // Iterate through blocks in module order until we find the merge block.  Due
  // scenarios where the merge block may be unreachable, we cannot guarantee
  // that we will find it, so check for the end of the function to be safe.
  // (Stopping at the merge block is just an optimization.)
  for (; block_it != enclosing_function->end() &&
         block_it != merge_block_iterator;
       ++block_it) {
    // We are interested in looking at troublesome definitions in this block.
    opt::BasicBlock* defining_block = &*block_it;

    // If it's not possible to reach |defining_block| from |bb_from| then the
    // edge to |bb_to| from |bb_from| has no impact on |defining_block|.
    if (!CanReachWithoutTraversingLoopBackEdge(context, bb_from,
                                               defining_block)) {
      continue;
    }

    // If |bb_to| dominates |defining_block|, the definition will not be
    // impacted by a new edge to |bb_to|.
    if (dominator_analysis->Dominates(bb_to, defining_block)) {
      continue;
    }

    // Look through all the uses of all instructions in |defining_block|.
    for (auto& inst : *defining_block) {
      if (!context->get_def_use_mgr()->WhileEachUse(
              &inst,
              [context, bb_to](opt::Instruction* user,
                               uint32_t operand_index) -> bool {
                // If this use is in an OpPhi, we need to check that dominance
                // of the relevant *parent* block is not spoiled.  Otherwise we
                // need to check that dominance of the block containing the use
                // is not spoiled.
                opt::BasicBlock* use_block_or_phi_parent =
                    user->opcode() == SpvOpPhi
                        ? context->cfg()->block(
                              user->GetSingleWordOperand(operand_index + 1))
                        : context->get_instr_block(user);

                // There might not be any relevant block, e.g. if the use is in
                // a decoration; in this case the new edge is unproblematic.
                if (use_block_or_phi_parent == nullptr) {
                  return true;
                }

                // If it is possible to get from |bb_to| to
                // |use_block_or_phi_parent| (without going through a loop
                // back-edge) then adding the edge would introduce a path
                // involving |bb_from|->|bb_to|->|use_block_or_phi_parent| that
                // does *not* go through |defining_block|; thus
                // |use_block_or_phi_parent| would not be dominated by
                // |defining_block|, which would be invalid.
                return !CanReachWithoutTraversingLoopBackEdge(
                    context, bb_to, use_block_or_phi_parent);

              })) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace fuzzerutil

}  // namespace fuzz
}  // namespace spvtools
