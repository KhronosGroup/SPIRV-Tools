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
    for (auto id : overflow_ids) {
      message_.add_overflow_id(id);
    }
  }
}

bool TransformationFlattenConditionalBranch::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  uint32_t header_block_id = message_.header_block_id();

  // |header_block_id| must refer to a block label.
  {
    auto label = ir_context->get_def_use_mgr()->GetDef(header_block_id);
    if (label->opcode() != SpvOpLabel) {
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
      // There is a chain of blocks with one predecessors from the header block
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
      instructions_to_fresh_ids.emplace(instruction, fresh_ids);
    }
  }
  // Keep track of the fresh ids used.
  std::set<uint32_t> used_fresh_ids;

  // Keep track of the number of overflow ids used.
  uint32_t overflow_ids_used = 0;

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

    // Check the instructions in the block
    bool all_instructions_compatible = block->WhileEachInst(
        [this, &ir_context, &instructions_to_fresh_ids, &used_fresh_ids,
         &overflow_ids_used](opt::Instruction* instruction) {
          // The instruction cannot be an atomic or barrier instruction
          if (instruction->IsAtomicOp() ||
              instruction->opcode() == SpvOpControlBarrier ||
              instruction->opcode() == SpvOpMemoryBarrier ||
              instruction->opcode() == SpvOpNamedBarrierInitialize ||
              instruction->opcode() == SpvOpMemoryNamedBarrier ||
              instruction->opcode() == SpvOpTypeNamedBarrier) {
            return false;
          }

          // If the instruction is a load, store or function call, there must be
          // a mapping from the corresponding instruction descriptor to a list
          // of fresh ids.
          if (instruction->opcode() == SpvOpLoad ||
              instruction->opcode() == SpvOpStore ||
              instruction->opcode() == SpvOpFunctionCall) {
            // Keep a vector of the fresh ids needed by this instruction.
            std::vector<uint32_t> fresh_ids;
            uint32_t overflow_ids_needed;

            // Initialise overflow_ids_needed to the total number of ids needed.
            switch (instruction->opcode()) {
              case SpvOpStore:
                overflow_ids_needed = 2;
                break;
              case SpvOpLoad:
              case SpvOpFunctionCall:
                overflow_ids_needed = 5;
                break;
              default:
                assert(false && "This statement should not be reachable.");
                return false;
            }

            if (instructions_to_fresh_ids.count(instruction) != 0) {
              // There is a mapping from this instruction to a list of fresh
              // ids.

              fresh_ids = instructions_to_fresh_ids[instruction];
              // We can deduct the number of fresh ids specific to this
              // instruction from the number of overflow ids needed.
              overflow_ids_needed =
                  fresh_ids.size() > overflow_ids_needed
                      ? 0
                      : overflow_ids_needed - (uint32_t)fresh_ids.size();
            }

            // We need |overflow_ids_needed| overflow ids
            if (overflow_ids_used + overflow_ids_needed >
                (uint32_t)message_.overflow_id_size()) {
              return false;
            }

            // Add the overflow ids needed to fresh_ids
            for (; overflow_ids_needed > 0; overflow_ids_needed--) {
              fresh_ids.push_back(message_.overflow_id(overflow_ids_used));
              overflow_ids_used++;
            }

            // The ids must all be distinct and unused.
            for (uint32_t fresh_id : fresh_ids) {
              if (!CheckIdIsFreshAndNotUsedByThisTransformation(
                      fresh_id, ir_context, &used_fresh_ids)) {
                return false;
              }
            }
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
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationFlattenConditionalBranch::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_flatten_conditional_branch() = message_;
  return result;
}
}  // namespace fuzz
}  // namespace spvtools
