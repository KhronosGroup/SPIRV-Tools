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
    std::map<protobufs::InstructionDescriptor, std::vector<uint32_t>>
        instructions_to_fresh_ids) {
  message_.set_header_block_id(header_block_id);
  for (auto const& pair : instructions_to_fresh_ids) {
    protobufs::InstructionUint32ListPair mapping;
    *mapping.mutable_instruction_descriptor() = pair.first;
    for (auto id : pair.second) {
      mapping.add_id(id);
    }
    *message_.add_instruction_to_fresh_ids() = mapping;
  }
}

bool TransformationFlattenConditionalBranch::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* transformation_context */) const {
  // |message_.header_block_id| must refer to a block label.
  auto header_block = ir_context->cfg()->block(message_.header_block_id());
  if (!header_block) {
    return false;
  }

  // The header block must end with an OpBranchConditional instruction.
  if (header_block->terminator()->opcode() != SpvOpBranchConditional) {
    return false;
  }

  // Find the merge block.
  uint32_t merge_block_id = header_block->GetLabel()->GetSingleWordInOperand(0);

  // Find the first block where flow converges (it is not necessarily the merge
  // block).
  uint32_t convergence_block_id = merge_block_id;
  while (ir_context->cfg()->preds(convergence_block_id).size() == 1) {
    if (convergence_block_id == message_.header_block_id()) {
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
    if (!dominator_analysis->Dominates(message_.header_block_id(), block_id) ||
        !postdominator_analysis->Dominates(convergence_block_id, block_id)) {
      return false;
    }

    auto block = ir_context->cfg()->block(block_id);

    // The block must end with OpBranch, because inner constructs are not
    // allowed.
    if (block->terminator()->opcode() != SpvOpBranch) {
      return false;
    }

    // The block must not have a merge instruction.
    if (block->GetMergeInst()) {
      return false;
    }

    // Get the mapping from instruction descriptors to the fresh ids available
    // for it.
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::vector<uint32_t>>
        instructions_to_num_fresh_ids;
    for (auto pair : message_.instruction_to_fresh_ids()) {
      std::vector<uint32_t> fresh_ids;
      for (uint32_t id : pair.id()) {
        fresh_ids.push_back(id);
      }
      instructions_to_num_fresh_ids.emplace(
          TupleFromInstructionDescriptor(pair.instruction_descriptor()),
          fresh_ids);
    }
    // Keep track of the fresh ids used.
    std::set<uint32_t> used_fresh_ids;

    // Check the instructions in the block
    bool all_instructions_compatible =
        block->WhileEachInst([&ir_context, &instructions_to_num_fresh_ids,
                              &used_fresh_ids](opt::Instruction* instruction) {
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
            auto instruction_descriptor =
                MakeInstructionDescriptor(ir_context, instruction);

            std::tuple<uint32_t, uint32_t, uint32_t> inst_desc_tuple =
                TupleFromInstructionDescriptor(instruction_descriptor);

            if (instructions_to_num_fresh_ids.count(inst_desc_tuple) == 0) {
              // There is no mapping from this instruction to a list of fresh
              // ids.
              return false;
            }

            std::vector<uint32_t> fresh_ids =
                instructions_to_num_fresh_ids[inst_desc_tuple];

            // The ids must all be distinct and unused.
            for (uint32_t fresh_id : fresh_ids) {
              if (!CheckIdIsFreshAndNotUsedByThisTransformation(
                      fresh_id, ir_context, &used_fresh_ids)) {
                return false;
              }
            }

            // If the instruction is OpStore, there must be at least 2 fresh
            // ids.
            if (instruction->opcode() == SpvOpStore && fresh_ids.size() < 2) {
              return false;
            }

            // If the instruction is OpLoad or OpFunctionCall, there must be at
            // least 5 fresh ids.
            if ((instruction->opcode() == SpvOpLoad ||
                 instruction->opcode() == SpvOpFunctionCall) &&
                fresh_ids.size() < 5) {
              return false;
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
