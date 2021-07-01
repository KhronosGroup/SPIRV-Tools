// Copyright (c) 2021 Alastair F. Donaldson
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

#include "source/reduce/structured_construct_to_block_reduction_opportunity_finder.h"

#include <unordered_set>

#include "source/reduce/structured_construct_to_block_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

std::vector<std::unique_ptr<ReductionOpportunity>>
StructuredConstructToBlockReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context, uint32_t target_function) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  // Consider every function in the module.
  for (auto* function : GetTargetFunctions(context, target_function)) {
    // For every header block in the function, there is potentially a region of
    // blocks that could be collapsed.
    std::unordered_map<opt::BasicBlock*, std::unordered_set<opt::BasicBlock*>>
        regions;

    // Regions are identified using dominators and postdominators, so we compute
    // those for the function.
    auto* dominators = context->GetDominatorAnalysis(function);
    auto* postdominators = context->GetPostDominatorAnalysis(function);

    // Consider every block in the function.
    for (auto& block : *function) {
      // If a block has an unreachable predecessor then folding away a region in
      // which that block is contained gets complicated, so we ignore regions
      // that contain such blocks. We note whether this block suffers from this
      // problem.
      bool has_unreachable_predecessor =
          HasUnreachablePredecessor(block, context);

      // Look through all the regions we have identified so far to see whether
      // this block is part of a region, or spoils a region (by having an
      // unreachable predecessor).
      for (auto entry = regions.begin(); entry != regions.end();) {
        // |block| is in this region if it is dominated by the header,
        // post-dominated by the merge, and different from the merge.
        assert(&block != entry->first &&
               "The block should not be the region's header because we only "
               "make a region when we encounter its header.");
        if (entry->first->MergeBlockId() != block.id() &&
            dominators->Dominates(entry->first, &block) &&
            postdominators->Dominates(
                entry->first->GetMergeInst()->GetSingleWordInOperand(0),
                block.id())) {
          if (has_unreachable_predecessor) {
            // The block would be in this region, but it has an unreachable
            // predecessor. This spoils the region, so we remove it.
            entry = regions.erase(entry);
            continue;
          } else {
            // Add the block to the region.
            entry->second.insert(&block);
          }
        }
        ++entry;
      }
      if (block.MergeBlockIdIfAny() == 0) {
        // The block isn't a header, so it doesn't constitute a new region.
        continue;
      }
      if (!context->IsReachable(block)) {
        // The block isn't reachable, so it doesn't constitute a new region.
        continue;
      }
      auto* merge_block = context->cfg()->block(
          block.GetMergeInst()->GetSingleWordInOperand(0));
      if (!context->IsReachable(*merge_block)) {
        // The block's merge is unreachable, so it doesn't constitute a new
        // region.
        continue;
      }
      assert(dominators->Dominates(&block, merge_block) &&
             "The merge block is reachable, so the header must dominate it");
      if (!postdominators->Dominates(merge_block, &block)) {
        // The block is not post-dominated by its merge. This happens for
        // instance when there is a break from a conditional, or an early exit.
        // This also means that we don't add a region.
        continue;
      }
      // We have a reachable header block with a rechable merge that
      // postdominates the header: this means we have a new region.
      regions.emplace(&block, std::unordered_set<opt::BasicBlock*>());
    }

    // Now that we have found all the regions and blocks within them, we check
    // whether any region defines an id that is used outside the region. If this
    // is *not* the case, then we have an opportunity to collapse the region
    // down to its header block and merge block.
    for (auto& entry : regions) {
      if (DefinitionsRestrictedToRegion(*entry.first, entry.second, context)) {
        result.emplace_back(
            MakeUnique<StructuredConstructToBlockReductionOpportunity>(
                context, entry.first->id()));
      }
    }
  }
  return result;
}

bool StructuredConstructToBlockReductionOpportunityFinder::
    DefinitionsRestrictedToRegion(
        const opt::BasicBlock& header,
        const std::unordered_set<opt::BasicBlock*>& region,
        opt::IRContext* context) {
  for (auto& block : region) {
    if (!block->WhileEachInst(
            [context, &header, &region](opt::Instruction* inst) -> bool {
              if (inst->result_id() == 0) {
                return true;
              }
              if (!context->get_def_use_mgr()->WhileEachUse(
                      inst->result_id(),
                      [context, &header, &region](opt::Instruction* user,
                                                  uint32_t) -> bool {
                        auto user_block = context->get_instr_block(user);
                        if (user_block == &header &&
                            (user->opcode() == SpvOpLoopMerge ||
                             user->opcode() == SpvOpBranch ||
                             user->opcode() == SpvOpBranchConditional ||
                             user->opcode() == SpvOpSwitch)) {
                          return true;
                        }
                        if (user_block == nullptr ||
                            region.count(user_block) == 0) {
                          return false;
                        }
                        return true;
                      })) {
                return false;
              }
              return true;
            })) {
      return false;
    }
  }
  return true;
}

bool StructuredConstructToBlockReductionOpportunityFinder::
    HasUnreachablePredecessor(const opt::BasicBlock& block,
                              opt::IRContext* context) {
  for (auto pred : context->cfg()->preds(block.id())) {
    if (!context->IsReachable(*context->cfg()->block(pred))) {
      return true;
    }
  }
  return false;
}

std::string StructuredConstructToBlockReductionOpportunityFinder::GetName()
    const {
  return "StructuredConstructToBlockReductionOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
