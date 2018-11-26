// Copyright (c) 2018 Google LLC
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

#include "cut_loop_reduction_pass.h"
#include "cut_loop_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

using namespace opt;

namespace {
const uint32_t kMergeNodeIndex = 0;
}

std::vector<std::unique_ptr<ReductionOpportunity>>
CutLoopReductionPass::GetAvailableOpportunities(opt::IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  for (auto& function : *context->module()) {
    auto loop_descriptor = context->GetLoopDescriptor(&function);
    for (auto loop : loop_descriptor->GetLoopsInBinaryLayoutOrder()) {
      if (CanBeCut(*loop, function)) {
        result.push_back(MakeUnique<CutLoopReductionOpportunity>(loop));
      }
    }
  }
  return result;
}

std::string CutLoopReductionPass::GetName() const {
  return "CutLoopReductionPass";
}

bool CutLoopReductionPass::CanBeCut(const opt::Loop& loop,
                                    const opt::Function& function) const {
  auto cfg = loop.GetContext()->cfg();
  auto dom_tree =
      loop.GetContext()->GetDominatorAnalysis(&function)->GetDomTree();
  // Consider each block from which the loop's merge block can be immediately
  // reached.
  for (auto pred : cfg->preds(loop.GetMergeBlock()->id())) {
    // Walk the dominator tree backwards from pred, searching for a merge
    // instruction whose merge block is different to the loop's merge block.
    for (auto current_dominator = cfg->block(pred);
         current_dominator != loop.GetHeaderBlock();
         current_dominator = dom_tree.ImmediateDominator(current_dominator)) {
      if (!current_dominator->GetMergeInst()) {
        // This dominator isn't a merge instruction.
        continue;
      }
      auto merge_block_id =
          current_dominator->GetMergeInst()->GetSingleWordInOperand(
              kMergeNodeIndex);
      if (loop.GetMergeBlock()->id() != merge_block_id) {
        // We have found a merge instruction that dominates our loop break,
        // such that the associated merge block is different from the loop's
        // merge block.  Cutting the loop would lead to an invalid module.
        return false;
      }
    }
  }
  return true;
}

}  // namespace reduce
}  // namespace spvtools
