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

#ifndef SOURCE_REDUCE_CUT_LOOP_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_CUT_LOOP_REDUCTION_OPPORTUNITY_H_

#include "reduction_opportunity.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/function.h"

namespace spvtools {
namespace reduce {

using namespace opt;

// TODO: comment.
class CutLoopReductionOpportunity : public ReductionOpportunity {
 public:
  // TODO: comment.
  explicit CutLoopReductionOpportunity(Function::iterator loop_construct_header,
                                       Function* enclosing_function)
      : loop_construct_header_(loop_construct_header),
        enclosing_function_(enclosing_function) {}

  // We require the loop header to be reachable.
  bool PreconditionHolds() override;

 protected:
  // TODO: comment.
  void Apply() override;

 private:
  Function::iterator loop_construct_header_;
  Function* enclosing_function_;

  void ReplaceSelectionTargetWithClosestMerge(
      IRContext* context, const CFG& cfg,
      const DominatorAnalysis& dominator_analysis,
      uint32_t original_target_block_id, uint32_t predecessor_block_id);
  uint32_t FindClosestMerge(const CFG& cfg,
                            const DominatorAnalysis& dominator_analysis,
                            uint32_t block_id);
  void ChangeLoopToSelection(IRContext* context, const CFG& cfg);
  void RedirectEdge(uint32_t source_id, uint32_t original_target_id,
                    uint32_t new_target_id, IRContext* context, const CFG& cfg);
  void AdaptPhiNodesForRemovedEdge(uint32_t from_id, BasicBlock* to_block);
  void AdaptPhiNodesForAddedEdge(uint32_t from_id, BasicBlock* to_id,
                                 IRContext* context);
  uint32_t FindOrCreateGlobalUndef(IRContext* context, uint32_t type_id);
  bool ContainedInStructuredControlFlowConstruct(
      uint32_t block_id, BasicBlock* selection_construct_header,
      const DominatorAnalysis& dominator_analysis);
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_CUT_LOOP_REDUCTION_OPPORTUNITY_H_
