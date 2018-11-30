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

// Captures an opportunity to replace a structured loop with a selection.
class StructuredLoopToSelectionReductionOpportunity
    : public ReductionOpportunity {
 public:
  // Constructs an opportunity from a loop header block and the function that
  // encloses it.
  explicit StructuredLoopToSelectionReductionOpportunity(
      BasicBlock* loop_construct_header, Function* enclosing_function)
      : loop_construct_header_(loop_construct_header),
        enclosing_function_(enclosing_function) {}

  // We require the loop header to be reachable.  A structured loop might
  // become unreachable as a result of turning another structured loop into
  // a selection.
  bool PreconditionHolds() override;

 protected:
  // Perform the structured loop to selection transformation.
  void Apply() override;

 private:
  BasicBlock* loop_construct_header_;
  Function* enclosing_function_;

  // In a naive manner (linear time per invocation) checks whether the global
  // value list has an OpUndef of the given type, adding one if not, and
  // returns the id of such an OpUndef.
  //
  // TODO: This will likely be used by other reduction passes, so should be
  // factored out in due course.  Parts of the spirv-opt framework provide
  // similar functionality, so there may be a case for further refactoring.
  uint32_t FindOrCreateGlobalUndef(IRContext* context, uint32_t type_id);

  void RedirectToClosestMergeBlock(uint32_t original_target_id,
                                   IRContext* context,
                                   const DominatorAnalysis& dominator_analysis,
                                   const CFG& cfg);

  uint32_t FindClosestMerge(const CFG& cfg,
                            const DominatorAnalysis& dominator_analysis,
                            uint32_t block_id);

  void RedirectEdge(uint32_t source_id, uint32_t original_target_id,
                    uint32_t new_target_id, IRContext* context, const CFG& cfg);

  void AdaptPhiNodesForRemovedEdge(uint32_t from_id, BasicBlock* to_block);

  void AdaptPhiNodesForAddedEdge(uint32_t from_id, BasicBlock* to_id,
                                 IRContext* context);

  bool ContainedInStructuredControlFlowConstruct(
      uint32_t block_id, BasicBlock* selection_construct_header,
      const DominatorAnalysis& dominator_analysis);

  void ChangeLoopToSelection(IRContext* context, const CFG& cfg);
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_CUT_LOOP_REDUCTION_OPPORTUNITY_H_
