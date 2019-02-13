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

#ifndef SOURCE_REDUCE_REMOVE_SELECTION_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_REMOVE_SELECTION_REDUCTION_OPPORTUNITY_H_

#include <source/opt/basic_block.h>
#include "reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// An opportunity to remove a selection construct, demoting it to a branch to
// one of the former targets of the selection.
class RemoveSelectionReductionOpportunity : public ReductionOpportunity {
 public:
  // Constructs a reduction opportunity from |block|, which heads the selection
  // construct to be removed, and |choose_lhs|, which dictates whether the block
  // will jump straight to the former LHS or RHS of the selection construct.
  RemoveSelectionReductionOpportunity(opt::BasicBlock* block, bool choose_lhs)
      : block_(block), choose_lhs_(choose_lhs) {}

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  opt::BasicBlock* block_;
  bool choose_lhs_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  //   SOURCE_REDUCE_REMOVE_SELECTION_REDUCTION_OPPORTUNITY_H_
