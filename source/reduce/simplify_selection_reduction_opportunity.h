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

#ifndef SOURCE_REDUCE_SIMPLIFY_SELECTION_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_SIMPLIFY_SELECTION_REDUCTION_OPPORTUNITY_H_

#include "reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// An opportunity to simplify a selection construct so that both of its branches
// target the same block.
class SimplifySelectionReductionOpportunity : public ReductionOpportunity {
 public:
  // Constructs a reduction opportunity from the |branch_instruction|, which is
  // required to be the conditional branch of the block heading a selection
  // construct, and |redirect_to_lhs|, which dictates whether the construct
  // should be simplified to have both branches point to the former LHS or
  // former RHS of the construct.
  SimplifySelectionReductionOpportunity(opt::Instruction* branch_instruction,
                                        bool redirect_to_lhs)
      : branch_instruction_(branch_instruction),
        redirect_to_lhs_(redirect_to_lhs) {
    assert(
        branch_instruction_->opcode() == SpvOpBranchConditional &&
        "Precondition of SimplifySelectionReductionOpportunity does not hold.");
  }

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  opt::Instruction* branch_instruction_;
  bool redirect_to_lhs_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  //   SOURCE_REDUCE_SIMPLIFY_SELECTION_REDUCTION_OPPORTUNITY_H_
