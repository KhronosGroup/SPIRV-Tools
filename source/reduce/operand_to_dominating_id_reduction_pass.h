// Copyright (c) 2018 Google Inc.
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

#ifndef SOURCE_REDUCE_OPERAND_TO_DOMINATING_ID_REDUCTION_PASS_H_
#define SOURCE_REDUCE_OPERAND_TO_DOMINATING_ID_REDUCTION_PASS_H_

#include "reduction_pass.h"

namespace spvtools {
namespace reduce {

// A reduction pass that aims to bring to SPIR-V (and generalize) the idea from
// human-readable languages of e.g. replacing an expression with one of its
// arguments, (x + y) -> x, or with a reference to an identifier that was
// assigned to higher up in the program.  The generalization of this is to
// replace an id with a different id of the same type defined in some
// dominating instruction.
//
// If id x is defined and then used several times, changing each use of x to
// some dominating definition may eventually allow the statement defining x
// to be eliminated by another pass.
class OperandToDominatingIdReductionPass : public ReductionPass {
 public:
  // Creates the reduction pass in the context of the given target environment
  // |target_env|
  explicit OperandToDominatingIdReductionPass(const spv_target_env target_env)
      : ReductionPass(target_env) {}

  ~OperandToDominatingIdReductionPass() override = default;

  // The name of this pass.
  std::string GetName() const final;

 protected:
  // Finds all opportunities for replacing an operand with a dominating
  // instruction in a given module.
  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      opt::IRContext* context) const final;

 private:
  void GetOpportunitiesForDominatingInst(
      std::vector<std::unique_ptr<ReductionOpportunity>>* opportunities,
      opt::Instruction* dominating_instruction,
      opt::Function::iterator candidate_dominator_block,
      opt::Function* function, opt::IRContext* context) const;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_OPERAND_TO_DOMINATING_ID_REDUCTION_PASS_H_
