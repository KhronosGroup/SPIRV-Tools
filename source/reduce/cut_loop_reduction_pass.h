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

#ifndef SOURCE_REDUCE_CUT_LOOP_REDUCTION_PASS_H_
#define SOURCE_REDUCE_CUT_LOOP_REDUCTION_PASS_H_

#include "reduction_pass.h"

namespace spvtools {
namespace reduce {

// A reduction pass for cutting structured loops, paving the way for their
// constituent blocks to be more aggressively reduced.  A loop is cut by
// replacing all references to its continue target with references to its merge
// block, and eliminating the loop's merge instruction.  No blocks are removed
// by the pass, so the loop's continue target blocks, and what was the loop's
// back edge, persist; another pass for eliminating blocks may end up being able
// to remove them.
class CutLoopReductionPass : public ReductionPass {
 public:
  // Creates the reduction pass in the context of the given target environment
  // |target_env|
  explicit CutLoopReductionPass(const spv_target_env target_env)
      : ReductionPass(target_env) {}

  ~CutLoopReductionPass() override = default;

  // The name of this pass.
  std::string GetName() const final;

 protected:
  // Finds all opportunities for cutting a loop in the given module.
  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      opt::IRContext* context) const final;

 private:
  // Decides whether the given loop is suitable for cutting.
  bool CanBeCut(const opt::Loop& loop, const opt::Function& function) const;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_CUT_LOOP_REDUCTION_PASS_H_
