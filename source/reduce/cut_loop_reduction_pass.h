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

// A reduction pass for cutting loops, paving the way for their constituent
// blocks to be more aggressively reduced.
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
  // Finds all opportunities for removing a loop in the given module.
  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      opt::IRContext* context) const final;

 private:
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_CUT_LOOP_REDUCTION_PASS_H_
