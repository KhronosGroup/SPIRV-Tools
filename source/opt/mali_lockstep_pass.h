// Copyright (c) 2026 Lee Gao
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

#ifndef SOURCE_OPT_MALI_LOCKSTEP_PASS_H_
#define SOURCE_OPT_MALI_LOCKSTEP_PASS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// MaliBrdige: the strided loops of a 32-thread compute shader written for one
// 32-lane wave, made to take the same number of iterations in every invocation,
// so that a shared-memory barrier pass can put workgroup barriers inside them.
//
// A loop whose header has i = OpPhi(LocalInvocationId.x, i + W) (W = the
// workgroup's X size), whose one way out is a conditional branch on (n <= i) or
// (i >= n) with n defined before the loop, after every Workgroup access of the
// loop, gets a second phi b = OpPhi(0, b + W) and its exit test made on b
// instead of i.
class MaliLockstepPass : public Pass {
  public:
    MaliLockstepPass() = default;
    const char *name() const override { return "mali-lockstep"; }
    Status Process() override;

    IRContext::Analysis GetPreservedAnalyses() override {
        return IRContext::kAnalysisNone;
    }
};

} // namespace opt
} // namespace spvtools

#endif // SOURCE_OPT_MALI_LOCKSTEP_PASS_H_
