// Copyright (c) 2018 Google LLC.
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

#ifndef SOURCE_OPT_LOOP_UTILS_H_
#define SOURCE_OPT_LOOP_UTILS_H_
#include <list>
#include <memory>
#include <vector>
#include "opt/loop_descriptor.h"

namespace spvtools {

namespace ir {
class Loop;
class IRContext;
}  // namespace ir

namespace opt {

// LoopUtils is used to encapsulte loop optimizations and from the passes which
// use them. Any pass which needs a loop optimization should do it through this
// or through a pass which is using this.
class LoopUtils {
 public:
  LoopUtils(ir::IRContext* context, ir::Loop* loop)
      : context_(context),
        loop_(loop),
        function_(*loop_->GetHeaderBlock()->GetParent()) {}

  // The converts the current loop to loop closed SSA form.
  // In the loop closed SSA, all loop exiting values go through a dedicated Phi
  // instruction. For instance:
  //
  // for (...) {
  //   A1 = ...
  //   if (...)
  //     A2 = ...
  //   A = phi A1, A2
  // }
  // ... = op A ...
  //
  // Becomes
  //
  // for (...) {
  //   A1 = ...
  //   if (...)
  //     A2 = ...
  //   A = phi A1, A2
  // }
  // C = phi A
  // ... = op C ...
  //
  // This makes some loop transformations (such as loop unswitch) simpler
  // (removes the needs to take care of exiting variables).
  void MakeLoopClosedSSA();

  // Create dedicate exit basic block. This ensure all exit basic blocks has the
  // loop as sole predecessors.
  // By construction, structured control flow already has a dedicated exit
  // block.
  // Preserves: CFG, def/use and instruction to block mapping.
  void CreateLoopDedicatedExits();

  // Perfom a partial unroll of |loop| by given |factor|. This will copy the
  // body of the loop |factor| times. So a |factor| of one would give a new loop
  // with the original body plus one unrolled copy body.
  bool PartiallyUnroll(size_t factor);

  // Fully unroll |loop|.
  bool FullyUnroll();

  // This function validates that |loop| meets the assumptions made by the
  // implementation of the loop unroller. As the implementation accommodates
  // more types of loops this function can reduce its checks.
  //
  // The conditions checked to ensure the loop can be unrolled are as follows:
  // 1. That the loop is in structured order.
  // 2. That the condinue block is a branch to the header.
  // 3. That the only phi used in the loop is the induction variable.
  //  TODO(stephen@codeplay.com): This is a temporary mesure, after the loop is
  //  converted into LCSAA form and has a single entry and exit we can rewrite
  //  the other phis.
  // 4. That this is an inner most loop, or that loops contained within this
  // loop have already been fully unrolled.
  // 5. That each instruction in the loop is only used within the loop.
  // (Related to the above phi condition).
  bool CanPerformUnroll();

  // Maintains the loop descriptor object after the unroll functions have been
  // called, otherwise the analysis should be invalidated.
  void Finalize();

 private:
  ir::IRContext* context_;
  ir::Loop* loop_;
  ir::Function& function_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_UTILS_H_
