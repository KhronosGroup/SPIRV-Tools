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

#ifndef LIBSPIRV_OPT_LOOP_UTILS_H_
#define LIBSPIRV_OPT_LOOP_UTILS_H_

namespace spvtools {

namespace ir {
class Loop;
class IRContext;
}  // namespace ir

namespace opt {

// Set of basic loop transformation.
class LoopUtils {
 public:
  LoopUtils(ir::IRContext* context, ir::Loop* loop)
      : context_(context), loop_(loop) {}

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

 private:
  ir::IRContext* context_;
  ir::Loop* loop_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_LOOP_UTILS_H_
