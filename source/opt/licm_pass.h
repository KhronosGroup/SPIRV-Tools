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

#ifndef SOURCE_OPT_LICM_PASS_H_
#define SOURCE_OPT_LICM_PASS_H_

#include "opt/basic_block.h"
#include "opt/instruction.h"
#include "opt/loop_descriptor.h"
#include "opt/pass.h"

#include <queue>

namespace spvtools {
namespace opt {

class LICMPass : public Pass {
 public:
  LICMPass() {}

  const char* name() const override { return "loop-invariant-code-motion"; }
  Status Process(opt::IRContext*) override;

 private:
  // Searches the IRContext for functions and processes each, moving invariants
  // outside loops within the function where possible
  // Returns true if a change was made to a function within the IRContext
  bool ProcessIRContext();

  // Checks the function for loops, calling ProcessLoop on each one found.
  // Returns true if a change was made to the function, false otherwise.
  bool ProcessFunction(opt::Function* f);

  // Checks for invariants in the loop and attempts to move them to the loops
  // preheader. Works from inner loop to outer when nested loops are found.
  // Returns true if a change was made to the loop, false otherwise.
  bool ProcessLoop(opt::Loop* loop, opt::Function* f);

  // Analyses each instruction in |bb|, hoisting invariants to |pre_header_bb|.
  // Each child of |bb| wrt to |dom_tree| is pushed to |loop_bbs|
  bool AnalyseAndHoistFromBB(opt::Loop* loop, opt::Function* f,
                             opt::BasicBlock* bb,
                             std::vector<opt::BasicBlock*>* loop_bbs);

  // Returns true if |bb| is immediately contained in |loop|
  bool IsImmediatelyContainedInLoop(opt::Loop* loop, opt::Function* f,
                                    opt::BasicBlock* bb);

  // Move the instruction to the given BasicBlock
  // This method will update the instruction to block mapping for the context
  void HoistInstruction(opt::Loop* loop, opt::Instruction* inst);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LICM_PASS_H_
