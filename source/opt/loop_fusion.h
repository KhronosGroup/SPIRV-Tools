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

#ifndef SOURCE_OPT_LOOP_FUSION_H_
#define SOURCE_OPT_LOOP_FUSION_H_

#include <map>
#include <set>
#include <vector>

#include "opt/ir_context.h"
#include "opt/loop_descriptor.h"
#include "opt/loop_utils.h"
#include "opt/scalar_analysis.h"

namespace spvtools {
namespace opt {

class LoopFusion {
 public:
  LoopFusion(ir::IRContext* context, ir::Loop* loop_0, ir::Loop* loop_1)
      : context_(context),
        loop_0_(loop_0),
        loop_1_(loop_1),
        containing_function_(loop_0->GetHeaderBlock()->GetParent()) {}

  // Checks if the |loop_0| and |loop_1| are compatible for fusion.
  // That means:
  //   * they both have one induction variable
  //   * they have the same upper and lower bounds
  //     - same inital value
  //     - same condition
  //   * they have the same update step
  //   * they are adjacent, with |loop_0| appearing before |loop_1|
  //   * there are no break/continue in either of them
  //   * they both have pre-header blocks (required for ScalarEvolutionAnalysis
  //     and dependence checking).
  bool AreCompatible();

  // Checks if compatible |loop_0| and |loop_1| are legal to fuse.
  // * fused loops do not have any dependencies with dependence distance greater
  //   than 0 that did not exist in the original loops.
  // * there are no function calls in the loops (could have side-effects)
  bool IsLegal();

  // Perform the actual fusion of |loop_0_| and |loop_1_|. The loops have to be
  // compatible and the fusion has to be legal.
  void Fuse();

 private:
  // Check that the initial values are the same.
  bool CheckInit();

  // Check that the conditions are the same.
  bool CheckCondition();

  // Check that the steps are the same.
  bool CheckStep();

  // Returns |true| if |instruction| is used in the continue or condition block
  // of |loop|.
  bool UsedInContinueOrConditionBlock(ir::Instruction* instruction,
                                      ir::Loop* loop);

  // Remove entries in |instructions| that are not used in the continue or
  // condition block of |loop|.
  void RemoveIfNotUsedContinueOrConditionBlock(
      std::vector<ir::Instruction*>* instructions, ir::Loop* loop);

  // Returns |true| if |instruction| is used in |loop|.
  bool IsUsedInLoop(ir::Instruction* instruction, ir::Loop* loop);

  // Returns |true| if |loop| has at least one barrier or function call.
  bool ContainsBarriersOrFunctionCalls(ir::Loop* loop);

  // Get all instructions in the |loop| (except in the latch block) that have
  // the opcode |opcode|.
  std::pair<std::vector<ir::Instruction*>, std::vector<ir::Instruction*>>
  GetLoadsAndStoresInLoop(ir::Loop* loop);

  // Given a vector of memory operations (OpLoad/OpStore), constructs a map from
  // variables to the loads/stores that those variables.
  std::map<ir::Instruction*, std::vector<ir::Instruction*>> LocationToMemOps(
      const std::vector<ir::Instruction*>& mem_ops);

  ir::IRContext* context_;

  // The original loops to be fused.
  ir::Loop* loop_0_;
  ir::Loop* loop_1_;

  // The function that contains |loop_0_| and |loop_1_|.
  ir::Function* containing_function_ = nullptr;

  // The induction variables for |loop_0_| and |loop_1_|.
  ir::Instruction* induction_0_ = nullptr;
  ir::Instruction* induction_1_ = nullptr;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_FUSION_H_
