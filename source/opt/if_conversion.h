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

#ifndef SOURCE_OPT_IF_CONVERSION_H_
#define SOURCE_OPT_IF_CONVERSION_H_

#include "source/opt/basic_block.h"
#include "source/opt/ir_builder.h"
#include "source/opt/pass.h"
#include "source/opt/pass_token.h"
#include "source/opt/types.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class IfConversion : public Pass {
 public:
  Status Process(opt::IRContext* context) override;

  opt::IRContext::Analysis GetPreservedAnalyses() override {
    return opt::IRContext::kAnalysisDefUse |
           opt::IRContext::kAnalysisDominatorAnalysis |
           opt::IRContext::kAnalysisInstrToBlockMapping |
           opt::IRContext::kAnalysisCFG | opt::IRContext::kAnalysisNameMap;
  }

 private:
  // Returns true if |id| is a valid type for use with OpSelect. OpSelect only
  // allows scalars, vectors and pointers as valid inputs.
  bool CheckType(uint32_t id);

  // Returns the basic block containing |id|.
  opt::BasicBlock* GetBlock(uint32_t id);

  // Returns the basic block for the |predecessor|'th index predecessor of
  // |phi|.
  opt::BasicBlock* GetIncomingBlock(opt::Instruction* phi,
                                    uint32_t predecessor);

  // Returns the instruction defining the |predecessor|'th index of |phi|.
  opt::Instruction* GetIncomingValue(opt::Instruction* phi,
                                     uint32_t predecessor);

  // Returns the id of a OpCompositeConstruct boolean vector. The composite has
  // the same number of elements as |vec_data_ty| and each member is |cond|.
  // |where| indicates the location in |block| to insert the composite
  // construct. If necessary, this function will also construct the necessary
  // type instructions for the boolean vector.
  uint32_t SplatCondition(analysis::Vector* vec_data_ty, uint32_t cond,
                          InstructionBuilder* builder);

  // Returns true if none of |phi|'s users are in |block|.
  bool CheckPhiUsers(opt::Instruction* phi, opt::BasicBlock* block);

  // Returns |false| if |block| is not appropriate to transform. Only
  // transforms blocks with two predecessors. Neither incoming block can be
  // dominated by |block|. Both predecessors must share a common dominator that
  // is terminated by a conditional branch.
  bool CheckBlock(opt::BasicBlock* block, DominatorAnalysis* dominators,
                  opt::BasicBlock** common);

  // Moves |inst| to |target_block| if it does not already dominate the block.
  // Any instructions that |inst| depends on are move if necessary.  It is
  // assumed that |inst| can be hoisted to |target_block| as defined by
  // |CanHoistInstruction|.  |dominators| is the dominator analysis for the
  // function that contains |target_block|.
  void HoistInstruction(opt::Instruction* inst, opt::BasicBlock* target_block,
                        DominatorAnalysis* dominators);

  // Returns true if it is legal to move |inst| and the instructions it depends
  // on to |target_block| if they do not already dominate |target_block|.
  bool CanHoistInstruction(opt::Instruction* inst,
                           opt::BasicBlock* target_block,
                           DominatorAnalysis* dominators);
};

class IfConversionToken : public PassToken {
 public:
  IfConversionToken() = default;
  ~IfConversionToken() override = default;

  const char* name() const override { return "if-conversion"; }

  std::unique_ptr<Pass> CreatePass() const override {
    return MakeUnique<IfConversion>();
  }
};

}  //  namespace opt
}  //  namespace spvtools

#endif  // SOURCE_OPT_IF_CONVERSION_H_
