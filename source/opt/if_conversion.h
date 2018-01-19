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

#ifndef LIBSPIRV_OPT_IF_CONVERSION_H_
#define LIBSPIRV_OPT_IF_CONVERSION_H_

#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class IfConversion : public Pass {
 public:
  const char* name() const override { return "if-conversion"; }
  Status Process(ir::IRContext* context) override;

  ir::IRContext::Analysis GetPreservedAnalyses() override {
    return ir::IRContext::kAnalysisDefUse |
           ir::IRContext::kAnalysisDominatorAnalysis |
           ir::IRContext::kAnalysisInstrToBlockMapping |
           ir::IRContext::kAnalysisCFG;
  }

 private:
  bool CheckType(uint32_t id);
  ir::BasicBlock* GetBlock(uint32_t id);
  ir::BasicBlock* GetIncomingBlock(ir::Instruction* phi, uint32_t predecessor);
  ir::Instruction* GetIncomingValue(ir::Instruction* phi, uint32_t predecessor);
  ir::BasicBlock* CommonDominator(ir::BasicBlock* inc0, ir::BasicBlock* inc1,
                                  const DominatorAnalysis& dominators);
};

}  //  namespace opt
}  //  namespace spvtools

#endif  //  LIBSPIRV_OPT_IF_CONVERSION_H_
