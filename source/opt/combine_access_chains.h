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

#ifndef LIBSPIRV_OPT_COMBINE_ACCESS_CHAINS_H_
#define LIBSPIRV_OPT_COMBINE_ACCESS_CHAINS_H_

#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class CombineAccessChainsPass : public Pass {
 public:
  const char* name() const override { return "combine-access-chains"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisNameMap;
  }

 private:
  bool ProcessFunction(Function& function);
  bool CombinePtrAccessChain(Instruction* inst);

  uint32_t GetConstantValue(const analysis::Constant* constant_inst);
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_COMBINE_ACCESS_CHAINS_H_
