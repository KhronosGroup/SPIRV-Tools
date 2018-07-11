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

#ifndef SOURCE_OPT_SIMPLIFICATION_PASS_H_
#define SOURCE_OPT_SIMPLIFICATION_PASS_H_

#include "source/opt/function.h"
#include "source/opt/ir_context.h"
#include "source/opt/pass.h"
#include "source/opt/pass_token.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class SimplificationPass : public Pass {
 public:
  Status Process(opt::IRContext*) override;
  virtual opt::IRContext::Analysis GetPreservedAnalyses() override {
    return opt::IRContext::kAnalysisDefUse |
           opt::IRContext::kAnalysisInstrToBlockMapping |
           opt::IRContext::kAnalysisDecorations |
           opt::IRContext::kAnalysisCombinators | opt::IRContext::kAnalysisCFG |
           opt::IRContext::kAnalysisDominatorAnalysis |
           opt::IRContext::kAnalysisNameMap;
  }

 private:
  // Returns true if the module was changed.  The simplifier is called on every
  // instruction in |function| until nothing else in the function can be
  // simplified.
  bool SimplifyFunction(opt::Function* function);
};

class SimplificationPassToken : public PassToken {
 public:
  SimplificationPassToken() = default;
  ~SimplificationPassToken() override = default;

  const char* name() const override { return "simplify-instructions"; }

  std::unique_ptr<Pass> CreatePass() const override {
    return MakeUnique<SimplificationPass>();
  }
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_SIMPLIFICATION_PASS_H_
