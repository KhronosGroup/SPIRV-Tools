// Copyright (c) 2017 Google Inc.
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

#ifndef SOURCE_OPT_COMPACT_IDS_PASS_H_
#define SOURCE_OPT_COMPACT_IDS_PASS_H_

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"
#include "source/opt/pass_token.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class CompactIdsPass : public Pass {
 public:
  Status Process(opt::IRContext*) override;

  // Return the mask of preserved Analyses.
  opt::IRContext::Analysis GetPreservedAnalyses() override {
    return opt::IRContext::kAnalysisInstrToBlockMapping |
           opt::IRContext::kAnalysisDominatorAnalysis |
           opt::IRContext::kAnalysisLoopAnalysis;
  }
};

class CompactIdsPassToken : public PassToken {
 public:
  CompactIdsPassToken() = default;
  ~CompactIdsPassToken() override = default;

  const char* name() const override { return "compact-ids"; }

  std::unique_ptr<Pass> CreatePass() const override {
    return MakeUnique<CompactIdsPass>();
  }
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_COMPACT_IDS_PASS_H_
