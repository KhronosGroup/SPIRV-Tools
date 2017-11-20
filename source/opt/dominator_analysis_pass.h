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

#ifndef DOMINATOR_ANALYSIS_PASS_H
#define DOMINATOR_ANALYSIS_PASS_H
#include <cstdint>
#include "dominator_tree.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

class DominatorAnalysis {
 public:
  DominatorAnalysis();

  void InitializeTree(ir::Module& TheModule);

  void InitializeTree(const ir::Function* F);

  bool Dominates(const ir::BasicBlock* A, const ir::BasicBlock* B,
                 const ir::Function* F) const;

  bool Dominates(uint32_t A, uint32_t B, const ir::Function* F) const;

  bool StrictlyDominates(const ir::BasicBlock* A, const ir::BasicBlock* B,
                         const ir::Function* F) const;

  bool StrictlyDominates(uint32_t A, uint32_t B, const ir::Function* F) const;

  void CheckAllNodesForDomination(ir::Module& M, std::ostream& OutStream) const;

 private:
  // Each function in the module will create its own dominator tree
  std::map<const ir::Function*, DominatorTree> Trees;
};

class DominatorAnalysisPass : public Pass {
 public:
  Status Process(ir::IRContext* c) override {
    DominatorAnalysis DA;
    DA.InitializeTree(*c->module());
    // DA.CheckAllNodesForDomination(*c->module(), std::cout);
    return Status::SuccessWithoutChange;
  }

  const char* name() const override { return "Dominator Analysis Pass"; }
};

}  // ir
}  // spvtools

#endif  // DOMINATOR_ANALYSIS_PASS_H
