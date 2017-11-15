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

#include "dominator_analysis_pass.h"
#include <iostream>
#include "cfa.h"

namespace spvtools {
namespace opt {

DominatorAnalysis::DominatorAnalysis() {}

void DominatorAnalysis::InitializeTree(ir::Module& module) {
  for (const ir::Function& func : module) {
    InitializeTree(&func);
  }
}

void DominatorAnalysis::InitializeTree(const ir::Function* F) {
  Trees[F] = {};

  Trees[F].InitializeTree(F);
  Trees[F].DumpTreeAsDot(std::cout);
}

bool DominatorAnalysis::Dominates(const ir::BasicBlock* A,
                                  const ir::BasicBlock* B,
                                  const ir::Function* F) const {
  return Dominates(A->id(), B->id(), F);
}
bool DominatorAnalysis::Dominates(uint32_t A, uint32_t B,
                                  const ir::Function* F) const {
  auto itr = Trees.find(F);
  return itr->second.Dominates(A, B);
}

bool DominatorAnalysis::StrictlyDominates(const ir::BasicBlock* A,
                                          const ir::BasicBlock* B,
                                          const ir::Function* F) const {
  return StrictlyDominates(A->id(), B->id(), F);
}

bool DominatorAnalysis::StrictlyDominates(uint32_t A, uint32_t B,
                                          const ir::Function* F) const {
  auto itr = Trees.find(F);
  return itr->second.StrictlyDominates(A, B);
}

void DominatorAnalysis::CheckAllNodesForDomination(
    ir::Module& module, std::ostream& OutStream) const {
  for (ir::Function& F : module) {
    // Skip over empty functions
    if (F.cbegin() == F.cend()) {
      continue;
    }

    for (ir::BasicBlock& BB : F) {
      for (ir::BasicBlock& BB2 : F) {
        OutStream << BB.id();

        if (Dominates(&BB, &BB2, &F)) {
          OutStream << " dominates ";
        } else {
          OutStream << " does not dominate ";
        }
        OutStream << BB2.id() << "\n";
      }
    }
  }
}

}  // opt
}  // spvtools
