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
#include "cfa.h"

namespace spvtools {
namespace opt {

DominatorAnalysis* DominatorAnalysisPass::GetDominatorAnalysis(
    const ir::Function* f) {
  if (DomTrees.find(f) == DomTrees.end()) {
    DomTrees[f].InitializeTree(f);
  }

  return &DomTrees[f];
}

PostDominatorAnalysis* DominatorAnalysisPass::GetPostDominatorAnalysis(
    const ir::Function* f) {
  if (PostDomTrees.find(f) == PostDomTrees.end()) {
    PostDomTrees[f].InitializeTree(f);
  }

  return &PostDomTrees[f];
}

void DominatorAnalysisBase::InitializeTree(const ir::Function* f) {
  Tree.InitializeTree(f);
}

bool DominatorAnalysisBase::Dominates(const ir::BasicBlock* A,
                                      const ir::BasicBlock* B) const {
  if (!A || !B) return false;
  return Dominates(A->id(), B->id());
}

bool DominatorAnalysisBase::Dominates(uint32_t A, uint32_t B) const {
  return Tree.Dominates(A, B);
}

bool DominatorAnalysisBase::StrictlyDominates(const ir::BasicBlock* A,
                                              const ir::BasicBlock* B) const {
  if (!A || !B) return false;
  return StrictlyDominates(A->id(), B->id());
}

bool DominatorAnalysisBase::StrictlyDominates(uint32_t A, uint32_t B) const {
  return Tree.StrictlyDominates(A, B);
}

void DominatorAnalysisBase::DumpAsDot(std::ostream& Out) const {
  Tree.DumpTreeAsDot(Out);
}

ir::BasicBlock* DominatorAnalysisBase::ImmediateDominator(
    const ir::BasicBlock* node) const {
  if (!node) return nullptr;
  return Tree.ImmediateDominator(node);
}

ir::BasicBlock* DominatorAnalysisBase::ImmediateDominator(uint32_t id) const {
  return Tree.ImmediateDominator(id);
}

}  // opt
}  // spvtools
