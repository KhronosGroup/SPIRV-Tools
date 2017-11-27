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

// Interface to perform dominator or postdominator analysis on a given function.
class DominatorAnalysisBase {
 public:
  DominatorAnalysisBase(bool isPostDom) : Tree(isPostDom) {}

  // Calculate the dominator (or postdominator) tree for given function F.
  void InitializeTree(const ir::Function* F);

  // Returns true if BasicBlock A dominates BasicBlock B.
  bool Dominates(const ir::BasicBlock* A, const ir::BasicBlock* B) const;

  // Returns true if BasicBlock A dominates BasicBlock B. Same as above only
  // using the BasicBlock IDs.
  bool Dominates(uint32_t A, uint32_t B) const;

  // Returns true if BasicBlock A strictly dominates BasicBlock B.
  bool StrictlyDominates(const ir::BasicBlock* A,
                         const ir::BasicBlock* B) const;

  // Returns true if BasicBlock A strictly dominates BasicBlock B. Same as above
  // only using the BasicBlock IDs.
  bool StrictlyDominates(uint32_t A, uint32_t B) const;

  // Dump the tree structure into the given stream in the dot format.
  void DumpAsDot(std::ostream& Out) const;

  // Return the immediate dominator of node A or return NULL if it is an entry
  // node.
  ir::BasicBlock* ImmediateDominator(const ir::BasicBlock* A) const;

  // Return the immediate dominator of node A or return NULL if it is an entry
  // node. Same as above but operates on IDs.§
  ir::BasicBlock* ImmediateDominator(uint32_t A) const;

  // Returns true if this is a postdomiator tree.
  bool isPostDominator() const { return Tree.isPostDominator(); }

  // Return the tree itself for manual operations, such as traversing the roots.
  // For normal dominance relationships the methods above should be used.
  DominatorTree& GetDomTree() { return Tree; }
  const DominatorTree& GetDomTree() const { return Tree; }

 protected:
  DominatorTree Tree;
};

// Derived class for normal dominator analysis.
class DominatorAnalysis : public DominatorAnalysisBase {
 public:
  DominatorAnalysis() : DominatorAnalysisBase(false) {}
};

// Derived class for postdominator analysis.
class PostDominatorAnalysis : public DominatorAnalysisBase {
 public:
  PostDominatorAnalysis() : DominatorAnalysisBase(true) {}
};

// A simple mechanism to cache the result for the dominator tree.
class DominatorAnalysisPass {
 public:
  DominatorAnalysisPass() {}

  // Gets the dominator analysis for function F.
  DominatorAnalysis* GetDominatorAnalysis(const ir::Function* F);

  // Gets the postdominator analysis for function F.
  PostDominatorAnalysis* GetPostDominatorAnalysis(const ir::Function* F);

 private:
  // Each function in the module will create its own dominator tree. We cache
  // the result so it doesn't need to be rebuilt each time.
  std::map<const ir::Function*, DominatorAnalysis> DomTrees;
  std::map<const ir::Function*, PostDominatorAnalysis> PostDomTrees;
};

}  // ir
}  // spvtools

#endif  // DOMINATOR_ANALYSIS_PASS_H
