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

#ifndef LIBSPIRV_OPT_DOMINATOR_ANALYSIS_PASS_H_
#define LIBSPIRV_OPT_DOMINATOR_ANALYSIS_PASS_H_

#include <cstdint>
#include <map>

#include "dominator_tree.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// Interface to perform dominator or postdominator analysis on a given function.
class DominatorAnalysisBase {
 public:
  explicit DominatorAnalysisBase(bool isPostDom) : tree_(isPostDom) {}

  // Calculate the dominator (or postdominator) tree for given function F.
  inline void InitializeTree(const ir::Function* f) { tree_.InitializeTree(f); }

  // Returns true if BasicBlock A dominates BasicBlock B.
  inline bool Dominates(const ir::BasicBlock* a,
                        const ir::BasicBlock* b) const {
    if (!a || !b) return false;
    return Dominates(a->id(), b->id());
  }

  // Returns true if BasicBlock A dominates BasicBlock B. Same as above only
  // using the BasicBlock IDs.
  inline bool Dominates(uint32_t a, uint32_t b) const {
    return tree_.Dominates(a, b);
  }

  // Returns true if BasicBlock A strictly dominates BasicBlock B.
  inline bool StrictlyDominates(const ir::BasicBlock* a,
                                const ir::BasicBlock* b) const {
    if (!a || !b) return false;
    return StrictlyDominates(a->id(), b->id());
  }

  // Returns true if BasicBlock A strictly dominates BasicBlock B. Same as above
  // only using the BasicBlock IDs.
  inline bool StrictlyDominates(uint32_t a, uint32_t b) const {
    return tree_.StrictlyDominates(a, b);
  }

  // Return the immediate dominator of node A or return NULL if it is an entry
  // node.
  inline ir::BasicBlock* ImmediateDominator(const ir::BasicBlock* node) const {
    if (!node) return nullptr;
    return tree_.ImmediateDominator(node);
  }

  // Return the immediate dominator of node A or return NULL if it is an entry
  // node. Same as above but operates on IDs.
  inline ir::BasicBlock* ImmediateDominator(uint32_t node_id) const {
    return tree_.ImmediateDominator(node_id);
  }

  // Returns true if A is reachable from the entry.
  inline bool IsReachable(const ir::BasicBlock* node) const {
    if (!node) return false;
    return tree_.ReachableFromRoots(node->id());
  }

  // Returns true if A is reachable from the entry.
  inline bool IsReachable(uint32_t node_id) const {
    return tree_.ReachableFromRoots(node_id);
  }

  // Dump the tree structure into the given stream in the dot format.
  inline void DumpAsDot(std::ostream& out) const { tree_.DumpTreeAsDot(out); }

  // Returns true if this is a postdomiator tree.
  inline bool IsPostDominator() const { return tree_.IsPostDominator(); }

  // Return the tree itself for manual operations, such as traversing the roots.
  // For normal dominance relationships the methods above should be used.
  inline DominatorTree& GetDomTree() { return tree_; }
  inline const DominatorTree& GetDomTree() const { return tree_; }

 protected:
  DominatorTree tree_;
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
  // Gets the dominator analysis for function F.
  DominatorAnalysis* GetDominatorAnalysis(const ir::Function* f) {
    if (dominator_trees_.find(f) == dominator_trees_.end()) {
      dominator_trees_[f].InitializeTree(f);
    }

    return &dominator_trees_[f];
  }

  // Gets the postdominator analysis for function F.
  PostDominatorAnalysis* GetPostDominatorAnalysis(const ir::Function* f) {
    if (post_dominator_trees_.find(f) == post_dominator_trees_.end()) {
      post_dominator_trees_[f].InitializeTree(f);
    }

    return &post_dominator_trees_[f];
  }

 private:
  // Each function in the module will create its own dominator tree. We cache
  // the result so it doesn't need to be rebuilt each time.
  std::map<const ir::Function*, DominatorAnalysis> dominator_trees_;
  std::map<const ir::Function*, PostDominatorAnalysis> post_dominator_trees_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DOMINATOR_ANALYSIS_PASS_H_
