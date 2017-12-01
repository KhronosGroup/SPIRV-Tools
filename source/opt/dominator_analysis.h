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

namespace spvtools {
namespace opt {

// Interface to perform dominator or postdominator analysis on a given function.
class DominatorAnalysisBase {
 public:
  explicit DominatorAnalysisBase(bool is_post_dom) : tree_(is_post_dom) {}

  // Calculate the dominator (or postdominator) tree for given function |f|.
  inline void InitializeTree(const ir::Function* f, const ir::CFG& cfg) {
    tree_.InitializeTree(f, cfg);
  }

  // Return true if BasicBlock |a| dominates BasicBlock |b|.
  inline bool Dominates(const ir::BasicBlock* a,
                        const ir::BasicBlock* b) const {
    if (!a || !b) return false;
    return Dominates(a->id(), b->id());
  }

  // Return true if BasicBlock |a| dominates BasicBlock |b|. Same as above only
  // using the BasicBlock IDs.
  inline bool Dominates(uint32_t a, uint32_t b) const {
    return tree_.Dominates(a, b);
  }

  // Return true if BasicBlock |a| strictly dominates BasicBlock |b|.
  inline bool StrictlyDominates(const ir::BasicBlock* a,
                                const ir::BasicBlock* b) const {
    if (!a || !b) return false;
    return StrictlyDominates(a->id(), b->id());
  }

  // Return true if BasicBlock |a| strictly dominates BasicBlock |b|. Same as
  // above only using the BasicBlock IDs.
  inline bool StrictlyDominates(uint32_t a, uint32_t b) const {
    return tree_.StrictlyDominates(a, b);
  }

  // Return the immediate dominator of |node| or return nullptr if it is has no
  // dominator.
  inline ir::BasicBlock* ImmediateDominator(const ir::BasicBlock* node) const {
    if (!node) return nullptr;
    return tree_.ImmediateDominator(node);
  }

  // Return the immediate dominator of |node_id| or return nullptr if it is has
  // no dominator. Same as above but operates on IDs.
  inline ir::BasicBlock* ImmediateDominator(uint32_t node_id) const {
    return tree_.ImmediateDominator(node_id);
  }

  // Return true if |node| is reachable from the entry.
  inline bool IsReachable(const ir::BasicBlock* node) const {
    if (!node) return false;
    return tree_.ReachableFromRoots(node->id());
  }

  // Return true if |node_id| is reachable from the entry.
  inline bool IsReachable(uint32_t node_id) const {
    return tree_.ReachableFromRoots(node_id);
  }

  // Dump the tree structure into the given |out| stream in the dot format.
  inline void DumpAsDot(std::ostream& out) const { tree_.DumpTreeAsDot(out); }

  // Return true if this is a postdominator tree.
  inline bool IsPostDominator() const { return tree_.IsPostDominator(); }

  // Return the tree itself for manual operations, such as traversing the roots.
  // For normal dominance relationships the methods above should be used.
  inline DominatorTree& GetDomTree() { return tree_; }
  inline const DominatorTree& GetDomTree() const { return tree_; }

  // Force the dominator tree to be removed
  inline void ClearTree() { tree_.ClearTree(); }

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

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DOMINATOR_ANALYSIS_PASS_H_
