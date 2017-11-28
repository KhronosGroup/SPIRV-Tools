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

#ifndef LIBSPIRV_OPT_DOMINATOR_ANALYSIS_TREE_H_
#define LIBSPIRV_OPT_DOMINATOR_ANALYSIS_TREE_H_

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "module.h"

namespace spvtools {
namespace opt {
// This helper struct forms the nodes in the tree, with each node containing its
// children. It also contains two values, for the pre and post indexes in the
// tree which are used to compare two nodes.
struct DominatorTreeNode {
  explicit DominatorTreeNode(ir::BasicBlock* bb)
      : bb_(bb),
        parent_(nullptr),
        childrens_({}),
        dfs_num_pre_(-1),
        dfs_num_post_(-1) {}

  inline uint32_t id() const { return bb_->id(); }

  ir::BasicBlock* bb_;
  DominatorTreeNode* parent_;
  std::vector<DominatorTreeNode*> childrens_;

  // These indexes are used to compare two given nodes. A node is a child or
  // grandchild of another node if its preorder index is greater than the
  // first nodes preorder index AND if its postorder index is less than the
  // first nodes postorder index.
  int dfs_num_pre_;
  int dfs_num_post_;
};

// A class representing a tree of BasicBlocks in a given function, where each
// node is dominated by its parent.
class DominatorTree {
 public:
  using DominatorTreeNodeList = std::vector<DominatorTreeNode*>;
  using iterator = DominatorTreeNodeList::iterator;
  using const_iterator = DominatorTreeNodeList::const_iterator;

  DominatorTree() : postdominator_(false) {}
  explicit DominatorTree(bool post) : postdominator_(post) {}

  iterator begin() { return roots_.begin(); }
  iterator end() { return roots_.end(); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  const_iterator cbegin() const { return roots_.begin(); }
  const_iterator cend() const { return roots_.end(); }

  // Get the unique root of the tree.
  // It is guaranteed to work on a dominator tree.
  // A postdominator may have more than one element.
  DominatorTreeNode* GetRoot() {
    assert(roots_.size() == 1);
    return *begin();
  }

  const DominatorTreeNode* GetRoot() const {
    assert(roots_.size() == 1);
    return *begin();
  }

  // Dumps the tree in the graphvis dot format into the stream.
  void DumpTreeAsDot(std::ostream& OutStream) const;

  // Build the (post-)dominator tree for the function |F|
  // Any existing data will be overwritten
  void InitializeTree(const ir::Function* F);

  // Check if BasicBlock B is a dominator of BasicBlock A.
  bool Dominates(const ir::BasicBlock* A, const ir::BasicBlock* B) const;

  // Check if BasicBlock B is a dominator of BasicBlock A. This function uses
  // the IDs of A and B.
  bool Dominates(uint32_t A, uint32_t B) const;

  // Check if BasicBlock A strictly dominates B
  bool StrictlyDominates(const ir::BasicBlock* A,
                         const ir::BasicBlock* B) const;

  bool StrictlyDominates(uint32_t A, uint32_t B) const;

  // Returns the immediate dominator of basicblock A.
  ir::BasicBlock* ImmediateDominator(const ir::BasicBlock* A) const;

  // Returns the immediate dominator of basicblock A.
  ir::BasicBlock* ImmediateDominator(uint32_t A) const;

  // Returns true if BasicBlock A is reachable by this tree. A node would be
  // unreachable if it cannot be reached by traversal from the start node or for
  // a postdominator tree, cannot be reached from the exit nodes.
  inline bool ReachableFromRoots(const ir::BasicBlock* A) const {
    if (!A) return false;
    return ReachableFromRoots(A->id());
  }

  // Same as the above method but takes in the ID of the BasicBlock rather than
  // the BasicBlock itself.
  bool ReachableFromRoots(uint32_t A) const;

  // Returns true if this tree is a post dominator tree or not.
  bool IsPostDominator() const { return postdominator_; }

  // Clean up the tree.
  void ClearTree() {
    nodes_.clear();
    roots_.clear();
  }

 private:
  // Adds the BasicBlock to the tree structure if it doesn't already exist.
  DominatorTreeNode* GetOrInsertNode(ir::BasicBlock* BB);

  // Applies the std::function 'func' to 'node' then applies it to node's
  // children.
  void Visit(const DominatorTreeNode* node,
             std::function<void(const DominatorTreeNode*)> func) const;

  // Wrapper function which gets the list of BasicBlock->DominatingBasicBlock
  // from the CFA and stores it in the edges parameter.
  //
  // The |edges| vector will contain the dominator tree as pairs of nodes.
  // The first node in the pair is a node in the graph. The second node in the
  // pair is its immediate dominator.
  // The root of the tree has him self as immediate dominator.
  void GetDominatorEdges(
      const ir::Function* F, ir::BasicBlock* DummyStartNode,
      std::vector<std::pair<ir::BasicBlock*, ir::BasicBlock*>>& edges);

  // The roots of the tree.
  std::vector<DominatorTreeNode*> roots_;

  // Pairs each basic block id to the tree node containing that basic block.
  std::map<uint32_t, DominatorTreeNode> nodes_;

  // True if this is a post dominator tree.
  bool postdominator_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DOMINATOR_ANALYSIS_TREE_H_
