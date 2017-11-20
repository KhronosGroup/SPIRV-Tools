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

#include "dominator_tree.h"
#include <iostream>
#include "cfa.h"

namespace spvtools {
namespace opt {

// This helper class is basically a massive workaround for the current way that
// depth first is implemented
// TODO: Either clean this up with a nicer way of doing it all or reimplememt
// parts of DFS to avoid needing these functions
class BasicBlockSuccessorHelper {
 public:
  BasicBlockSuccessorHelper(ir::Function* func) : F(func) {}

  using GetBlocksFunction =
      std::function<const std::vector<ir::BasicBlock*>*(const ir::BasicBlock*)>;

  // Returns the list of predessor functions
  // TODO: Just a hack to get this working, doesn't even check if pred is in the
  // list
  GetBlocksFunction GetPredFunctor() {
    return [&](const ir::BasicBlock* BB) {
      auto v = &Pred[BB];
      return v;
    };
  }

  // Returns a vector of the list of successor nodes from a given node
  // TODO: As above
  GetBlocksFunction GetSuccessorFunctor() {
    return [&](const ir::BasicBlock* BB) {
      //   Successors.clear();
      auto findResult = Successors.find(BB);

      if (findResult != Successors.end()) {
        return &findResult->second;
      } else {
        // If we haven't already found a successor add it to the list
        Successors[BB] = {};

        BB->ForEachSuccessorLabel([&](const uint32_t successorID) {

          // TODO: If we keep somthing like this, avoid going over the full N
          // functions each time
          for (auto itr = F->begin(); itr < F->end(); ++itr) {
            ir::BasicBlock* bb = &*itr;
            if (successorID == bb->id()) {
              Successors[BB].push_back(bb);

              if (Pred.find(bb) == Pred.end()) {
                Pred[bb] = {};
              }

              if (std::find(Pred[bb].begin(), Pred[bb].end(), BB) ==
                  Pred[bb].end())
                Pred[bb].push_back(const_cast<ir::BasicBlock*>(BB));
            }
          }
        });

        return &Successors[BB];
      }
    };
  }

 private:
  ir::Function* F;
  std::map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>> Successors;
  std::map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>> Pred;
};

const ir::BasicBlock* DominatorTree::GetImmediateDominatorOrNull(
    uint32_t A) const {
  auto itr = Nodes.find(A);
  if (itr != Nodes.end()) {
    return itr->second.Parent->BB;
  }

  return nullptr;
}

const ir::BasicBlock* DominatorTree::GetImmediateDominatorOrNull(
    const ir::BasicBlock* A) const {
  return GetImmediateDominatorOrNull(A->id());
}

bool DominatorTree::StrictlyDominates(uint32_t A, uint32_t B) const {
  if (A == B) return false;
  return Dominates(A, B);
}

bool DominatorTree::StrictlyDominates(const ir::BasicBlock* A,
                                      const ir::BasicBlock* B) const {
  return DominatorTree::StrictlyDominates(A->id(), B->id());
}

bool DominatorTree::Dominates(uint32_t A, uint32_t B) const {
  // Node A dominates node B if they are the same
  if (A == B) return true;

  const DominatorTreeNode* nodeA = &Nodes.find(A)->second;
  const DominatorTreeNode* nodeB = &Nodes.find(B)->second;

  if (nodeA->DepthFirstInCount < nodeB->DepthFirstInCount &&
      nodeA->DepthFirstOutCount > nodeB->DepthFirstOutCount) {
    return true;
  }

  return false;
}

bool DominatorTree::Dominates(const ir::BasicBlock* A,
                              const ir::BasicBlock* B) const {
  return Dominates(A->id(), B->id());
}

DominatorTree::DominatorTreeNode* DominatorTree::GetOrInsertNode(
    const ir::BasicBlock* BB) {
  uint32_t id = BB->id();
  if (Nodes.find(id) == Nodes.end()) {
    Nodes[id] = {BB};
  }

  return &Nodes[id];
}

void DominatorTree::GetDominatorEdges(
    const ir::Function* F,
    std::vector<std::pair<ir::BasicBlock*, ir::BasicBlock*>>& edges) {
  // Ignore preorder operation
  auto nop_preorder = [](const ir::BasicBlock*) {};

  // Ignore backedge operation
  auto nop_backedge = [](const ir::BasicBlock*, const ir::BasicBlock*) {};

  // Each time the depth first traversal calls the postorder callback
  // std::function we push that node into the postorder vector to create our
  // postorder list
  std::vector<const ir::BasicBlock*> postorder;
  auto postorder_function = [&](const ir::BasicBlock* b) {
    postorder.push_back(b);
  };

  // TODO: Refactor helper class and get rid of this const cast
  BasicBlockSuccessorHelper helper{const_cast<ir::Function*>(F)};

  // The successor function tells DepthFirstTraversal how to move to successive
  // nodes by providing an interface to get a list of successor nodes from any
  // given node
  auto successorFunctor = helper.GetSuccessorFunctor();
  CFA<ir::BasicBlock>::DepthFirstTraversal(F->entry().get(), successorFunctor,
                                           nop_preorder, postorder_function,
                                           nop_backedge);

  // predecessorFunctor does the same as the successor functor but for all nodes
  // preceding a given node
  auto predecessorFunctor = helper.GetPredFunctor();
  edges =
      CFA<ir::BasicBlock>::CalculateDominators(postorder, predecessorFunctor);
}

void DominatorTree::InitializeTree(const ir::Function* F) {
  // Skip over empty functions
  if (F->cbegin() == F->cend()) {
    return;
  }

  // Get the immedate dominator for each node
  std::vector<std::pair<ir::BasicBlock*, ir::BasicBlock*>> edges;
  GetDominatorEdges(F, edges);

  // Transform the vector<pair> into the tree structure which we can use to
  // efficiently query dominace
  for (auto edge : edges) {
    uint32_t nodeID = edge.first->id();
    uint32_t dominatorID = edge.second->id();

    DominatorTreeNode* first = GetOrInsertNode(edge.first);

    if (nodeID == dominatorID) continue;

    DominatorTreeNode* second = GetOrInsertNode(edge.second);

    first->Parent = second;
    second->Children.push_back(first);
  }

  // Locate the root of the tree
  auto itr = Nodes.find(F->entry()->id());
  if (itr != Nodes.end()) {
    Root = &itr->second;
  }

  int index = 0;
  auto preFunc = [&](const DominatorTreeNode* node) {
    const_cast<DominatorTreeNode*>(node)->DepthFirstInCount = ++index;
  };

  auto postFunc = [&](const DominatorTreeNode* node) {
    const_cast<DominatorTreeNode*>(node)->DepthFirstOutCount = ++index;
  };

  auto ignore_e = [](const DominatorTreeNode*, const DominatorTreeNode*) {};

  auto getSucc = [&](const DominatorTreeNode* node) { return &node->Children; };

  CFA<DominatorTreeNode>::DepthFirstTraversal(Root, getSucc, preFunc, postFunc,
                                              ignore_e);
}

void DominatorTree::DumpTreeAsDot(std::ostream& OutStream) const {
  if (!Root) return;

  OutStream << "digraph {\n";
  Visit(Root, [&](const DominatorTreeNode* node) {
    OutStream << node->BB->id() << "[label=\"" << node->BB->id() << "\"];\n";
    if (node->Parent) {
      OutStream << node->Parent->BB->id() << " -> " << node->BB->id() << ";\n";
    }
  });
  OutStream << "}\n";
}

void DominatorTree::Visit(
    const DominatorTreeNode* Node,
    std::function<void(const DominatorTreeNode*)> func) const {
  // Apply the function to the node
  func(Node);

  // Apply the function to every child node
  for (const DominatorTreeNode* child : Node->Children) {
    Visit(child, func);
  }
}

}  // opt
}  // spvtools
