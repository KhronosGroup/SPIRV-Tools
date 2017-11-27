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
#include "cfa.h"

#include <iostream>
#include <set>

using namespace spvtools;
using namespace spvtools::opt;

namespace {

// Wrapper around CFA::DepthFirstTraversal to provide an interface to perform
// depth first search on generic BasicBlock types. Will call post and pre order
// user defined functions during traversal
//
// BBType - BasicBlock type. Will either be ir::BasicBlock or DominatorTreeNode
// SuccessorLambda - Lamdba matching the signature of 'const
// std::vector<BBType>*(const BBType *A)'. Will return a vector of the nodes
// succeding BasicBlock A.
// PostLambda - Lamdba matching the signature of 'void (const BBType*)' will be
// called on each node traversed AFTER their children.
// PreLambda - Lamdba matching the signature of 'void (const BBType*)' will be
// called on each node traversed BEFORE their children.
template <typename BBType, typename SuccessorLambda, typename PreLambda,
          typename PostLambda>
static void depthFirstSearch(const BBType* BB, SuccessorLambda successors,
                             PreLambda pre, PostLambda post) {
  // Ignore backedge operation.
  auto nop_backedge = [](const BBType*, const BBType*) {};
  CFA<BBType>::DepthFirstTraversal(BB, successors, pre, post, nop_backedge);
}

// Wrapper around CFA::DepthFirstTraversal to provide an interface to perform
// depth first search on generic BasicBlock types. This overload is for only
// performing user defined post order.
//
// BBType - BasicBlock type. Will either be ir::BasicBlock or DominatorTreeNode
// SuccessorLambda - Lamdba matching the signature of 'const
// std::vector<BBType>*(const BBType *A)'. Will return a vector of the nodes
// succeding BasicBlock A.
// PostLambda - Lamdba matching the signature of 'void (const BBType*)' will be
// called on each node traversed after their children.
template <typename BBType, typename SuccessorLambda, typename PostLambda>
static void depthFirstSearchPostOrder(const BBType* BB,
                                      SuccessorLambda successors,
                                      PostLambda post) {
  // Ignore preorder operation.
  auto nop_preorder = [](const BBType*) {};
  depthFirstSearch(BB, successors, nop_preorder, post);
}

// Small type trait to get the function class type.
template <typename BBType>
struct GetFunctionClass {
  using FunctionType = ir::Function;
};

// This helper class is basically a massive workaround for the current way that
// depth first is implemented.
template <typename BBType>
class BasicBlockSuccessorHelper {
  // This should eventually become const ir::BasicBlock.
  using BasicBlock = BBType;
  using Function = typename GetFunctionClass<BBType>::FunctionType;

  using BasicBlockListTy = std::vector<BasicBlock*>;
  using BasicBlockMapTy = std::map<const BasicBlock*, BasicBlockListTy>;

 public:
  BasicBlockSuccessorHelper(Function& func, BasicBlock* dummyStartNode,
                            bool post);

  // CFA::CalculateDominators requires std::vector<BasicBlock*>.
  using GetBlocksFunction =
      std::function<const std::vector<BasicBlock*>*(const BasicBlock*)>;

  // Returns the list of predecessor functions.
  GetBlocksFunction GetPredFunctor() {
    return [&](const BasicBlock* BB) {
      BasicBlockListTy* v = &Predecessors[BB];
      return v;
    };
  }

  // Returns a vector of the list of successor nodes from a given node.
  GetBlocksFunction GetSuccessorFunctor() {
    return [&](const BasicBlock* BB) {
      BasicBlockListTy* v = &Successors[BB];
      return v;
    };
  }

 private:
  bool InvertGraph;
  BasicBlockMapTy Successors;
  BasicBlockMapTy Predecessors;

  // Build a bi-directional graph from the CFG of F.
  // If InvertGraph is true, all edge are reverted (successors becomes
  // predecessors and vise versa).
  // For convenience, the start of the graph is dummyStartNode. The dominator
  // tree construction requires a unique entry node, which cannot be guarantied
  // for the postdominator graph. The dummyStartNode BB is here to gather all
  // entry nodes.
  void CreateSuccessorMap(Function& F, BasicBlock* dummyStartNode);
};

template <typename BBType>
BasicBlockSuccessorHelper<BBType>::BasicBlockSuccessorHelper(
    Function& func, BasicBlock* dummyStartNode, bool Invert)
    : InvertGraph(Invert) {
  CreateSuccessorMap(func, dummyStartNode);
}

template <typename BBType>
void BasicBlockSuccessorHelper<BBType>::CreateSuccessorMap(
    Function& F, BasicBlock* dummyStartNode) {
  std::map<uint32_t, BasicBlock*> IDtoBBMap;
  auto GetSuccessorBasicBlock = [&](uint32_t successorID) {
    BasicBlock*& Succ = IDtoBBMap[successorID];
    if (!Succ) {
      for (BasicBlock& BBIt : F) {
        if (successorID == BBIt.id()) {
          Succ = &BBIt;
          break;
        }
      }
    }
    return Succ;
  };

  if (InvertGraph) {
    // For the post dominator tree, we see the inverted graph.
    // Successors in the inverted graph are the predecessors in the CFG.
    // The tree construction requires 1 entry point, so we add a dummy node
    // that is connected to all function exiting basic blocks.
    // An exiting basic block is a block with an OpKill, OpUnreachable,
    // OpReturn or OpReturnValue as terminator instruction.
    for (BasicBlock& BB : F) {
      if (BB.hasSuccessor()) {
        BasicBlockListTy& PredList = Predecessors[&BB];
        BB.ForEachSuccessorLabel([&](const uint32_t successorID) {
          BasicBlock* Succ = GetSuccessorBasicBlock(successorID);
          // Inverted graph: our successors in the CFG
          // are our predecessors in the inverted graph.
          Successors[Succ].push_back(&BB);
          PredList.push_back(Succ);
        });
      } else {
        Successors[dummyStartNode].push_back(&BB);
        Predecessors[&BB].push_back(dummyStartNode);
      }
    }
  } else {
    // Technically, this is not needed, but it unifies
    // the handling of dominator and postdom tree later on.
    Successors[dummyStartNode].push_back(F.entry().get());
    Predecessors[F.entry().get()].push_back(dummyStartNode);
    for (BasicBlock& BB : F) {
      BasicBlockListTy& SuccList = Successors[&BB];

      BB.ForEachSuccessorLabel([&](const uint32_t successorID) {
        BasicBlock* Succ = GetSuccessorBasicBlock(successorID);
        SuccList.push_back(Succ);
        Predecessors[Succ].push_back(&BB);
      });
    }
  }
}

}  // namespace

namespace spvtools {
namespace opt {

bool DominatorTree::StrictlyDominates(uint32_t A, uint32_t B) const {
  if (A == B) return false;
  return Dominates(A, B);
}

bool DominatorTree::StrictlyDominates(const ir::BasicBlock* A,
                                      const ir::BasicBlock* B) const {
  return DominatorTree::StrictlyDominates(A->id(), B->id());
}

bool DominatorTree::Dominates(uint32_t A, uint32_t B) const {
  // Check that both of the inputs are actual nodes.
  auto aItr = Nodes.find(A);
  auto bItr = Nodes.find(B);
  if (aItr == Nodes.end() || bItr == Nodes.end()) return false;

  // Node A dominates node B if they are the same.
  if (A == B) return true;
  const DominatorTreeNode* nodeA = &aItr->second;
  const DominatorTreeNode* nodeB = &bItr->second;

  if (nodeA->DepthFirstPreOrderIndex < nodeB->DepthFirstPreOrderIndex &&
      nodeA->DepthFirstPostOrderIndex > nodeB->DepthFirstPostOrderIndex) {
    return true;
  }

  return false;
}

bool DominatorTree::Dominates(const ir::BasicBlock* A,
                              const ir::BasicBlock* B) const {
  return Dominates(A->id(), B->id());
}

ir::BasicBlock* DominatorTree::ImmediateDominator(
    const ir::BasicBlock* A) const {
  return ImmediateDominator(A->id());
}

ir::BasicBlock* DominatorTree::ImmediateDominator(uint32_t A) const {
  // Check that A is a valid node in the tree.
  auto aItr = Nodes.find(A);
  if (aItr == Nodes.end()) return nullptr;

  const DominatorTreeNode* nodeA = &aItr->second;

  if (nodeA->Parent == nullptr) {
    return nullptr;
  }

  return nodeA->Parent->BB;
}

DominatorTreeNode* DominatorTree::GetOrInsertNode(ir::BasicBlock* BB) {
  DominatorTreeNode* DTN = nullptr;

  std::map<uint32_t, DominatorTreeNode>::iterator ItNode = Nodes.find(BB->id());
  if (ItNode == Nodes.end()) {
    DTN = &Nodes.emplace(std::make_pair(BB->id(), DominatorTreeNode{BB}))
               .first->second;
  } else
    DTN = &ItNode->second;

  return DTN;
}

void DominatorTree::GetDominatorEdges(
    const ir::Function* F, ir::BasicBlock* DummyStartNode,
    std::vector<std::pair<ir::BasicBlock*, ir::BasicBlock*>>& edges) {
  // Each time the depth first traversal calls the postorder callback
  // std::function we push that node into the postorder vector to create our
  // postorder list.
  std::vector<const ir::BasicBlock*> postorder;
  auto postorder_function = [&](const ir::BasicBlock* b) {
    postorder.push_back(b);
  };

  // CFA::CalculateDominators requires std::vector<ir::BasicBlock*>
  // BB are derived from F, so we need to const cast it at some point
  // no modification is made on F.
  BasicBlockSuccessorHelper<ir::BasicBlock> helper{
      *const_cast<ir::Function*>(F), DummyStartNode, PostDominator};

  // The successor function tells DepthFirstTraversal how to move to successive
  // nodes by providing an interface to get a list of successor nodes from any
  // given node.
  auto successorFunctor = helper.GetSuccessorFunctor();

  // PredecessorFunctor does the same as the successor functor but for all nodes
  // preceding a given node.
  auto predecessorFunctor = helper.GetPredFunctor();

  // If we're building a post dominator tree we traverse the tree in reverse
  // using the predecessor function in place of the successor function and vice
  // versa.
  depthFirstSearchPostOrder(DummyStartNode, successorFunctor,
                            postorder_function);
  edges =
      CFA<ir::BasicBlock>::CalculateDominators(postorder, predecessorFunctor);
}

void DominatorTree::InitializeTree(const ir::Function* F) {
  ResetTree();

  // Skip over empty functions.
  if (F->cbegin() == F->cend()) {
    return;
  }

  std::unique_ptr<ir::Instruction> DummyLabel{new ir::Instruction(
      F->GetParent()->context(), SpvOp::SpvOpLabel, 0, -1, {})};
  // Create a dummy start node which will point to all of the roots of the tree
  // to allow us to work with a singular root.
  ir::BasicBlock DummyStartNode(std::move(DummyLabel));

  // Get the immedate dominator for each node.
  std::vector<std::pair<ir::BasicBlock*, ir::BasicBlock*>> edges;
  GetDominatorEdges(F, &DummyStartNode, edges);

  // Transform the vector<pair> into the tree structure which we can use to
  // efficiently query dominace.
  for (auto edge : edges) {
    if (&DummyStartNode == edge.first) continue;
    DominatorTreeNode* first = GetOrInsertNode(edge.first);

    if (&DummyStartNode == edge.second) {
      if (std::find(Roots.begin(), Roots.end(), first) == Roots.end())
        Roots.push_back(first);
      continue;
    }

    DominatorTreeNode* second = GetOrInsertNode(edge.second);

    first->Parent = second;
    second->Children.push_back(first);
  }

  int index = 0;
  auto preFunc = [&](const DominatorTreeNode* node) {
    const_cast<DominatorTreeNode*>(node)->DepthFirstPreOrderIndex = ++index;
  };

  auto postFunc = [&](const DominatorTreeNode* node) {
    const_cast<DominatorTreeNode*>(node)->DepthFirstPostOrderIndex = ++index;
  };

  auto getSucc = [&](const DominatorTreeNode* node) { return &node->Children; };

  for (auto Root : *this) depthFirstSearch(Root, getSucc, preFunc, postFunc);
}

void DominatorTree::ResetTree() {
  // Clean up look-ups.
  Nodes.clear();
  Roots.clear();
}

void DominatorTree::DumpTreeAsDot(std::ostream& OutStream) const {
  OutStream << "digraph {\n";
  OutStream << "Dummy [label=\"Entry\"];\n";
  for (auto Root : *this) {
    Visit(Root, [&](const DominatorTreeNode* node) {

      // Print the node.
      if (node->BB) {
        OutStream << node->BB->id() << "[label=\"" << node->BB->id()
                  << "\"];\n";
      }

      // Print the arrow from the parent to this node. Entry nodes will not have
      // parents so draw them as children from the dummy node.
      if (node->Parent) {
        OutStream << node->Parent->BB->id() << " -> " << node->BB->id()
                  << ";\n";
      } else {
        OutStream << "Dummy -> " << node->BB->id() << " [style=dotted];\n";
      }
    });
  }
  OutStream << "}\n";
}

void DominatorTree::Visit(
    const DominatorTreeNode* Node,
    std::function<void(const DominatorTreeNode*)> func) const {
  // Apply the function to the node.
  func(Node);

  // Apply the function to every child node.
  for (const DominatorTreeNode* child : Node->Children) {
    Visit(child, func);
  }
}

// Internal methods to the node helper struct.
DominatorTreeNode::DominatorTreeNode(ir::BasicBlock* bb)
    : BB(bb),
      Parent(nullptr),
      Children({}),
      DepthFirstPreOrderIndex(-1),
      DepthFirstPostOrderIndex(-1) {}

uint32_t DominatorTreeNode::id() const { return BB->id(); }

}  // opt
}  // spvtools
