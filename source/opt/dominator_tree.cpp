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

#include <iostream>
#include <memory>
#include <set>

#include "cfa.h"
#include "dominator_tree.h"

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
static void DepthFirstSearch(const BBType* bb, SuccessorLambda successors,
                             PreLambda pre, PostLambda post) {
  // Ignore backedge operation.
  auto nop_backedge = [](const BBType*, const BBType*) {};
  CFA<BBType>::DepthFirstTraversal(bb, successors, pre, post, nop_backedge);
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
static void DepthFirstSearchPostOrder(const BBType* bb,
                                      SuccessorLambda successors,
                                      PostLambda post) {
  // Ignore preorder operation.
  auto nop_preorder = [](const BBType*) {};
  DepthFirstSearch(bb, successors, nop_preorder, post);
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
  BasicBlockSuccessorHelper(Function& func, BasicBlock* dummy_start_node,
                            bool post);

  // CFA::CalculateDominators requires std::vector<BasicBlock*>.
  using GetBlocksFunction =
      std::function<const std::vector<BasicBlock*>*(const BasicBlock*)>;

  // Returns the list of predecessor functions.
  GetBlocksFunction GetPredFunctor() {
    return [&](const BasicBlock* bb) {
      BasicBlockListTy* v = &predecessors_[bb];
      return v;
    };
  }

  // Returns a vector of the list of successor nodes from a given node.
  GetBlocksFunction GetSuccessorFunctor() {
    return [&](const BasicBlock* bb) {
      BasicBlockListTy* v = &successors_[bb];
      return v;
    };
  }

 private:
  bool invert_graph_;
  BasicBlockMapTy successors_;
  BasicBlockMapTy predecessors_;

  // Build a bi-directional graph from the CFG of F.
  // If invert_graph_ is true, all edge are reverted (successors becomes
  // predecessors and vise versa).
  // For convenience, the start of the graph is dummyStartNode. The dominator
  // tree construction requires a unique entry node, which cannot be guarantied
  // for the postdominator graph. The dummyStartNode BB is here to gather all
  // entry nodes.
  void CreateSuccessorMap(Function& f, BasicBlock* dummy_start_node);
};

template <typename BBType>
BasicBlockSuccessorHelper<BBType>::BasicBlockSuccessorHelper(
    Function& func, BasicBlock* dummy_start_node, bool invert)
    : invert_graph_(invert) {
  CreateSuccessorMap(func, dummy_start_node);
}

template <typename BBType>
void BasicBlockSuccessorHelper<BBType>::CreateSuccessorMap(
    Function& f, BasicBlock* dummy_start_node) {
  std::map<uint32_t, BasicBlock*> id_to_BB_map;
  auto GetSuccessorBasicBlock = [&](uint32_t successor_id) {
    BasicBlock*& Succ = id_to_BB_map[successor_id];
    if (!Succ) {
      for (BasicBlock& BBIt : f) {
        if (successor_id == BBIt.id()) {
          Succ = &BBIt;
          break;
        }
      }
    }
    return Succ;
  };

  if (invert_graph_) {
    // For the post dominator tree, we see the inverted graph.
    // successors_ in the inverted graph are the predecessors in the CFG.
    // The tree construction requires 1 entry point, so we add a dummy node
    // that is connected to all function exiting basic blocks.
    // An exiting basic block is a block with an OpKill, OpUnreachable,
    // OpReturn or OpReturnValue as terminator instruction.
    for (BasicBlock& bb : f) {
      if (bb.hasSuccessor()) {
        BasicBlockListTy& pred_list = predecessors_[&bb];
        bb.ForEachSuccessorLabel([&](const uint32_t successor_id) {
          BasicBlock* succ = GetSuccessorBasicBlock(successor_id);
          // Inverted graph: our successors in the CFG
          // are our predecessors in the inverted graph.
          successors_[succ].push_back(&bb);
          pred_list.push_back(succ);
        });
      } else {
        successors_[dummy_start_node].push_back(&bb);
        predecessors_[&bb].push_back(dummy_start_node);
      }
    }
  } else {
    // Technically, this is not needed, but it unifies
    // the handling of dominator and postdom tree later on.
    successors_[dummy_start_node].push_back(f.entry().get());
    predecessors_[f.entry().get()].push_back(dummy_start_node);
    for (BasicBlock& bb : f) {
      BasicBlockListTy& succ_list = successors_[&bb];

      bb.ForEachSuccessorLabel([&](const uint32_t successor_id) {
        BasicBlock* succ = GetSuccessorBasicBlock(successor_id);
        succ_list.push_back(succ);
        predecessors_[succ].push_back(&bb);
      });
    }
  }
}

}  // namespace

namespace spvtools {
namespace opt {

bool DominatorTree::StrictlyDominates(uint32_t a, uint32_t b) const {
  if (a == b) return false;
  return Dominates(a, b);
}

bool DominatorTree::StrictlyDominates(const ir::BasicBlock* a,
                                      const ir::BasicBlock* b) const {
  return DominatorTree::StrictlyDominates(a->id(), b->id());
}

bool DominatorTree::Dominates(uint32_t a, uint32_t b) const {
  // Check that both of the inputs are actual nodes.
  auto a_itr = nodes_.find(a);
  auto b_itr = nodes_.find(b);
  if (a_itr == nodes_.end() || b_itr == nodes_.end()) return false;

  // Node A dominates node B if they are the same.
  if (a == b) return true;
  const DominatorTreeNode* nodeA = &a_itr->second;
  const DominatorTreeNode* nodeB = &b_itr->second;

  if (nodeA->dfs_num_pre_ < nodeB->dfs_num_pre_ &&
      nodeA->dfs_num_post_ > nodeB->dfs_num_post_) {
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

ir::BasicBlock* DominatorTree::ImmediateDominator(uint32_t a) const {
  // Check that A is a valid node in the tree.
  auto a_itr = nodes_.find(a);
  if (a_itr == nodes_.end()) return nullptr;

  const DominatorTreeNode* node = &a_itr->second;

  if (node->parent_ == nullptr) {
    return nullptr;
  }

  return node->parent_->bb_;
}

DominatorTreeNode* DominatorTree::GetOrInsertNode(ir::BasicBlock* bb) {
  DominatorTreeNode* dtn = nullptr;

  std::map<uint32_t, DominatorTreeNode>::iterator node_iter =
      nodes_.find(bb->id());
  if (node_iter == nodes_.end()) {
    dtn = &nodes_.emplace(std::make_pair(bb->id(), DominatorTreeNode{bb}))
               .first->second;
  } else
    dtn = &node_iter->second;

  return dtn;
}

void DominatorTree::GetDominatorEdges(
    const ir::Function* f, ir::BasicBlock* dummy_start_node,
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
      *const_cast<ir::Function*>(f), dummy_start_node, postdominator_};

  // The successor function tells DepthFirstTraversal how to move to successive
  // nodes by providing an interface to get a list of successor nodes from any
  // given node.
  auto successor_functor = helper.GetSuccessorFunctor();

  // The predecessor functor does the same as the successor functor
  // but for all nodes preceding a given node.
  auto predecessor_functor = helper.GetPredFunctor();

  // If we're building a post dominator tree we traverse the tree in reverse
  // using the predecessor function in place of the successor function and vice
  // versa.
  DepthFirstSearchPostOrder(dummy_start_node, successor_functor,
                            postorder_function);
  edges =
      CFA<ir::BasicBlock>::CalculateDominators(postorder, predecessor_functor);
}

void DominatorTree::InitializeTree(const ir::Function* f) {
  ClearTree();

  // Skip over empty functions.
  if (f->cbegin() == f->cend()) {
    return;
  }

  std::unique_ptr<ir::Instruction> dummy_label{new ir::Instruction(
      f->GetParent()->context(), SpvOp::SpvOpLabel, 0, -1, {})};
  // Create a dummy start node which will point to all of the roots of the tree
  // to allow us to work with a singular root.
  ir::BasicBlock dummy_start_node(std::move(dummy_label));

  // Get the immediate dominator for each node.
  std::vector<std::pair<ir::BasicBlock*, ir::BasicBlock*>> edges;
  GetDominatorEdges(f, &dummy_start_node, edges);

  // Transform the vector<pair> into the tree structure which we can use to
  // efficiently query dominance.
  for (auto edge : edges) {
    if (&dummy_start_node == edge.first) continue;
    DominatorTreeNode* first = GetOrInsertNode(edge.first);

    if (&dummy_start_node == edge.second) {
      if (std::find(roots_.begin(), roots_.end(), first) == roots_.end())
        roots_.push_back(first);
      continue;
    }

    DominatorTreeNode* second = GetOrInsertNode(edge.second);

    first->parent_ = second;
    second->childrens_.push_back(first);
  }

  int index = 0;
  auto preFunc = [&](const DominatorTreeNode* node) {
    const_cast<DominatorTreeNode*>(node)->dfs_num_pre_ = ++index;
  };

  auto postFunc = [&](const DominatorTreeNode* node) {
    const_cast<DominatorTreeNode*>(node)->dfs_num_post_ = ++index;
  };

  auto getSucc = [&](const DominatorTreeNode* node) {
    return &node->childrens_;
  };

  for (auto root : *this) DepthFirstSearch(root, getSucc, preFunc, postFunc);
}

void DominatorTree::DumpTreeAsDot(std::ostream& out_stream) const {
  out_stream << "digraph {\n";
  out_stream << "Dummy [label=\"Entry\"];\n";
  for (auto Root : *this) {
    Visit(Root, [&](const DominatorTreeNode* node) {

      // Print the node.
      if (node->bb_) {
        out_stream << node->bb_->id() << "[label=\"" << node->bb_->id()
                   << "\"];\n";
      }

      // Print the arrow from the parent to this node. Entry nodes will not have
      // parents so draw them as children from the dummy node.
      if (node->parent_) {
        out_stream << node->parent_->bb_->id() << " -> " << node->bb_->id()
                   << ";\n";
      } else {
        out_stream << "Dummy -> " << node->bb_->id() << " [style=dotted];\n";
      }
    });
  }
  out_stream << "}\n";
}

void DominatorTree::Visit(
    const DominatorTreeNode* node,
    std::function<void(const DominatorTreeNode*)> func) const {
  // Apply the function to the node.
  func(node);

  // Apply the function to every child node.
  for (const DominatorTreeNode* child : node->childrens_) {
    Visit(child, func);
  }
}

}  // namespace opt
}  // namespace spvtools
