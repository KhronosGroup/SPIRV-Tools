// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef SPVTOOLS_CFA_H_
#define SPVTOOLS_CFA_H_

#include <algorithm>
#include <cassert>
#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using std::find;
using std::function;
using std::get;
using std::pair;
using std::unordered_map;
using std::unordered_set;
using std::vector;

namespace spvtools {

// Control Flow Analysis of control flow graphs of basic block nodes |BB|.
template<class BB> class CFA {
  using bb_ptr = BB*;
  using cbb_ptr = const BB*;
  using bb_iter = typename std::vector<BB*>::const_iterator;
  using get_blocks_func =
    std::function<const std::vector<BB*>*(const BB*)>;

  struct block_info {
    cbb_ptr block;  ///< pointer to the block
    bb_iter iter;   ///< Iterator to the current child node being processed
  };

  /// Returns true if a block with @p id is found in the @p work_list vector
  ///
  /// @param[in] work_list  Set of blocks visited in the the depth first traversal
  ///                       of the CFG
  /// @param[in] id         The ID of the block being checked
  ///
  /// @return true if the edge work_list.back().block->id() => id is a back-edge
  static bool FindInWorkList(
      const std::vector<block_info>& work_list, uint32_t id);

public:
  /// @brief Depth first traversal starting from the \p entry BasicBlock
  ///
  /// This function performs a depth first traversal from the \p entry
  /// BasicBlock and calls the pre/postorder functions when it needs to process
  /// the node in pre order, post order. It also calls the backedge function
  /// when a back edge is encountered.
  ///
  /// @param[in] entry      The root BasicBlock of a CFG
  /// @param[in] successor_func  A function which will return a pointer to the
  ///                            successor nodes
  /// @param[in] preorder   A function that will be called for every block in a
  ///                       CFG following preorder traversal semantics
  /// @param[in] postorder  A function that will be called for every block in a
  ///                       CFG following postorder traversal semantics
  /// @param[in] backedge   A function that will be called when a backedge is
  ///                       encountered during a traversal
  /// NOTE: The @p successor_func and predecessor_func each return a pointer to a
  /// collection such that iterators to that collection remain valid for the
  /// lifetime of the algorithm.
  static void DepthFirstTraversal(const BB* entry,
    get_blocks_func successor_func,
    std::function<void(cbb_ptr)> preorder,
    std::function<void(cbb_ptr)> postorder,
    std::function<void(cbb_ptr, cbb_ptr)> backedge);
  
  /// @brief Calculates dominator edges for a set of blocks
  ///
  /// Computes dominators using the algorithm of Cooper, Harvey, and Kennedy
  /// "A Simple, Fast Dominance Algorithm", 2001.
  ///
  /// The algorithm assumes there is a unique root node (a node without
  /// predecessors), and it is therefore at the end of the postorder vector.
  ///
  /// This function calculates the dominator edges for a set of blocks in the CFG.
  /// Uses the dominator algorithm by Cooper et al.
  ///
  /// @param[in] postorder        A vector of blocks in post order traversal order
  ///                             in a CFG
  /// @param[in] predecessor_func Function used to get the predecessor nodes of a
  ///                             block
  ///
  /// @return the dominator tree of the graph, as a vector of pairs of nodes.
  /// The first node in the pair is a node in the graph. The second node in the
  /// pair is its immediate dominator in the sense of Cooper et.al., where a block
  /// without predecessors (such as the root node) is its own immediate dominator.
  static vector<pair<BB*, BB*>> CalculateDominators(
    const vector<cbb_ptr>& postorder, get_blocks_func predecessor_func);
};

template<class BB> bool CFA<BB>::FindInWorkList(const vector<block_info>& work_list,
                                                uint32_t id) {
  for (const auto b : work_list) {
    if (b.block->id() == id) return true;
  }
  return false;
}

template<class BB> void CFA<BB>::DepthFirstTraversal(const BB* entry,
  get_blocks_func successor_func,
  function<void(cbb_ptr)> preorder,
  function<void(cbb_ptr)> postorder,
  function<void(cbb_ptr, cbb_ptr)> backedge) {
  unordered_set<uint32_t> processed;

  /// NOTE: work_list is the sequence of nodes from the root node to the node
  /// being processed in the traversal
  vector<block_info> work_list;
  work_list.reserve(10);

  work_list.push_back({ entry, begin(*successor_func(entry)) });
  preorder(entry);
  processed.insert(entry->id());

  while (!work_list.empty()) {
    block_info& top = work_list.back();
    if (top.iter == end(*successor_func(top.block))) {
      postorder(top.block);
      work_list.pop_back();
    }
    else {
      BB* child = *top.iter;
      top.iter++;
      if (FindInWorkList(work_list, child->id())) {
        backedge(top.block, child);
      }
      if (processed.count(child->id()) == 0) {
        preorder(child);
        work_list.emplace_back(
          block_info{ child, begin(*successor_func(child)) });
        processed.insert(child->id());
      }
    }
  }
}

template<class BB>
vector<pair<BB*, BB*>> CFA<BB>::CalculateDominators(
  const vector<cbb_ptr>& postorder, get_blocks_func predecessor_func) {
  struct block_detail {
    size_t dominator;  ///< The index of blocks's dominator in post order array
    size_t postorder_index;  ///< The index of the block in the post order array
  };
  const size_t undefined_dom = postorder.size();

  unordered_map<cbb_ptr, block_detail> idoms;
  for (size_t i = 0; i < postorder.size(); i++) {
    idoms[postorder[i]] = { undefined_dom, i };
  }
  idoms[postorder.back()].dominator = idoms[postorder.back()].postorder_index;

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto b = postorder.rbegin() + 1; b != postorder.rend(); ++b) {
      const vector<BB*>& predecessors = *predecessor_func(*b);
      // Find the first processed/reachable predecessor that is reachable
      // in the forward traversal.
      auto res = find_if(begin(predecessors), end(predecessors),
        [&idoms, undefined_dom](BB* pred) {
        return idoms.count(pred) &&
          idoms[pred].dominator != undefined_dom;
      });
      if (res == end(predecessors)) continue;
      const BB* idom = *res;
      size_t idom_idx = idoms[idom].postorder_index;

      // all other predecessors
      for (const auto* p : predecessors) {
        if (idom == p) continue;
        // Only consider nodes reachable in the forward traversal.
        // Otherwise the intersection doesn't make sense and will never
        // terminate.
        if (!idoms.count(p)) continue;
        if (idoms[p].dominator != undefined_dom) {
          size_t finger1 = idoms[p].postorder_index;
          size_t finger2 = idom_idx;
          while (finger1 != finger2) {
            while (finger1 < finger2) {
              finger1 = idoms[postorder[finger1]].dominator;
            }
            while (finger2 < finger1) {
              finger2 = idoms[postorder[finger2]].dominator;
            }
          }
          idom_idx = finger1;
        }
      }
      if (idoms[*b].dominator != idom_idx) {
        idoms[*b].dominator = idom_idx;
        changed = true;
      }
    }
  }

  vector<pair<bb_ptr, bb_ptr>> out;
  for (auto idom : idoms) {
    // NOTE: performing a const cast for convenient usage with
    // UpdateImmediateDominators
    out.push_back({ const_cast<BB*>(get<0>(idom)),
      const_cast<BB*>(postorder[get<1>(idom).dominator]) });
  }
  return out;
}

} // namespace spvtools

#endif  // SPVTOOLS_CFA_H_
