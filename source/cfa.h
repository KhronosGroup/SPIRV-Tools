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

} // namespace spvtools

#endif  // SPVTOOLS_CFA_H_
