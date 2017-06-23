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

#ifndef LIBSPIRV_UTIL_MOVE_TO_FRONT_H_
#define LIBSPIRV_UTIL_MOVE_TO_FRONT_H_

#include <cassert>
#include <cstdint>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace spvutils {

// Log(n) move-to-front implementation. Implements two main functions:
// IdFromRank - access id by its 1-indexed rank in the move-to-front sequence.
// RankFromId - get the rank of the given id in the move-to-front sequence.
// Accessing an id with any of the two functions moves the id to the front
// of the sequence (rank of 1).
//
// The implementation is based on an AVL-based order statistic tree.
//
// Terminology
// id: SPIR-V id.
// rank: 1-indexed value showing how recently the id was accessed.
// node: handle used internally to access node data.
// size: size of the subtree of a node (including the node).
// height: distance from a node to the farthest leaf.
class MoveToFront {
  struct Node;
 public:
  explicit MoveToFront(size_t reserve_capacity = 128) {
    nodes_.reserve(reserve_capacity);

    // Create NIL node.
    nodes_.emplace_back(Node());
  }

  // Returns 1-indexed rank of id in the move-to-front sequence and moves
  // id to the front. Example:
  // Before the call: 4 8 2 1 7
  // RankFromId(8) returns 1
  // After the call: 8 4 2 1 7
  // If id is not in the list, and is equal to next_id_, then a new value is
  // inserted at the front. The function returns 0 in this case.
  // RankFromId(9) returns 0
  // After the call: 9 8 4 2 1 7
  // Calling the function with a value which is not in the list and is not
  // equal to next_id_ will result in assertion.
  size_t RankFromId(uint32_t id);

  // Returns id corresponding to a 1-indexed rank in the move-to-front sequence
  // and moves the id to the front. Example:
  // Before the call: 4 8 2 1 7
  // IdFromRank(1) returns 8
  // After the call: 8 4 2 1 7
  // If rank is 0, then a new id equal to next_id_ is inserted.
  // IdFromRank(0) returns 9
  // After the call:  9 8 4 2 1 7
  uint32_t IdFromRank(size_t rank);

  // Permanently removes the id from the move-to-front sequence.
  // The implementation has artificially restricted functionality which only
  // allows to add new ids sequentially. So once the id is removed using this
  // function it cannot be reinserted.
  void DeprecateId(uint32_t id) {
    auto it = id_to_node_.find(id);
    assert(it != id_to_node_.end());
    RemoveNode(it->second);
    // The iterator should still be valid, even if RemoveNode has modified
    // id_to_node_. But just in case erase by id, not by iterator.
    id_to_node_.erase(id);
  }

  // Returns the number of elements in the move-to-front sequence.
  size_t GetSize() const {
    return SizeOf(root_);
  }

  // The methods below are for testing only.

  // Inserts the value in the internal tree data structure. For testing only.
  void TestInsert(uint32_t val) {
    InsertNode(CreateNode(val, val));
  }

  // Removes the value from the internal tree data structure. For testing only.
  void TestRemove(uint32_t val) {
    const auto it = id_to_node_.find(val);
    assert (it != id_to_node_.end());
    RemoveNode(it->second);
  }

  // Prints the internal tree data structure to |out|. For testing only.
  void PrintTree(std::ostream& out, bool print_timestamp = false) const {
    if (root_)
      PrintTreeInternal(out, root_, 1, print_timestamp);
  }

 private:
  // Internal tree data structure uses handles instead of pointers. Leaves and
  // root parent reference a singleton under handle 0. Although dereferencing
  // a null pointer is not possible, inappropriate access to handle 0 would
  // cause an assertion. Handles are not garbage collected if id was deprecated
  // with DeprecateId(). But handles are recycled when a node is repositioned.

  // Internal tree data structure node.
  struct Node {
    // Fields are in order from biggest bit width to smallest.
    // SPIR-V id.
    uint32_t id = 0;
    // Timestamp from a logical clock which updates every time the element is
    // accessed through IdFromRank or RankFromId.
    uint32_t timestamp = 0;
    // The size of the node's subtree, including the node.
    // SizeOf(LeftOf(node)) + SizeOf(RightOf(node)) + 1.
    uint32_t size = 0;
    // Handles to connected nodes.
    uint32_t left = 0;
    uint32_t right = 0;
    uint32_t parent = 0;
    // Distance to the farthest leaf.
    // Leaves have height 0, real nodes at least 1.
    uint32_t height = 0;
  };

  // Creates node and sets correct values. Non-NIL nodes should be created only
  // through this function.
  uint32_t CreateNode(uint32_t timestamp, uint32_t id) {
    uint32_t handle = static_cast<uint32_t>(nodes_.size());
    nodes_.emplace_back(Node());
    Node& node = nodes_.back();
    node.timestamp = timestamp;
    node.id = id;
    node.size = 1;
    // Non-NIL nodes start with height 1 because their NIL children are leaves.
    node.height = 1;
    id_to_node_.emplace(id, handle);
    return handle;
  }

  // Node accessor methods. Naming is designed to be similar to natural
  // language as these functions tend to be used in sequences, for example:
  // ParentOf(LeftestDescendentOf(RightOf(node)))

  // Returns id of the node referenced by |handle|.
  uint32_t IdOf(uint32_t node) const {
    return nodes_.at(node).id;
  }

  // Returns left child of |node|.
  uint32_t LeftOf(uint32_t node) const {
    return nodes_.at(node).left;
  }

  // Returns right child of |node|.
  uint32_t RightOf(uint32_t node) const {
    return nodes_.at(node).right;
  }

  // Returns parent of |node|.
  uint32_t ParentOf(uint32_t node) const {
    return nodes_.at(node).parent;
  }

  // Returns timestamp of |node|.
  uint32_t TimestampOf(uint32_t node) const {
    assert(node);
    return nodes_.at(node).timestamp;
  }

  // Returns size of |node|.
  uint32_t SizeOf(uint32_t node) const {
    return nodes_.at(node).size;
  }

  // Returns height of |node|.
  uint32_t HeightOf(uint32_t node) const {
    return nodes_.at(node).height;
  }

  // Returns mutable reference to id of |node|.
  uint32_t& MutableIdOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).id;
  }

  // Returns mutable reference to handle of left child of |node|.
  uint32_t& MutableLeftOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).left;
  }

  // Returns mutable reference to handle of right child of |node|.
  uint32_t& MutableRightOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).right;
  }

  // Returns mutable reference to handle of parent of |node|.
  uint32_t& MutableParentOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).parent;
  }

  // Returns mutable reference to timestamp of |node|.
  uint32_t& MutableTimestampOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).timestamp;
  }

  // Returns mutable reference to size of |node|.
  uint32_t& MutableSizeOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).size;
  }

  // Returns mutable reference to height of |node|.
  uint32_t& MutableHeightOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).height;
  }

  // Returns true iff |node| is left child of its parent.
  bool IsLeftChild(uint32_t node) const {
    assert(node);
    return LeftOf(ParentOf(node)) == node;
  }

  // Returns true iff |node| is right child of its parent.
  bool IsRightChild(uint32_t node) const {
    assert(node);
    return RightOf(ParentOf(node)) == node;
  }

  // Returns true iff |node| has no relatives.
  bool IsOrphan(uint32_t node) const {
    assert(node);
    return !ParentOf(node) && !LeftOf(node) && !RightOf(node);
  }

  // Returns the height difference between right and left subtrees.
  int BalanceOf(uint32_t node) const {
    return int(HeightOf(RightOf(node))) - int(HeightOf(LeftOf(node)));
  }

  // Updates size and height of the node, assuming that the children have
  // correct values.
  void UpdateNode(uint32_t node);

  // Returns the most LeftOf(LeftOf(... descendent which is not leaf.
  uint32_t LeftestDescendantOf(uint32_t node) const {
    uint32_t parent = 0;
    while (node) {
      parent = node;
      node = LeftOf(node);
    }
    return parent;
  }

  // Returns the most RightOf(RightOf(... descendent which is not leaf.
  uint32_t RightestDescendantOf(uint32_t node) const {
    uint32_t parent = 0;
    while (node) {
      parent = node;
      node = RightOf(node);
    }
    return parent;
  }

  // Prints the internal tree data structure for debug purposes in the following
  // format:
  // 10H3S4----5H1S1-----D2
  //           15H2S2----12H1S1----D3
  // Right links are horizontal, left links step down one line.
  // 5H1S1 is read as id 5, height 1, size 1. Optionally node label can also
  // contain timestamp (5H1S1T15). D3 stands for depth 3.
  void PrintTreeInternal(std::ostream& out, uint32_t node, size_t depth,
                         bool print_timestamp) const;

  // Inserts node in the tree. The node must be an orphan.
  void InsertNode(uint32_t node);

  // Removes node from the tree. May change id_to_node_ if removal uses a
  // scapegoat. Returns the removed (orphaned) handle for recycling. The
  // returned handle may not be equal to |node| if scapegoat was used.
  uint32_t RemoveNode(uint32_t node);

  // Rotates |node| left, reassigns all connections and returns the node
  // which takes place of the |node|.
  uint32_t RotateLeft(const uint32_t node);

  // Rotates |node| right, reassigns all connections and returns the node
  // which takes place of the |node|.
  uint32_t RotateRight(const uint32_t node);

  // Root node handle. The tree is empty if root_ is 0.
  uint32_t root_ = 0;

  // Incremented counters for next timestamp and id.
  uint32_t next_timestamp_ = 1;
  uint32_t next_id_ = 1;

  // Holds all tree nodes. Indices of this vector are node handles.
  std::vector<Node> nodes_;

  // Maps ids to node handles.
  std::unordered_map<uint32_t, uint32_t> id_to_node_;
};

}  // namespace spvutils

#endif  // LIBSPIRV_UTIL_MOVE_TO_FRONT_H_
