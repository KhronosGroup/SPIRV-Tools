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
// IdFromRank - access id by its 0-indexed rank in the move-to-front sequence.
// RankFromId - get the rank of the given id in the move-to-front sequence.
// Accessing an id with any of the two functions moves the id to the front
// of the sequence (rank of 0).
//
// The implementation is based on an AVL-based order statistic tree.
//
// Terminology
// id: SPIR-V id.
// rank: 0-indexed value showing how recently the id was accessed.
// node: uint16_t handle used internally to access nodes.
// size: size of the subtree of a node (including the node).
// height: distance from a node to the farthest leaf.
// index: not used to avoid confusion with rank and handle.
class MoveToFront {
  struct Node;
 public:
  explicit MoveToFront(size_t reserve_capacity = 128) {
    nodes_.reserve(reserve_capacity);

    // Create NIL node.
    nodes_.emplace_back(Node());
  }

  // Returns the 0-indexed rank of id in the move-to-front sequence and moves
  // id to the front. Example:
  // Before the call: 4 8 2 1 7
  // RankFromId(8) returns 1
  // After the call: 8 4 2 1 7
  size_t RankFromId(uint32_t id);

  // Returns id corresponding to a 0-indexed rank in the move-to-front sequence
  // and moves id to the front. Example:
  // Before the call: 4 8 2 1 7
  // IdFromRank(1) returns 8
  // After the call: 8 4 2 1 7
  uint32_t IdFromRank(size_t rank);

  // Permanently removes the id from the move-to-front sequence.
  void DeprecateId(uint32_t id) {
    auto it = id_to_node_.find(id);
    assert(it != id_to_node_.end());
    RemoveNode(it->second);
    // The iterator should still be valid, even if RemoveNode has modified
    // id_to_node_. But just in case erase by id, not by iterator.
    id_to_node_.erase(id);

#ifndef NDEBUG
    deprecated_ids_.insert(id);
#endif
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
  // Internal tree data structure node.
  struct Node {
    // Fields are in order from biggest bit width to smallest.
    // SPIR-V id.
    uint32_t id = 0;
    // Timestamp from a logical clock which updates every time the element is
    // accessed.
    uint32_t timestamp = 0;
    // The size of the node's subtee, including the node.
    // SizeOf(LeftOf(node)) + SizeOf(RightOf(node)) + 1.
    uint16_t size = 0;
    // Handles to connected nodes.
    uint16_t left = 0;
    uint16_t right = 0;
    uint16_t parent = 0;
    // Distance to the farthest leaf.
    // Leaves have height 0, real nodes at least 1.
    uint8_t height = 0;
  };

  // Creates node and sets correct values. Nodes should be created only through
  // this function.
  uint16_t CreateNode(uint32_t timestamp, uint32_t id) {
    nodes_.emplace_back(Node());
    Node& node = nodes_.back();
    node.timestamp = timestamp;
    node.id = id;
    node.size = 1;
    node.height = 1;
    uint16_t handle = static_cast<uint16_t>(nodes_.size() - 1);
    id_to_node_.emplace(id, handle);
    return handle;
  }

  // Internal tree data structure uses handles instead of pointers. Leaves and
  // root parent reference a singleton under handle 0. Although dereferencing
  // a null pointer is not possible, inappropriate access to handle 0 would
  // cause an assertion. Handles are not garbage collected if id is deprecated.
  // But handles are recycled when a node is repositioned.

  // Node accessor methods. Naming is designed to be similar to natural
  // language as these functions tend to be used in sequences, for example:
  // ParentOf(LeftestDescendentOf(RightOf(node)))

  // Returns immutable id of the node referenced by |handle|.
  inline uint32_t IdOf(uint16_t handle) const {
    return nodes_.at(handle).id;
  }

  // Returns immutable handle to left of the node referenced by |handle|.
  inline uint16_t LeftOf(uint16_t handle) const {
    return nodes_.at(handle).left;
  }

  // Returns immutable handle to right of the node referenced by |handle|.
  inline uint16_t RightOf(uint16_t handle) const {
    return nodes_.at(handle).right;
  }

  // Returns immutable handle to parent of the node referenced by |handle|.
  inline uint16_t ParentOf(uint16_t handle) const {
    return nodes_.at(handle).parent;
  }

  // Returns immutable timestamp of the node referenced by |handle|.
  inline uint32_t TimestampOf(uint16_t handle) const {
    assert(handle);
    return nodes_.at(handle).timestamp;
  }

  // Returns immutable size of the node referenced by |handle|.
  inline uint16_t SizeOf(uint16_t handle) const {
    return nodes_.at(handle).size;
  }

  // Returns immutable height of the node referenced by |handle|.
  inline uint16_t HeightOf(uint16_t handle) const {
    return nodes_.at(handle).height;
  }

  // Returns mutable id of the node referenced by |handle|.
  inline uint32_t& MutableIdOf(uint16_t handle) {
    assert(handle);
    return nodes_.at(handle).id;
  }

  // Returns mutable handle to left of the node referenced by |handle|.
  inline uint16_t& MutableLeftOf(uint16_t handle) {
    assert(handle);
    return nodes_.at(handle).left;
  }

  // Returns mutable handle to right of the node referenced by |handle|.
  inline uint16_t& MutableRightOf(uint16_t handle) {
    assert(handle);
    return nodes_.at(handle).right;
  }

  // Returns mutable handle to parent of the node referenced by |handle|.
  inline uint16_t& MutableParentOf(uint16_t handle) {
    assert(handle);
    return nodes_.at(handle).parent;
  }

  // Returns mutable timestamp of the node referenced by |handle|.
  inline uint32_t& MutableTimestampOf(uint16_t handle) {
    assert(handle);
    return nodes_.at(handle).timestamp;
  }

  // Returns mutable size of the node referenced by |handle|.
  inline uint16_t& MutableSizeOf(uint16_t handle) {
    assert(handle);
    return nodes_.at(handle).size;
  }

  // Returns mutable height of the node referenced by |handle|.
  inline uint8_t& MutableHeightOf(uint16_t handle) {
    assert(handle);
    return nodes_.at(handle).height;
  }

  // Returns true iff |handle| is left child of its parent.
  inline bool IsLeftChild(uint16_t handle) const {
    assert(handle);
    return LeftOf(ParentOf(handle)) == handle;
  }

  // Returns true iff |handle| is right child of its parent.
  inline bool IsRightChild(uint16_t handle) const {
    assert(handle);
    return RightOf(ParentOf(handle)) == handle;
  }

  // Returns the sibling of handle or 0 if no sibling.
  inline uint16_t SiblingOf(uint16_t handle) const {
    assert(handle);
    const uint16_t parent = ParentOf(handle);
    return LeftOf(parent) == handle ? RightOf(parent) : LeftOf(parent);
  }

  // Returns the height difference between right and left subtrees.
  inline int BalanceOf(uint16_t handle) const {
    return int(HeightOf(RightOf(handle))) - int(HeightOf(LeftOf(handle)));
  }

  // Updates size and height of the node, assuming that the children have
  // correct values.
  void UpdateNode(uint16_t handle);

  // Returns the most LeftOf(LeftOf(... descendent which is not leaf.
  uint16_t LeftestDescendantOf(uint16_t handle) const {
    uint16_t parent = 0;
    while (handle) {
      parent = handle;
      handle = LeftOf(handle);
    }
    return parent;
  }

  // Returns the most RightOf(RightOf(... descendent which is not leaf.
  uint16_t RightestDescendantOf(uint16_t handle) const {
    uint16_t parent = 0;
    while (handle) {
      parent = handle;
      handle = RightOf(handle);
    }
    return parent;
  }

  // Prints the internal tree data structure for debug purposes in the following
  // format:
  // 10H3S4----5H1S1-----D2
  //           15H2S2----12H1S1----D3
  // Right links are horisontal, left links step down one line.
  // 5H1S1 is read as id 5, height 1, size 1. Optionally node label can also
  // contain timestamp (5H1S1T15). D3 stands for depth 3.
  void PrintTreeInternal(std::ostream& out, uint16_t node, size_t depth,
                         bool print_timestamp) const;

  // Inserts node handle in the tree.
  void InsertNode(uint16_t node);

  // Removes node from the tree. May change id_to_node_ if removal uses a
  // scapegoat. Returns the removed (orphaned) handle for recycling. The
  // returned handle may not be equal to |node| if scapegoat was used.
  uint16_t RemoveNode(uint16_t node);

  // Rotates |node| left, reasigns all connections and returns the node
  // which takes place of the |node|.
  uint16_t RotateLeft(const uint16_t node);

  // Rotates |node| right, reasigns all connections and returns the node
  // which takes place of the |node|.
  uint16_t RotateRight(const uint16_t node);

  uint16_t root_ = 0;
  uint32_t next_timestamp_ = 1;

  // Holds all tree nodes. Indices of this vector are node handles.
  std::vector<Node> nodes_;

  // Maps ids to node handles.
  std::unordered_map<uint32_t, uint16_t> id_to_node_;
#ifndef NDEBUG
  // Set of deprecated nodes, is used for debug only.
  std::unordered_set<uint32_t> deprecated_ids_;
#endif
};

}  // namespace spvutils

#endif  // LIBSPIRV_UTIL_MOVE_TO_FRONT_H_
