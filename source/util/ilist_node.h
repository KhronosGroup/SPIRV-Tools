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

#ifndef LIBSPIRV_OPT_ILIST_NODE_H_
#define LIBSPIRV_OPT_ILIST_NODE_H_

#include <cassert>

namespace spvtools {
namespace utils {

template <class NodeType>
class IntrusiveList;

// IntrusiveNodeBase is the base class for nodes in an IntrusiveList.
// See the comments in ilist.h on how to use the class.

template <class NodeType>
class IntrusiveNodeBase {
 public:
  // Creates a new node that is not in a list.
  inline IntrusiveNodeBase();

  // Returns the node that comes after the given node in the list, if one
  // exists.  If the given node is not in a list or is at the end of the list,
  // the return value is nullptr.
  inline NodeType* NextNode() const;

  // Returns the node that comes before the given node in the list, if one
  // exists.  If the given node is not in a list or is at the start of the
  // list, the return value is nullptr.
  inline NodeType* PreviousNode() const;

  // Inserts the given node immediately before |pos| in the list.
  // If the given node is already in a list, it will first be removed
  // from that list.
  //
  // It is assumed that the given node is of type NodeType.  It is an error if
  // |pos| is not already in a list.
  inline void InsertBefore(NodeType* pos);

  // Inserts the given node immediately after |pos| in the list.
  // If the given node is already in a list, it will first be removed
  // from that list.
  //
  // It is assumed that the given node is of type NodeType.  It is an error if
  // |pos| is not already in a list.
  inline void InsertAfter(NodeType* pos);

  // Removes the given node from the list.  It is assumed that the node is
  // in a list.  Note that this does not free any storage related to the node.
  inline void RemoveFromList();

 private:
  // The pointers to the next and previous nodes in the list.
  // If the current node is not part of a list, then |next_node_| and
  // |previous_node_| are equal to |nullptr|.
  NodeType* next_node_;
  NodeType* previous_node_;

  // Only true for the sentinel node stored in the list itself.
  bool is_sentinel_;

  friend IntrusiveList<NodeType>;
};

// Implementation of IntrusiveNodeBase

template <class NodeType>
inline IntrusiveNodeBase<NodeType>::IntrusiveNodeBase()
    : next_node_(nullptr), previous_node_(nullptr), is_sentinel_(false) {}

template <class NodeType>
inline NodeType* IntrusiveNodeBase<NodeType>::NextNode() const {
  if (!next_node_->is_sentinel_) return next_node_;
  return nullptr;
}

template <class NodeType>
inline NodeType* IntrusiveNodeBase<NodeType>::PreviousNode() const {
  if (!previous_node_->is_sentinel_) return previous_node_;
  return nullptr;
}

template <class NodeType>
inline void IntrusiveNodeBase<NodeType>::InsertBefore(NodeType* pos) {
  assert(!this->is_sentinel_ && "Sentinel nodes cannot be moved around.");
  assert(pos->previous_node_ != nullptr && "Pos should already be in a list.");
  if (this->previous_node_ != nullptr) this->RemoveFromList();

  this->next_node_ = pos;
  this->previous_node_ = pos->previous_node_;
  pos->previous_node_ = static_cast<NodeType*>(this);
  this->previous_node_->next_node_ = static_cast<NodeType*>(this);
}

template <class NodeType>
inline void IntrusiveNodeBase<NodeType>::InsertAfter(NodeType* pos) {
  assert(!this->is_sentinel_ && "Sentinel nodes cannot be moved around.");
  assert(pos->previous_node_ != nullptr && "Pos should already be in a list.");
  if (this->previous_node_ != nullptr) this->RemoveFromList();

  this->previous_node_ = pos;
  this->next_node_ = pos->next_node_;
  pos->next_node_ = static_cast<NodeType*>(this);
  this->next_node_->previous_node_ = static_cast<NodeType*>(this);
}

template <class NodeType>
inline void IntrusiveNodeBase<NodeType>::RemoveFromList() {
  assert(!this->is_sentinel_ && "Sentinel nodes cannot be moved around.");
  assert(this->next_node_ != nullptr && "Cannot remove a node from a list if it is not in a list.");
  assert(this->previous_node_ != nullptr && "Cannot remove a node from a list if it is not in a list.");

  this->next_node_->previous_node_ = this->previous_node_;
  this->previous_node_->next_node_ = this->next_node_;
  this->next_node_ = nullptr;
  this->previous_node_ = nullptr;
}

}  // namespace utils
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_ILIST_NODE_H_
