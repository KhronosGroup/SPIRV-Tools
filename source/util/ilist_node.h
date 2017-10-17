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
  inline IntrusiveNodeBase(const IntrusiveNodeBase&);
  inline IntrusiveNodeBase& operator=(const IntrusiveNodeBase&);
  inline IntrusiveNodeBase(IntrusiveNodeBase&& that);

  // Will destroy a node.  It is an error to destroy a node that is part of a
  // list, unless it is an sentinel.
  virtual ~IntrusiveNodeBase();

  IntrusiveNodeBase& operator=(IntrusiveNodeBase&& that);

  // Returns true if |this| is in a list.
  inline bool IsInAList() const;

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
  // in a list.  Note that this does not free any storage related to the node,
  // it becomes the caller's responsibility to free the storage.
  inline void RemoveFromList();

 protected:
  // This function will replace |this| with |target|.  No nodes that are not
  // sentinels, |target| takes the place of |this|.  If the nodes are sentinels,
  // then it will cause all of the nodes to be move from one list to another.
  void ReplaceWith(NodeType* target);

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

template<class NodeType>
inline IntrusiveNodeBase<NodeType>::IntrusiveNodeBase(
    const IntrusiveNodeBase&) {
  next_node_ = nullptr;
  previous_node_ = nullptr;
  is_sentinel_ = false;
}

template<class NodeType>
inline IntrusiveNodeBase<NodeType>& IntrusiveNodeBase<NodeType>::operator=(
    const IntrusiveNodeBase&) {
  assert(!is_sentinel_);
  if (IsInAList()) RemoveFromList();
  return *this;
}

template<class NodeType>
inline IntrusiveNodeBase<NodeType>::IntrusiveNodeBase(IntrusiveNodeBase&& that)
    : next_node_(nullptr),
      previous_node_(nullptr),
      is_sentinel_(that.is_sentinel_) {
  if (is_sentinel_) {
    next_node_ = this;
    previous_node_ = this;
  }
  that.ReplaceWith(this);
}

template<class NodeType>
IntrusiveNodeBase<NodeType>::~IntrusiveNodeBase() {
  assert(is_sentinel_ || !IsInAList());
}

template<class NodeType>
IntrusiveNodeBase<NodeType>& IntrusiveNodeBase<NodeType>::operator=(
    IntrusiveNodeBase&& that) {
  that.ReplaceWith(this);
  return *this;
}

template<class NodeType>
inline bool IntrusiveNodeBase<NodeType>::IsInAList() const {
  return next_node_ != nullptr;
}

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
  assert(pos->IsInAList() && "Pos should already be in a list.");
  if (this->IsInAList()) this->RemoveFromList();

  this->next_node_ = pos;
  this->previous_node_ = pos->previous_node_;
  pos->previous_node_ = static_cast<NodeType*>(this);
  this->previous_node_->next_node_ = static_cast<NodeType*>(this);
}

template <class NodeType>
inline void IntrusiveNodeBase<NodeType>::InsertAfter(NodeType* pos) {
  assert(!this->is_sentinel_ && "Sentinel nodes cannot be moved around.");
  assert(pos->IsInAList() && "Pos should already be in a list.");
  if (this->IsInAList()) {
    this->RemoveFromList();
  }

  this->previous_node_ = pos;
  this->next_node_ = pos->next_node_;
  pos->next_node_ = static_cast<NodeType*>(this);
  this->next_node_->previous_node_ = static_cast<NodeType*>(this);
}

template <class NodeType>
inline void IntrusiveNodeBase<NodeType>::RemoveFromList() {
  assert(!this->is_sentinel_ && "Sentinel nodes cannot be moved around.");
  assert(this->IsInAList() &&
      "Cannot remove a node from a list if it is not in a list.");

  this->next_node_->previous_node_ = this->previous_node_;
  this->previous_node_->next_node_ = this->next_node_;
  this->next_node_ = nullptr;
  this->previous_node_ = nullptr;
}

template<class NodeType>
void IntrusiveNodeBase<NodeType>::ReplaceWith(NodeType* target) {
  if (is_sentinel_) {
    assert(target->next_node_ == target);
    assert(this->is_sentinel_);
  } else {
    assert(IsInAList() && "Sentinel nodes must always be part of a list.");
    assert(!this->is_sentinel_ &&
        "Cannot turn a sentinel node into one that is not.");
  }

  if (this->next_node_ != this) {
    target->next_node_ = this->next_node_;
    target->previous_node_ = this->previous_node_;

    target->next_node_->previous_node_ = static_cast<NodeType*>(target);
    target->previous_node_->next_node_ = static_cast<NodeType*>(target);

    if (!this->is_sentinel_) {
      this->next_node_ = nullptr;
      this->previous_node_ = nullptr;
    } else {
      this->next_node_ = static_cast<NodeType*>(this);
      this->previous_node_ = static_cast<NodeType*>(this);
    }
  } else {
    target->next_node_ = static_cast<NodeType*>(target);
    target->previous_node_ = static_cast<NodeType*>(target);
  }
}

}  // namespace utils
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_ILIST_NODE_H_
