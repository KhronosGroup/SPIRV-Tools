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

#ifndef LIBSPIRV_OPT_ILIST_H_
#define LIBSPIRV_OPT_ILIST_H_

#include <cassert>
#include <type_traits>

#include "ilist_node.h"

namespace spvtools {
namespace utils {

// An IntrusiveList is a generic implementation of a doubly-linked list.  The
// intended convention for using this container is:
//
//      class Node : public IntrusiveNodeBase<Node> {
//        // Node that "Node", the class being defined is the template.
//        // Must have a default constructor accessible to List.
//        // Add whatever data is needed in the node
//      };
//
//      typedef IntrusiveList<Node> List;
//
// You can also inherit from IntrusiveList instead of a typedef if you want to
// add more functionality.
//
// The condition on the template for IntrusiveNodeBase is there to add some type
// checking to the container.  The compiler will still allow inserting elements
// of type IntrusiveNodeBase<Node>, but that would be an error. This assumption
// allows NextNode and PreviousNode to return pointers to Node, and casting will
// not be required by the user.

template <class NodeType>
class IntrusiveList {
 public:
  static_assert(
      std::is_base_of<IntrusiveNodeBase<NodeType>, NodeType>::value,
      "The type from the node must be derived from IntrusiveNodeBase, with "
      "itself in the template.");

  // Creates an empty list.
  inline IntrusiveList();

  // Moves the contents of the given list to the list being constructed.
  IntrusiveList(IntrusiveList&&);

  // Destorys the list.  Note that the elements of the list will not be deleted,
  // but thy will be removed from the list.
  ~IntrusiveList();

  // Moves all of the elements in the list on the RHS to the list on the LHS.
  IntrusiveList& operator=(IntrusiveList&&);

  // Basetype for iterators so an IntrusiveList can be traversed like STL
  // containers.
  template <class T>
  class iterator_template {
   public:
    inline iterator_template(const iterator_template& i) : node_(i.node_) {}
    inline iterator_template& operator++() {
      node_ = node_->NextNode();
      return *this;
    }
    inline iterator_template& operator--() {
      node_ = node_->PreviousNode();
      return *this;
    }
    inline iterator_template& operator=(const iterator_template& i) {
      node_ = i.node_;
      return *this;
    }
    inline T& operator*() const { return *node_; }
    inline T* operator->() const { return node_; }

    friend inline bool operator==(const iterator_template& lhs,
                                  const iterator_template& rhs) {
      return lhs.node_ == rhs.node_;
    }
    friend inline bool operator!=(const iterator_template& lhs,
                                  const iterator_template& rhs) {
      return !(lhs == rhs);
    }

   private:
    inline iterator_template(T* node) { node_ = node; }
    T* node_;

    friend IntrusiveList;
  };

  typedef iterator_template<NodeType> iterator;
  typedef iterator_template<const NodeType> const_iterator;

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;

  // Appends |node| to the end of the list.
  void push_back(NodeType* node);

 private:
  // Doing a deep copy of the list does not make sense if the list does not own
  // the data.  It is not clear who will own the newly created data.  Making
  // copies illegal for that reason.
  IntrusiveList(const IntrusiveList&);
  IntrusiveList& operator=(const IntrusiveList&);

  // A special node used to represent both the start and end of the list,
  // without being part of the list.
  NodeType sentinel_;
};

// Implementation of IntrusiveList

template <class NodeType>
inline IntrusiveList<NodeType>::IntrusiveList() : sentinel_() {
  sentinel_.next_node_ = &sentinel_;
  sentinel_.previous_node_ = &sentinel_;
  sentinel_.is_sentinel_ = true;
}

template <class NodeType>
IntrusiveList<NodeType>::IntrusiveList(IntrusiveList&& list) {
  this->sentinel_ = list.sentinel_;
  this->sentinel_.next_node_->previous_node_ = &sentinel_;
  this->sentinel_.previous_node_->next_node_ = &sentinel_;

  list.sentinel_.next_node_ = &list.sentinel_;
  list.sentinel_.previous_node_ = &list.sentinel_;
}

template <class NodeType>
IntrusiveList<NodeType>::~IntrusiveList() {
  for (auto i : *this) i.RemoveFromList();
}

template <class NodeType>
IntrusiveList<NodeType>& IntrusiveList<NodeType>::operator=(
    IntrusiveList<NodeType>&& list) {
  this->sentinel_ = list.sentinel_;
  this->sentinel_.next_node_->previous_node_ = &sentinel_;
  this->sentinel_.previous_node_->next_node_ = &sentinel_;

  list.sentinel_.next_node_ = &list.sentinel_;
  list.sentinel_.previous_node_ = &list.sentinel_;
  return *this;
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::iterator
IntrusiveList<NodeType>::begin() {
  return iterator(sentinel_.NextNode());
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::iterator
IntrusiveList<NodeType>::end() {
  return iterator(nullptr);
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::const_iterator
IntrusiveList<NodeType>::begin() const {
  return const_iterator(sentinel_.NextNode());
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::const_iterator
IntrusiveList<NodeType>::end() const {
  return const_iterator(nullptr);
}
template <class NodeType>
void IntrusiveList<NodeType>::push_back(NodeType* node) {
  node->InsertBefore(&sentinel_);
}

}  // namespace utils
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_ILIST_H_
