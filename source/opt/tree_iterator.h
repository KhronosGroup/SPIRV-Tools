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

#ifndef LIBSPIRV_OPT_TREE_ITERATOR_H_
#define LIBSPIRV_OPT_TREE_ITERATOR_H_

#include <stack>
#include <type_traits>
#include <utility>

namespace spvtools {
namespace opt {

// Helper class to iterate over a tree in a depth first order.
// The class assumes the data structure is a tree, tree node type implements a
// forward iterator.
// At each step, the iterator holds the pointer to the current node and state of
// the walk.
// The state is recorded by stacking the iteration position of the node
// children. To move to the next node, the iterator:
//  - Looks at the top of the stack;
//  - Sets the node behind the iterator as the current node;
//  - Increments the iterator if it has more children to visit, pops otherwise;
//  - If the current node has children, the children iterator is pushed into the
//    stack.
template <typename NodeTy>
class TreeDFIterator {
  static_assert(!std::is_pointer<NodeTy>::value &&
                    !std::is_reference<NodeTy>::value,
                "NodeTy should be a class");
  // Type alias to keep track of the const qualifier.
  using NodeIterator =
      typename std::conditional<std::is_const<NodeTy>::value,
                                typename NodeTy::const_iterator,
                                typename NodeTy::iterator>::type;

  // Type alias to keep track of the const qualifier.
  using NodePtr = NodeTy*;

 public:
  // Standard iterator interface.
  using reference = NodeTy&;
  using value_type = NodeTy;

  explicit inline TreeDFIterator(NodePtr top_node) : current_(top_node) {
    if (current_ && current_->begin() != current_->end())
      parent_iterators_.emplace(make_pair(current_, current_->begin()));
  }

  // end() iterator.
  inline TreeDFIterator() : TreeDFIterator(nullptr) {}

  bool operator==(const TreeDFIterator& x) const {
    return current_ == x.current_;
  }

  bool operator!=(const TreeDFIterator& x) const { return !(*this == x); }

  reference operator*() const { return *current_; }

  NodePtr operator->() const { return current_; }

  TreeDFIterator& operator++() {
    MoveToNextNode();
    return *this;
  }

  TreeDFIterator operator++(int) {
    TreeDFIterator tmp = *this;
    ++*this;
    return tmp;
  }

 private:
  // Moves the iterator to the next node in the tree.
  // If we are at the end, do nothing, otherwise
  // if our current node has children, use the children iterator and push the
  // current node into the stack.
  // If we reach the end of the local iterator, pop it.
  inline void MoveToNextNode() {
    if (!current_) return;
    if (parent_iterators_.empty()) {
      current_ = nullptr;
      return;
    }
    std::pair<NodePtr, NodeIterator>& next_it = parent_iterators_.top();
    // Set the new node.
    current_ = *next_it.second;
    // Update the iterator for the next child.
    ++next_it.second;
    // If we finished with node, pop it.
    if (next_it.first->end() == next_it.second) parent_iterators_.pop();
    // If our current node is not a leaf, store the iteration state for later.
    if (current_->begin() != current_->end())
      parent_iterators_.emplace(make_pair(current_, current_->begin()));
  }

  // The current node of the tree.
  NodePtr current_;
  // State of the tree walk: each pair contains the parent node (which has been
  // already visited) and the iterator of the next children to visit.
  // When all the children has been visited, we pop the entry, get the next
  // child and push back the pair if the children iterator is not end().
  std::stack<std::pair<NodePtr, NodeIterator>> parent_iterators_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_TREE_ITERATOR_H_
