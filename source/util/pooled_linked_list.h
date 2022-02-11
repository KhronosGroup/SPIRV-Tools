// Copyright (c) 2021 The Khronos Group Inc.
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

#ifndef SOURCE_UTIL_POOLED_LINKED_LIST_H_
#define SOURCE_UTIL_POOLED_LINKED_LIST_H_

#include <cstdint>
#include <vector>

namespace spvtools {
namespace utils {

// Implements a linked-list where list nodes come from a shared pool. This is
// meant to be used in scenarios where it is desirable to avoid many small
// allocations.
//
// Instead of pointers, the list uses indices to allow the underlying storage 
// to be modified without needing to modify the list. When removing elements 
// from the list, nodes are not deleted or recycled: to reclaim unused space,
// perform a sequence of |move_nodes| operations into a temporary pool, which
// then is moved into the old pool.
//
// This does *not* attempt to implement a full stl-compatible interface.
template <typename T>
class PooledLinkedList {
 public:
  struct Node {
    T element = {};
    int32_t next = -1;
  };

  using NodePool = std::vector<Node>;

  PooledLinkedList() = delete;
  PooledLinkedList(NodePool& nodes) : nodes_(nodes) {}

  // Shared iterator implementation (for iterator and const_iterator).
  template <typename ElementT, typename PoolT>
  class iterator_base {
   public:
    iterator_base(const iterator_base& i) : nodes_(i.nodes_), index_(i.index_) {}

    iterator_base& operator++() {
      index_ = nodes_->at(index_).next;
      return *this;
    }

    iterator_base& operator=(const iterator_base& i) {
      nodes_ = i.nodes_;
      index_ = i.index_;
      return *this;
    }

    ElementT& operator*() const { return nodes_->at(index_).element; }
    ElementT* operator->() const { return &nodes_->at(index_).element; }

    friend inline bool operator==(const iterator_base& lhs,
                                  const iterator_base& rhs) {
      return lhs.nodes_ == rhs.nodes_ && lhs.index_ == rhs.index_;
    }
    friend inline bool operator!=(const iterator_base& lhs,
                                  const iterator_base& rhs) {
      return lhs.nodes_ != rhs.nodes_ || lhs.index_ != rhs.index_;
    }

    // Define standard iterator types needs so this class can be
    // used with <algorithms>.
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = ElementT;
    using pointer = ElementT*;
    using const_pointer = const ElementT*;
    using reference = ElementT&;
    using const_reference = const ElementT&;
    using size_type = size_t;

   private:
    friend PooledLinkedList;

    iterator_base(PoolT* pool, int32_t index)
        : nodes_(pool), index_(index) {}

    PoolT* nodes_;
    int32_t index_ = -1;
  };

  using iterator = iterator_base<T, NodePool>;
  using const_iterator = iterator_base<const T, const NodePool>;

  bool empty() const { return head_ == -1; }
  size_t size() const { return size_; }

  T& front() { return nodes_[head_].element; }
  T& back() { return nodes_[tail_].element; }
  const T& front() const { return nodes_[head_].element; }
  const T& back() const { return nodes_[tail_].element; }

  iterator begin() { return iterator(&nodes_, head_); }
  iterator end() { return iterator(&nodes_, -1); }
  const_iterator begin() const { return const_iterator(&nodes_, head_); }
  const_iterator end() const { return const_iterator(&nodes_, -1); }

  // Inserts |element| at the back of the list.
  void push_back(T element) {
    int32_t new_tail = int32_t(nodes_.size());
    nodes_.push_back(Node{element, -1});
    if (head_ == -1) {
      head_ = new_tail;
      tail_ = new_tail;
    } else {
      nodes_[tail_].next = new_tail;
      tail_ = new_tail;
    }
    ++size_;
  }

  // Removes the first occurrence of |element| from the list.
  // Returns if |element| was removed.
  bool remove_first(T element) {
    int32_t* prev_next = &head_;
    for (int32_t prev_index = -1, index = head_; index != -1; /**/) {
      auto& node = nodes_[index];
      if (node.element == element) {
        // Snip from of the list, optionally fixing up tail pointer.
        if (tail_ == index) {
          assert(node.next == -1);
          tail_ = prev_index;
        }
        *prev_next = node.next;
        --size_;
        return true;
      } else {
        prev_next = &node.next;
      }
      prev_index = index;
      index = node.next;
    }
    return false;
  }

  // Moves the nodes in this list into |new_pool|, providing a way to compact
  // storage and reclaim unused space.
  //
  // Upon completing a sequence of |move_nodes| calls, you must swap storage
  // from |new_pool| into the pool used by your PooledLinkedLists.
  // Example usage:
  //
  //    NodePool old_pool;  // Existing lists use this pool
  //    NodePool new_pool;  // Temporary storage
  //    for (PooledLinkedList& list : lists) {
  //        list.move_to(new_pool);
  //    }
  //    old_pool = std::move(new_pool);
  void move_nodes(NodePool& new_pool) {
    // Be sure to construct the list in the same order, instead of simply
    // doing a sequence of push_backs.
    int32_t prev_entry = -1;
    for (int32_t index = head_; index != -1; index = nodes_[index].next) {
      int32_t this_entry = int32_t(new_pool.size());
      new_pool.push_back(Node{std::move(nodes_[index].element), -1});
      if (prev_entry == -1) {
        head_ = this_entry;
      } else {
        new_pool[prev_entry].next = this_entry;
      }
      prev_entry = this_entry;
    }
    tail_ = prev_entry;
  }

 private:
  NodePool& nodes_;
  int32_t head_ = -1;
  int32_t tail_ = -1;
  uint32_t size_ = 0;
};

template <typename T>
using PooledLinkedListNodes = typename PooledLinkedList<T>::NodePool;

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_POOLED_LINKED_LIST_H_
