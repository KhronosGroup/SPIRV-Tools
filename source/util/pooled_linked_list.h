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

namespace spvtools {
namespace utils {

// Implements a linked-list where list nodes come from a shared pool that is
// allocated in bulk.  This is meant to be used in scenarios where you have
// many short lists and want to avoid making a great many small allocations.
// 
// Instead of pointers, the list uses indices to allow the underlying storage 
// to be modified without needing to modify the list. When removing elements 
// from the list, nodes are not deleted or recycled: to reclaim unused space,
// perform a sequence of |move_to| operations into a new pool on all the 
// lists stored in the old pool.
//
// This does *not* attempt to implement a full stl-compatible interface.
template <typename T>
class PooledLinkedList {
 public:
  struct Head {
    int32_t head = -1;
    int32_t tail = -1;
  };

  struct Node {
    T element = {};
    int32_t next = -1;
  };

  PooledLinkedList() = default;
  ~PooledLinkedList() = default;

  PooledLinkedList(PooledLinkedList&& that) { *this = std::move(that); }

  PooledLinkedList(const PooledLinkedList&) = delete;
  PooledLinkedList& operator=(const PooledLinkedList&) = delete;

  Node& operator[](int32_t index) { return nodes_[index]; }
  const Node& operator[](int32_t index) const { return nodes_[index]; }

  bool empty(const Head& head) const { return head.head == -1; }

  // Inserts |element| at the back of the list, updating |list_head|
  void push_back(Head& list_head, T element) {
    int32_t new_tail = int32_t(nodes_.size());
    nodes_.push_back({element, -1});
    if (list_head.head == -1) {
      list_head.head = new_tail;
      list_head.tail = new_tail;
    } else {
      nodes_[list_head.tail].next = new_tail;
      list_head.tail = new_tail;
    }
  }

  // Removes the first occurrence of |element| from the list, updating |list_head|.
  // Returns if |element| was removed.
  bool remove_first(Head& list_head, T element) {
    int32_t* prev_next = &list_head.head;
    for (int32_t prev_index = -1, index = list_head.head; index != -1; /**/) {
      auto& node = nodes_[index];
      if (node.element == element) {
        // Snip from of the list, optionally fixing up tail pointer.
        if (list_head.tail == index) {
          assert(node.next == -1);
          list_head.tail = prev_index;
        }
        *prev_next = node.next;
        return true;
      } else {
        prev_next = &node.next;
      }
      prev_index = index;
      index = node.next;
    }
    return false;
  }

  // Moves the elements in the provided list into this pool.
  // Provides a way to compact the pool, reclaiming unused storage.
  void move_to(Head& list_head, PooledLinkedList& that) {
    // Be sure to construct the list in the same order, instead of simply
    // doing a sequence of push_backs.
    int32_t prev_entry = -1;
    for (int32_t index = list_head.head; index != -1;
         index = nodes_[index].next) {
      int32_t this_entry = int32_t(that.nodes_.size());
      that.nodes_.push_back({std::move(nodes_[index].element), -1});
      if (prev_entry == -1) {
        list_head.head = this_entry;
      } else {
        that.nodes_[prev_entry].next = this_entry;
      }
      prev_entry = this_entry;
    }
    list_head.tail = prev_entry;
  }

  PooledLinkedList& operator=(PooledLinkedList&& that) {
    nodes_ = std::move(that.nodes_);
    return *this;
  }

  const std::vector<Node>& nodes() const { return nodes_; }

 private:
  std::vector<Node> nodes_;
};



}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_POOLED_LINKED_LIST_H_
