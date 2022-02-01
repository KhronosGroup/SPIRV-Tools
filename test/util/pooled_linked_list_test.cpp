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

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "source/util/pooled_linked_list.h"

namespace spvtools {
namespace utils {
namespace {

using PooledLinkedListTest = ::testing::Test;

template <typename T>
static std::vector<T> ToVector(const PooledLinkedList<T>& pool,
                               const typename PooledLinkedList<T>::Head& head) {
  std::vector<T> vec;
  for (int32_t idx = head.head; idx != -1; idx = pool[idx].next) {
    vec.push_back(pool[idx].element);
  }
  return vec;
}

template <typename T>
static void AppendVector(PooledLinkedList<T>& pool,
                         typename PooledLinkedList<T>::Head& head,
                         const std::vector<T>& vec){
  for (const T& t : vec) {
    pool.push_back(head, t);
  }
}


TEST(PooledLinkedListTest, Empty) {
  PooledLinkedList<uint32_t> ll;
  PooledLinkedList<uint32_t>::Head head;
  EXPECT_TRUE(ll.empty(head));

  ll.push_back(head, 1u);
  EXPECT_TRUE(!ll.empty(head));
}

TEST(PooledLinkedListTest, PushBack) {
  const std::vector<uint32_t> vec = {1, 2, 3, 4, 5, 6};

  PooledLinkedList<uint32_t> ll;
  PooledLinkedList<uint32_t>::Head head;

  AppendVector(ll, head, vec);
  EXPECT_EQ(vec, ToVector(ll, head));
}

TEST(PooledLinkedListTest, RemoveFirst) {
  const std::vector<uint32_t> vec = {1, 2, 3, 4, 5, 6};

  PooledLinkedList<uint32_t> ll;
  PooledLinkedList<uint32_t>::Head head;

  EXPECT_FALSE(ll.remove_first(head, 0));
  AppendVector(ll, head, vec);
  EXPECT_FALSE(ll.remove_first(head, 0));

  std::vector<uint32_t> tmp = vec;
  while (!tmp.empty()) {
    size_t mid = tmp.size() / 2;
    uint32_t elt = tmp[mid];
    tmp.erase(tmp.begin() + mid);

    EXPECT_TRUE(ll.remove_first(head, elt));
    EXPECT_FALSE(ll.remove_first(head, elt));
    EXPECT_EQ(tmp, ToVector(ll, head));
  }
  EXPECT_TRUE(ll.empty(head));
}

TEST(PooledLinkedListTest, RemoveFirst_Duplicates) {
  const std::vector<uint32_t> vec = {3, 1, 2, 3, 3, 3, 3, 4, 3, 5, 3, 6, 3};

  PooledLinkedList<uint32_t> ll;
  PooledLinkedList<uint32_t>::Head head;
  AppendVector(ll, head, vec);

  std::vector<uint32_t> tmp = vec;
  while (!tmp.empty()) {
    size_t mid = tmp.size() / 2;
    uint32_t elt = tmp[mid];
    tmp.erase(std::find(tmp.begin(), tmp.end(), elt));

    EXPECT_TRUE(ll.remove_first(head, elt));
    EXPECT_EQ(tmp, ToVector(ll, head));
  }
  EXPECT_TRUE(ll.empty(head));
}


TEST(PooledLinkedList, MoveTo) {
  const std::vector<uint32_t> vec = {1, 2, 3, 4, 5, 6};

  PooledLinkedList<uint32_t> ll;
  PooledLinkedList<uint32_t>::Head head1;
  PooledLinkedList<uint32_t>::Head head2;

  AppendVector(ll, head1, vec);
  AppendVector(ll, head2, vec);
  EXPECT_EQ(ll.nodes().size(), vec.size()*2);

  // Move a single one to a new list.
  PooledLinkedList<uint32_t> ll_new;
  PooledLinkedList<uint32_t>::Head head_new = head2;
  ll.move_to(head_new, ll_new);
  EXPECT_EQ(ll_new.nodes().size(), vec.size());

  // New list should match.
  EXPECT_EQ(ToVector(ll_new, head_new), vec);

  // So should old lists.
  EXPECT_EQ(ToVector(ll, head1), vec);
  EXPECT_EQ(ToVector(ll, head2), vec);
}

}  // namespace
}  // namespace utils
}  // namespace spvtools
