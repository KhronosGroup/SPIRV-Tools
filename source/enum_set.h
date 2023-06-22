// Copyright (c) 2023 Google Inc.
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

#include <stdint.h>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#ifndef SOURCE_ENUM_SET_H_
#define SOURCE_ENUM_SET_H_

#include "source/latest_version_spirv_header.h"

namespace spvtools {

template <typename T>
class EnumSet {
  using BucketType = uint64_t;
  using ElementType = std::underlying_type_t<T>;
  static_assert(std::is_enum_v<T>, "EnumSets only works with enums.");
  static_assert(std::is_signed_v<ElementType> == false,
                "EnumSet doesn't supports signed enums.");
  static_assert(sizeof(T) * 8ULL <= std::numeric_limits<ElementType>::max());
  static constexpr size_t BUCKET_SIZE = sizeof(BucketType) * 8ULL;

  struct Bucket {
    BucketType data;
    T start;
  };

  std::vector<Bucket> buckets;

  static constexpr inline size_t compute_bucket_index(T value) {
    return static_cast<size_t>(value) / BUCKET_SIZE;
  }

  static constexpr inline size_t compute_bucket_offset(T value) {
    return static_cast<ElementType>(value) % BUCKET_SIZE;
  }

  static constexpr inline T compute_bucket_start(T value) {
    return static_cast<T>(BUCKET_SIZE * compute_bucket_index(value));
  }

  static constexpr inline uint64_t compute_mask_for_value(T value) {
    return 1ULL << compute_bucket_offset(value);
  }

  static constexpr inline T get_value_from_bucket(const Bucket& bucket,
                                                  ElementType offset) {
    return static_cast<T>(static_cast<ElementType>(bucket.start) + offset);
  }

  size_t find_bucket_for_value(T value) const {
    // Set is empty, insert at 0.
    if (buckets.size() == 0) {
      return 0;
    }

    size_t index = std::min(buckets.size() - 1, compute_bucket_index(value));
    const T needle = compute_bucket_start(value);

    const T bucket_start = buckets[index].start;
    // Computed index is the correct one.
    if (bucket_start == needle) {
      return index;
    }

    // Bucket contains smaller values. Linear scan right.
    if (bucket_start < needle) {
      for (index += 1; index < buckets.size() && buckets[index].start < needle;
           index++) {
      }
      return index;
    }

    // Bucket contains larger values, insert front.
    if (index == 0) {
      return index;
    }

    for (index -= 1; index > 0 && buckets[index].start > needle; index--) {
    }
    return buckets[index].start >= needle ? index : index + 1;
  }

  void create_bucket_for(size_t index, T value) {
    Bucket bucket = {1ULL << compute_bucket_offset(value),
                     compute_bucket_start(value)};
    buckets.emplace(buckets.begin() + index, std::move(bucket));
  }

 public:
  EnumSet() : buckets(0) {}

  EnumSet(T value) : EnumSet() { Add(value); }

  EnumSet(std::initializer_list<T> data) : EnumSet() {
    for (auto item : data) {
      Add(item);
    }
  }

  EnumSet(uint32_t count, const T* values) : EnumSet() {
    for (uint32_t i = 0; i < count; i++) {
      Add(values[i]);
    }
  }

  EnumSet(const EnumSet& other) : buckets(other.buckets) {}

  EnumSet(EnumSet&& other) : buckets(std::move(other.buckets)) {}

  EnumSet& operator=(const EnumSet& other) {
    buckets = other.buckets;
    return *this;
  }

  void Add(T value) {
    const size_t index = find_bucket_for_value(value);
    if (index >= buckets.size() ||
        buckets[index].start != compute_bucket_start(value)) {
      create_bucket_for(index, value);
      return;
    }
    auto& bucket = buckets[index];
    bucket.data |= compute_mask_for_value(value);
  }

  void Remove(T value) {
    const size_t index = find_bucket_for_value(value);
    if (index >= buckets.size() ||
        buckets[index].start != compute_bucket_start(value)) {
      return;
    }
    auto& bucket = buckets[index];
    bucket.data &= ~compute_mask_for_value(value);
    if (bucket.data == 0) {
      buckets.erase(buckets.cbegin() + index);
    }
  }

  bool Contains(T value) const {
    const size_t index = find_bucket_for_value(value);
    if (index >= buckets.size() ||
        buckets[index].start != compute_bucket_start(value)) {
      return false;
    }
    auto& bucket = buckets[index];
    return bucket.data & compute_mask_for_value(value);
  }

  void ForEach(std::function<void(T)> f) const {
    for (const auto& bucket : buckets) {
      for (uint8_t i = 0; i < BUCKET_SIZE; i++) {
        if (bucket.data & (1ULL << i)) {
          f(get_value_from_bucket(bucket, i));
        }
      }
    }
  }

  bool IsEmpty() const { return buckets.size() == 0; }

  friend bool operator==(const EnumSet& lhs, const EnumSet& rhs) {
    if (lhs.buckets.size() != rhs.buckets.size()) {
      return false;
    }

    for (size_t i = 0; i < lhs.buckets.size(); i++) {
      if (rhs.buckets[i].start != lhs.buckets[i].start ||
          rhs.buckets[i].data != lhs.buckets[i].data) {
        return false;
      }
    }
    return true;
  }

  friend bool operator!=(const EnumSet& lhs, const EnumSet& rhs) {
    return !(lhs == rhs);
  }

  bool HasAnyOf(const EnumSet<T>& in_set) const {
    if (in_set.IsEmpty()) {
      return true;
    }

    for (auto& lhs_bucket : buckets) {
      for (auto& rhs_bucket : in_set.buckets) {
        if (lhs_bucket.start != rhs_bucket.start) {
          continue;
        }

        if (lhs_bucket.data & rhs_bucket.data) {
          return true;
        }
      }
    }
    return false;
  }
};

// A set of spv::Capability, optimized for small capability values.
using CapabilitySet = EnumSet<spv::Capability>;

}  // namespace spvtools

#endif  // SOURCE_ENUM_SET_H_
