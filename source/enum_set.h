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

  // Each bucket can hold up to `BUCKET_SIZE` distinct, contiguous enum values.
  // The first value a bucket can hold must be aligned on `BUCKET_SIZE`.
  struct Bucket {
    // bit mask to store `BUCKET_SIZE` enums.
    BucketType data;
    // 1st enum this bucket can represent.
    T start;
  };

  // How many distinct values can a bucket hold? 1 bit per value.
  static constexpr size_t BUCKET_SIZE = sizeof(BucketType) * 8ULL;

  // Returns the index of the bucket `value` would be stored in the best case.
  static constexpr inline size_t compute_bucket_index(T value) {
    return static_cast<size_t>(value) / BUCKET_SIZE;
  }

  // Returns the start of the bucket the enum `value` would belongs to.
  static constexpr inline size_t compute_bucket_offset(T value) {
    return static_cast<ElementType>(value) % BUCKET_SIZE;
  }

  // Returns the first storable enum value stored by the bucket that would
  // contain `value`.
  static constexpr inline T compute_bucket_start(T value) {
    return static_cast<T>(BUCKET_SIZE * compute_bucket_index(value));
  }

  // Returns the bitmask used to represent the enum `value` in its bucket.
  static constexpr inline uint64_t compute_mask_for_value(T value) {
    return 1ULL << compute_bucket_offset(value);
  }

  // Returns the `enum` stored in `bucket` at `offset`.
  // `offset` is the bit-offset in the bucket storage.
  static constexpr inline T get_value_from_bucket(const Bucket& bucket,
                                                  ElementType offset) {
    return static_cast<T>(static_cast<ElementType>(bucket.start) + offset);
  }

 public:
  // Creates an empty set.
  EnumSet() : buckets(0) {}

  // Creates a set and store `value` in it.
  EnumSet(T value) : EnumSet() { Add(value); }

  // Creates a set and stores each `values` in it.
  EnumSet(std::initializer_list<T> values) : EnumSet() {
    for (auto item : values) {
      Add(item);
    }
  }

  // Creates a set, and insert `count` enum values pointed by `array` in it.
  EnumSet(ElementType count, const T* array) : EnumSet() {
    for (ElementType i = 0; i < count; i++) {
      Add(array[i]);
    }
  }

  // Copies the EnumSet `other` into a new EnumSet.
  EnumSet(const EnumSet& other) : buckets(other.buckets) {}

  // Moves the EnumSet `other` into a new EnumSet.
  EnumSet(EnumSet&& other) : buckets(std::move(other.buckets)) {}

  // Deep-copies the EnumSet `other` into this EnumSet.
  EnumSet& operator=(const EnumSet& other) {
    buckets = other.buckets;
    return *this;
  }

  // Add the enum value `value` into the set.
  // The set is unchanged if the value already exists.
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

  // Removes the value `value` into the set.
  // The set is unchanged if the value is not in the set.
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

  // Returns true if `value` is present in the set.
  bool Contains(T value) const {
    const size_t index = find_bucket_for_value(value);
    if (index >= buckets.size() ||
        buckets[index].start != compute_bucket_start(value)) {
      return false;
    }
    auto& bucket = buckets[index];
    return bucket.data & compute_mask_for_value(value);
  }

  // Calls `unaryFunction` once for each value in the set.
  // Values are sorted in increasing order using their numerical values.
  void ForEach(std::function<void(T)> unaryFunction) const {
    for (const auto& bucket : buckets) {
      for (uint8_t i = 0; i < BUCKET_SIZE; i++) {
        if (bucket.data & (1ULL << i)) {
          unaryFunction(get_value_from_bucket(bucket, i));
        }
      }
    }
  }

  // Returns true if the set is holds no values.
  bool IsEmpty() const { return buckets.size() == 0; }

  // Returns true if `lhs` and `rhs` hold the exact same values.
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

  // Returns true if `lhs` and `rhs` hold at least 1 different value.
  friend bool operator!=(const EnumSet& lhs, const EnumSet& rhs) {
    return !(lhs == rhs);
  }

  // Returns true if this set contains at least one value contained in `in_set`.
  // Note: If `in_set` is empty, this function returns true.
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

 private:
  // Storage for the buckets.
  std::vector<Bucket> buckets;

  // For a given enum `value`, finds the bucket index that could contain this
  // value. If no such bucket is found, the index at which the new bucket should
  // be inserted is returned.
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

  // Creates a new bucket to store `value` and inserts it at `index`.
  // If the `index` is past the end, the bucket is inserted at the end of the
  // vector.
  void create_bucket_for(size_t index, T value) {
    Bucket bucket = {1ULL << compute_bucket_offset(value),
                     compute_bucket_start(value)};
    buckets.emplace(buckets.begin() + index, std::move(bucket));
  }
};

// A set of spv::Capability.
using CapabilitySet = EnumSet<spv::Capability>;

}  // namespace spvtools

#endif  // SOURCE_ENUM_SET_H_
