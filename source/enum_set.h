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

#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#ifndef SOURCE_ENUM_SET_H_
#define SOURCE_ENUM_SET_H_

#include "source/latest_version_spirv_header.h"

namespace spvtools {

// This container is optimized to store and retrieve unsigned enum values.
// The base model for this implementation is an open-addressing hashtable with
// linear probing. For small enums (max index < 64), all operations are O(1).
//
// - Enums are stored in buckets (64 contiguous values max per bucket)
// - Buckets ranges don't overlap, but don't have to be contiguous.
// - Enums are packed into 64-bits buckets, using 1 bit per enum value.
//
// Example:
//  - MyEnum { A = 0, B = 1, C = 64, D = 65 }
//  - 2 buckets are required:
//      - bucket 0, storing values in the range [ 0;  64[
//      - bucket 1, storing values in the range [64; 128[
//
// - Buckets are stored in a sorted vector (sorted by bucket range).
// - Retrieval is done by computing the theoretical bucket index using the enum
// value, and
//   doing a linear scan from this position.
// - Insertion is done by retrieving the bucket and either:
//   - inserting a new bucket in the sorted vector when no buckets has a
//   compatible range.
//   - setting the corresponding bit in the bucket.
//   This means insertion in the middle/beginning can cause a memmove when no
//   bucket is available. In our case, this happens at most 23 times for the
//   largest enum we have (Opcodes).
template <typename T>
class EnumSet {
 private:
  using BucketType = uint64_t;
  using ElementType = std::underlying_type_t<T>;
  static_assert(std::is_enum_v<T>, "EnumSets only works with enums.");
  static_assert(std::is_signed_v<ElementType> == false,
                "EnumSet doesn't supports signed enums.");

  // Each bucket can hold up to `kBucketSize` distinct, contiguous enum values.
  // The first value a bucket can hold must be aligned on `kBucketSize`.
  struct Bucket {
    // bit mask to store `kBucketSize` enums.
    BucketType data;
    // 1st enum this bucket can represent.
    T start;

    friend bool operator==(const Bucket& lhs, const Bucket& rhs) {
      return lhs.start == rhs.start && lhs.data == rhs.data;
    }
  };

  // How many distinct values can a bucket hold? 1 bit per value.
  static constexpr size_t kBucketSize = sizeof(BucketType) * 8ULL;

 public:
  // Creates an empty set.
  EnumSet() : buckets_(0) {}

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
  EnumSet(const EnumSet& other) : buckets_(other.buckets_) {}

  // Moves the EnumSet `other` into a new EnumSet.
  EnumSet(EnumSet&& other) : buckets_(std::move(other.buckets_)) {}

  // Deep-copies the EnumSet `other` into this EnumSet.
  EnumSet& operator=(const EnumSet& other) {
    buckets_ = other.buckets_;
    return *this;
  }

  // Add the enum value `value` into the set.
  // The set is unchanged if the value already exists.
  void Add(T value) {
    const size_t index = FindBucketForValue(value);
    if (index >= buckets_.size() ||
        buckets_[index].start != ComputeBucketStart(value)) {
      InsertBucketFor(index, value);
      return;
    }
    auto& bucket = buckets_[index];
    bucket.data |= ComputeMaskForValue(value);
  }

  // Removes the value `value` into the set.
  // The set is unchanged if the value is not in the set.
  void Remove(T value) {
    const size_t index = FindBucketForValue(value);
    if (index >= buckets_.size() ||
        buckets_[index].start != ComputeBucketStart(value)) {
      return;
    }
    auto& bucket = buckets_[index];
    bucket.data &= ~ComputeMaskForValue(value);
    if (bucket.data == 0) {
      buckets_.erase(buckets_.cbegin() + index);
    }
  }

  // Returns true if `value` is present in the set.
  bool Contains(T value) const {
    const size_t index = FindBucketForValue(value);
    if (index >= buckets_.size() ||
        buckets_[index].start != ComputeBucketStart(value)) {
      return false;
    }
    auto& bucket = buckets_[index];
    return bucket.data & ComputeMaskForValue(value);
  }

  // Calls `unaryFunction` once for each value in the set.
  // Values are sorted in increasing order using their numerical values.
  void ForEach(std::function<void(T)> unaryFunction) const {
    for (const auto& bucket : buckets_) {
      for (uint8_t i = 0; i < kBucketSize; i++) {
        if (bucket.data & (1ULL << i)) {
          unaryFunction(GetValueFromBucket(bucket, i));
        }
      }
    }
  }

  // Returns true if the set is holds no values.
  bool IsEmpty() const { return buckets_.size() == 0; }

  // Returns true if this set contains at least one value contained in `in_set`.
  // Note: If `in_set` is empty, this function returns true.
  bool HasAnyOf(const EnumSet<T>& in_set) const {
    if (in_set.IsEmpty()) {
      return true;
    }

    auto lhs = buckets_.cbegin();
    auto rhs = in_set.buckets_.cbegin();

    while (lhs != buckets_.cend() && rhs != in_set.buckets_.cend()) {
      if (lhs->start == rhs->start) {
        if (lhs->data & rhs->data) {
          // At least 1 bit is shared. Early return.
          return true;
        }

        lhs++;
        rhs++;
        continue;
      }

      // LHS bucket is smaller than the current RHS bucket. Catching up on RHS.
      if (lhs->start < rhs->start) {
        lhs++;
        continue;
      }

      // Otherwise, RHS needs to catch up on LHS.
      rhs++;
    }

    return false;
  }

 private:
  // Returns the index of the last bucket in which `value` could be stored.
  static constexpr inline size_t ComputeLargestPossibleBucketIndexFor(T value) {
    return static_cast<size_t>(value) / kBucketSize;
  }

  // Returns the smallest enum value that could be contained in the same bucket
  // as `value`.
  static constexpr inline T ComputeBucketStart(T value) {
    return static_cast<T>(kBucketSize *
                          ComputeLargestPossibleBucketIndexFor(value));
  }

  //  Returns the index of the bit that corresponds to `value` in the bucket.
  static constexpr inline size_t ComputeBucketOffset(T value) {
    return static_cast<ElementType>(value) % kBucketSize;
  }

  // Returns the bitmask used to represent the enum `value` in its bucket.
  static constexpr inline BucketType ComputeMaskForValue(T value) {
    return 1ULL << ComputeBucketOffset(value);
  }

  // Returns the `enum` stored in `bucket` at `offset`.
  // `offset` is the bit-offset in the bucket storage.
  static constexpr inline T GetValueFromBucket(const Bucket& bucket,
                                               ElementType offset) {
    return static_cast<T>(static_cast<ElementType>(bucket.start) + offset);
  }

  // For a given enum `value`, finds the bucket index that could contain this
  // value. If no such bucket is found, the index at which the new bucket should
  // be inserted is returned.
  size_t FindBucketForValue(T value) const {
    // Set is empty, insert at 0.
    if (buckets_.size() == 0) {
      return 0;
    }

    const T wanted_start = ComputeBucketStart(value);
    assert(buckets_.size() > 0 &&
           "Size must not be 0 here. Has the code above changed?");
    size_t index = std::min(buckets_.size() - 1,
                            ComputeLargestPossibleBucketIndexFor(value));

    // This loops behaves like std::upper_bound with a reverse iterator.
    // Buckets are sorted. 3 main cases:
    //  - The bucket matches
    //    => returns the bucket index.
    //  - The found bucket is larger
    //    => scans left until it finds the correct bucket, or insertion point.
    //  - The found bucket is smaller
    //    => We are at the end, so we return past-end index for insertion.
    for (; buckets_[index].start >= wanted_start; index--) {
      if (index == 0) {
        return 0;
      }
    }

    return index + 1;
  }

  // Creates a new bucket to store `value` and inserts it at `index`.
  // If the `index` is past the end, the bucket is inserted at the end of the
  // vector.
  void InsertBucketFor(size_t index, T value) {
    const T bucket_start = ComputeBucketStart(value);
    Bucket bucket = {1ULL << ComputeBucketOffset(value), bucket_start};
    auto it = buckets_.emplace(buckets_.begin() + index, std::move(bucket));
#if defined(NDEBUG)
    (void)it;  // Silencing unused variable warning.
#else
    assert(std::next(it) == buckets_.end() ||
           std::next(it)->start > bucket_start);
    assert(it == buckets_.begin() || std::prev(it)->start < bucket_start);
#endif
  }

  // Returns true if `lhs` and `rhs` hold the exact same values.
  friend bool operator==(const EnumSet& lhs, const EnumSet& rhs) {
    if (lhs.buckets_.size() != rhs.buckets_.size()) {
      return false;
    }
    return lhs.buckets_ == rhs.buckets_;
  }

  // Returns true if `lhs` and `rhs` hold at least 1 different value.
  friend bool operator!=(const EnumSet& lhs, const EnumSet& rhs) {
    return !(lhs == rhs);
  }

  // Storage for the buckets.
  std::vector<Bucket> buckets_;
};

// A set of spv::Capability.
using CapabilitySet = EnumSet<spv::Capability>;

}  // namespace spvtools

#endif  // SOURCE_ENUM_SET_H_
