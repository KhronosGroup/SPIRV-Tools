// Copyright (c) 2016 Google Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#ifndef LIBSPIRV_CAPABILITY_SET_H
#define LIBSPIRV_CAPABILITY_SET_H

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <utility>

#include "spirv/1.1/spirv.h"

namespace libspirv {

// A set of values of type SpvCapability.
// It is fast and compact for the common case, where capability values
// are at most 63.  But it can represent capabilities with larger values,
// as may appear in extensions.
class CapabilitySet {
 private:
  // The ForEach method will call the functor on capabilities in
  // enum value order (lowest to highest).  To make that easier, use
  // an ordered set for the overflow values.
  using OverflowSetType = std::set<uint32_t>;

 public:
  // Construct an empty set.
  CapabilitySet() = default;
  // Construct an set with just the given capability.
  explicit CapabilitySet(SpvCapability c) { Add(c); }
  // Construct an set from an initializer list of capabilities.
  CapabilitySet(std::initializer_list<SpvCapability> cs) {
    for (auto c : cs) Add(c);
  }
  // Copy constructor.
  CapabilitySet(const CapabilitySet& other) { *this = other; }
  // Move constructor.  The moved-from set is emptied.
  CapabilitySet(CapabilitySet&& other) {
    mask_ = other.mask_;
    overflow_ = std::move(other.overflow_);
    other.mask_ = 0;
    other.overflow_.reset(nullptr);
  }
  // Assignment operator.
  CapabilitySet& operator=(const CapabilitySet& other) {
    if (&other != this) {
      mask_ = other.mask_;
      overflow_.reset(other.overflow_ ? new OverflowSetType(*other.overflow_)
                                      : nullptr);
    }
    return *this;
  }

  // Adds the given capability to the set.  This has no effect if the
  // capability is already in the set.
  void Add(SpvCapability c) { Add(static_cast<uint32_t>(c)); }
  // Adds the given capability (as a 32-bit word) to the set.  This has no
  // effect if the capability is already in the set.
  void Add(uint32_t);

  // Returns true if this capability is in the set.
  bool Contains(SpvCapability c) const {
    return Contains(static_cast<uint32_t>(c));
  }
  // Returns true if the capability represented as a 32-bit word is in the set.
  bool Contains(uint32_t word) const;

  // Applies f to each capability in the set, in order from smallest enum
  // value to largest.
  void ForEach(std::function<void(SpvCapability)> f) const;

 private:
  // Returns true if the set has capabilities with value greater than 63.
  bool HasOverflow() { return overflow_ != nullptr; }

  // Ensures that overflow_set_ references a set.  A new empty set is
  // allocated if one doesn't exist yet.  Returns overflow_set_.
  OverflowSetType& Overflow() {
    if (overflow_.get() == nullptr) {
      overflow_.reset(new OverflowSetType);
    }
    return *overflow_;
  }

  // Capabilities with values up to 63 are stored as bits in this mask.
  uint64_t mask_ = 0;
  // Capabilities with values larger than 63 are stored in this set.
  // This set should normally be empty or very small.
  std::unique_ptr<OverflowSetType> overflow_ = {};
};

}  // namespace libspirv

#endif  // LIBSPIRV_CAPABILITY_SET_H
