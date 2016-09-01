// Copyright (c) 2016 Google Inc.
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

#include "enum_set.h"

#include "spirv/1.1/spirv.hpp"

namespace {

// Determines whether the given enum value can be represented
// as a bit in a uint64_t mask. If so, then returns that mask bit.
// Otherwise, returns 0.
uint64_t AsMask(uint32_t word) {
  if (word > 63) return 0;
  return uint64_t(1) << word;
}
}

namespace libspirv {

template<typename EnumType>
void EnumSet<EnumType>::Add(uint32_t word) {
  if (auto new_bits = AsMask(word)) {
    mask_ |= new_bits;
  } else {
    Overflow().insert(word);
  }
}

template<typename EnumType>
bool EnumSet<EnumType>::Contains(uint32_t word) const {
  // We shouldn't call Overflow() since this is a const method.
  if (auto bits = AsMask(word)) {
    return mask_ & bits;
  } else if (auto overflow = overflow_.get()) {
    return overflow->find(word) != overflow->end();
  }
  // The word is large, but the set doesn't have large members, so
  // it doesn't have an overflow set.
  return false;
}

// Applies f to each capability in the set, in order from smallest enum
// value to largest.
void CapabilitySet::ForEach(std::function<void(SpvCapability)> f) const {
  for (uint32_t i = 0; i < 64; ++i) {
    if (mask_ & AsMask(i)) f(static_cast<SpvCapability>(i));
  }
  if (overflow_) {
    for (uint32_t c : *overflow_) f(static_cast<SpvCapability>(c));
  }
}

}  // namespace libspirv
