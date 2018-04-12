// Copyright (c) 2018 Google LLC
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

#ifndef LIBSPIRV_UTILS_BIT_VECTOR_H_
#define LIBSPIRV_UTILS_BIT_VECTOR_H_

#include <cassert>
#include <iostream>
#include <vector>

#include "ilist_node.h"

namespace spvtools {
namespace utils {

// Implements a bit vector class.
//
// All bits default to zero, and the upper bound is 2^32-1.
class BitVector {
 public:
  // Creates a bit vector contianing 0s.
  BitVector() : bits(1024 / sizeof(BitContainer), 0) {}

  // Sets the |i|th bit to 1.
  void Set(uint32_t i) {
    uint32_t element_index = i / sizeof(BitContainer);
    uint32_t bit_in_element = i % sizeof(BitContainer);

    if (element_index >= bits.size()) {
      bits.resize(element_index + 1, 0);
    }

    bits[element_index] |= (1 << bit_in_element);
  }

  // Sets the |i|th bit to 0.
  void Clear(uint32_t i) {
    uint32_t element_index = i / sizeof(BitContainer);
    uint32_t bit_in_element = i % sizeof(BitContainer);

    if (element_index >= bits.size()) {
      return;
    }
    bits[element_index] &= ~(1 << bit_in_element);
  }

  // Returns the |i|th bit.
  bool Get(uint32_t i) const {
    uint32_t element_index = i / sizeof(BitContainer);
    uint32_t bit_in_element = i % sizeof(BitContainer);

    if (element_index >= bits.size()) {
      return false;
    }

    return (bits[element_index] & (1 << bit_in_element)) != 0;
  }

  void ReportDensity(std::ostream& out) {
    uint32_t count = 0;

    for (BitContainer e : bits) {
      while (e != 0) {
        if ((e & 1) != 0) {
          ++count;
        }
        e = e >> 1;
      }
    }

    out << "count=" << count
        << ", total size (bytes)=" << bits.size() * sizeof(BitContainer)
        << ", bytes per element="
        << (double)(bits.size() * sizeof(BitContainer)) / (double)(count);
  }

 private:
  using BitContainer = uint64_t;
  std::vector<BitContainer> bits;
};

}  // namespace utils
}  // namespace spvtools

#endif  // LIBSPIRV_UTILS_BIT_VECTOR_H_
