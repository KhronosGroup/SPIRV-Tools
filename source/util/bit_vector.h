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

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace spvtools {
namespace utils {

// Implements a bit vector class.
//
// All bits default to zero, and the upper bound is 2^32-1.
class BitVector {
 public:
  // Creates a bit vector contianing 0s.
  BitVector() : bits(1024 / kBitContainerSize, 0) {}

  // Sets the |i|th bit to 1.
  void Set(uint32_t i) {
    uint32_t element_index = i / kBitContainerSize;
    uint32_t bit_in_element = i % kBitContainerSize;

    if (element_index >= bits.size()) {
      bits.resize(element_index + 1, 0);
    }

    bits[element_index] |= (static_cast<BitContainer>(1) << bit_in_element);
  }

  // Sets the |i|th bit to 0.
  void Clear(uint32_t i) {
    uint32_t element_index = i / kBitContainerSize;
    uint32_t bit_in_element = i % kBitContainerSize;

    if (element_index >= bits.size()) {
      return;
    }
    bits[element_index] &= ~(static_cast<BitContainer>(1) << bit_in_element);
  }

  // Returns the |i|th bit.
  bool Get(uint32_t i) const {
    uint32_t element_index = i / kBitContainerSize;
    uint32_t bit_in_element = i % kBitContainerSize;

    if (element_index >= bits.size()) {
      return false;
    }

    return (bits[element_index] &
            (static_cast<BitContainer>(1) << bit_in_element)) != 0;
  }

  void ReportDensity(std::ostream& out);

 private:
  using BitContainer = uint64_t;
  static const uint32_t kBitContainerSize = 64;
  std::vector<BitContainer> bits;
};

}  // namespace utils
}  // namespace spvtools

#endif  // LIBSPIRV_UTILS_BIT_VECTOR_H_
