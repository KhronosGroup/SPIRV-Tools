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

#include "bit_vector.h"

#include <iostream>

namespace spvtools {
namespace utils {

void BitVector::ReportDensity(std::ostream& out) {
  uint32_t count = 0;

  for (BitContainer e : bits_) {
    while (e != 0) {
      if ((e & 1) != 0) {
        ++count;
      }
      e = e >> 1;
    }
  }

  out << "count=" << count
      << ", total size (bytes)=" << bits_.size() * sizeof(BitContainer)
      << ", bytes per element="
      << (double)(bits_.size() * sizeof(BitContainer)) / (double)(count);
}
}  // namespace utils
}  // namespace spvtools
