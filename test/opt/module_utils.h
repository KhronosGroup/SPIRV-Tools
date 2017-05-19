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

#ifndef LIBSPIRV_TEST_OPT_MODULE_UTILS_H_
#define LIBSPIRV_TEST_OPT_MODULE_UTILS_H_

#include <vector>
#include "opt/opt_module.h"

namespace spvtest {

inline uint32_t GetIdBound(const spvtools::ir::Module& m) {
  std::vector<uint32_t> binary;
  m.ToBinary(&binary, false);
  // The 5-word header must always exist.
  EXPECT_LE(5u, binary.size());
  // The bound is the fourth word.
  return binary[3];
}

}  // namespace spvtest

#endif // LIBSPIRV_TEST_OPT_MODULE_UTILS_H_
