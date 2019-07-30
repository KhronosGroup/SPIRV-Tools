// Copyright (c) 2019 Google LLC
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

#include "source/opt/constants.h"
#include <gtest/gtest-param-test.h>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/types.h"

namespace spvtools {
namespace opt {
namespace analysis {
namespace {

using ConstantTest = ::testing::Test;
using ::testing::ValuesIn;

struct ConstantGetValueAsU64Case {
  bool is_signed;
  int width;
  std::vector<uint32_t> words;
  uint64_t expected_value;
};

using ConstantGetValueAsU64Test =
    ::testing::TestWithParam<ConstantGetValueAsU64Case>;

TEST_P(ConstantGetValueAsU64Test, Case) {
  Integer type(GetParam().width, GetParam().is_signed);
  IntConstant value(&type, GetParam().words);

  EXPECT_EQ(GetParam().expected_value, value.GetValueAsU64())
      << " got " << value.GetValueAsU64();
}

const uint32_t k32ones = ~uint32_t(0);
const uint64_t k64ones = ~uint64_t(0);
const uint64_t kBillion = uint64_t(1000 * 1000 * 1000);

INSTANTIATE_TEST_SUITE_P(AtMost32Bits, ConstantGetValueAsU64Test,
                         ValuesIn(std::vector<ConstantGetValueAsU64Case>{
                             // 4 bits
                             {false, 4, {0, 0}, 0},
                             {false, 4, {7, 0}, 7},
                             {false, 4, {15, 0}, 15},
                             {true, 4, {0, 0}, 0},
                             {true, 4, {7, 0}, 7},
                             {true, 4, {0xfffffff8, k32ones}, uint64_t(-8)},
                             {true, 4, {k32ones, k32ones}, k64ones},
                             // 16 bits
                             {false, 16, {0, 0}, 0},
                             {false, 16, {32767, 0}, 32767},
                             {false, 16, {32768, 0}, 32768},
                             {false, 16, {65000, 0}, 65000},
                             {true, 16, {0, 0}, 0},
                             {true, 16, {32767, 0}, 32767},
                             {true, 16, {0xfffffff8, k32ones}, uint64_t(-8)},
                             {true, 16, {k32ones, k32ones}, k64ones},
                             // 32 bits
                             {false, 32, {0, 0}, 0},
                             {false, 32, {1000000, 0}, 1000000},
                             {true, 32, {0xfffffff8, k32ones}, uint64_t(-8)},
                             {true, 32, {k32ones, k32ones}, k64ones},
                         }));

INSTANTIATE_TEST_SUITE_P(AtMost64Bits, ConstantGetValueAsU64Test,
                         ValuesIn(std::vector<ConstantGetValueAsU64Case>{
                             // 48 bits
                             {false, 48, {0, 0}, 0},
                             {false, 48, {5, 0}, 5},
                             {false,
                              48,
                              {0xfffffff8, 0x0000ffff},
                              (uint64_t(0xffff) << 32) | uint64_t(0xfffffff8)},
                             {false, 48, {k32ones, k32ones}, uint64_t(-1)},
                             {false, 48, {0xdcd65000, 1}, 8 * kBillion},
                             {true, 48, {0xfffffff8, k32ones}, uint64_t(-8)},
                             {true, 48, {k32ones, k32ones}, uint64_t(-1)},
                             {true, 48, {0xdcd65000, 1}, 8 * kBillion},

                             // 64 bits
                             {false, 64, {12, 0}, 12},
                             {false, 64, {0xdcd65000, 1}, 8 * kBillion},
                             {false, 48, {0xfffffff8, k32ones}, uint64_t(-8)},
                             {false, 64, {k32ones, k32ones}, k64ones},
                             {true, 64, {12, 0}, 12},
                             {true, 64, {0xdcd65000, 1}, 8 * kBillion},
                             {true, 48, {0xfffffff8, k32ones}, uint64_t(-8)},
                             {true, 64, {k32ones, k32ones}, k64ones},
                         }));

}  // namespace
}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
