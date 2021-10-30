// Copyright (c) 2017 Pierre Moreau
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

#include <string>

#include "gmock/gmock.h"
#include "test/link/linker_fixture.h"

namespace spvtools {
namespace {

using ::testing::HasSubstr;
using IdsLimit = spvtest::LinkerTest;

spvtest::Binary CreateBinary(uint32_t id_bound) {
  return {
      // clang-format off
      // Header
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_WORD(SPV_GENERATOR_KHRONOS, 0),
      id_bound,  // NOTE: Bound
      0u         // NOTE: Schema; reserved
      // clang-format on
  };
}

TEST_F(IdsLimit, UnderLimit) {
  spvtest::Binaries binaries = {CreateBinary(0x2FFFFFu),
                                CreateBinary(0x100000u)};

  spvtest::Binary linked_binary;
  ASSERT_EQ(SPV_SUCCESS, Link(binaries, &linked_binary)) << GetErrorMessage();
  EXPECT_THAT(GetErrorMessage(), std::string());
  EXPECT_EQ(0x3FFFFFu, linked_binary[3]);
}

TEST_F(IdsLimit, OverLimit) {
  spvtest::Binaries binaries = {CreateBinary(0x2FFFFFu),
                                CreateBinary(0x100000u), CreateBinary(3u)};

  spvtest::Binary linked_binary;

  ASSERT_EQ(SPV_SUCCESS, Link(binaries, &linked_binary)) << GetErrorMessage();
  EXPECT_THAT(
      GetErrorMessage(),
      HasSubstr("The minimum limit of IDs, 4194303, was exceeded: 4194304 is "
                "the current ID bound."));
  EXPECT_EQ(0x400000u, linked_binary[3]);
}

TEST_F(IdsLimit, Overflow) {
  spvtest::Binaries binaries = {CreateBinary(0xFFFFFFFFu),
                                CreateBinary(0x00000002u)};

  spvtest::Binary linked_binary;

  EXPECT_EQ(SPV_ERROR_INVALID_DATA, Link(binaries, &linked_binary));
  EXPECT_THAT(
      GetErrorMessage(),
      HasSubstr("Too many IDs (4294967296): combining all modules would "
                "overflow the 32-bit word of the SPIR-V header."));
}

}  // namespace
}  // namespace spvtools
