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

#include "linker_test.h"

namespace {

class BinaryVersion : public spvtools::LinkerTest {
 public:
  BinaryVersion() { binaries.reserve(3 * 6); }

  virtual void SetUp() {
    binaries.push_back({
      SpvMagicNumber,
      0x00000300u,
      SPV_GENERATOR_CODEPLAY,
      1u,  // NOTE: Bound
      0u   // NOTE: Schema; reserved
      });

    binaries.push_back({
      SpvMagicNumber,
      0x00000600u,
      SPV_GENERATOR_CODEPLAY,
      1u,  // NOTE: Bound
      0u   // NOTE: Schema; reserved
      });

    binaries.push_back({
      SpvMagicNumber,
      0x00000100u,
      SPV_GENERATOR_CODEPLAY,
      1u,  // NOTE: Bound
      0u   // NOTE: Schema; reserved
      });
  }
  virtual void TearDown() { binaries.clear(); }

  spvtools::Binaries binaries;
};

TEST_F(BinaryVersion, Default) {
  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_SUCCESS, linker.Link(binaries, linked_binary));

  ASSERT_EQ(0x00000600u, linked_binary[1]);
}

}  // anonymous namespace
