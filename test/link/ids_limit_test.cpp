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

class IdsLimit : public spvtools::LinkerTest {
 public:
  IdsLimit() { binaries.reserve(2); }

  virtual void SetUp() {
    binaries.push_back({
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      0x2FFFFFu, // NOTE: Bound
      0u,        // NOTE: Schema; reserved
    });
    binaries.push_back({
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      0x100000u, // NOTE: Bound
      0u,        // NOTE: Schema; reserved
    });
  }
  virtual void TearDown() { binaries.clear(); }

  spvtools::Binaries binaries;
};

TEST_F(IdsLimit, Default) {
  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_SUCCESS, linker.Link(binaries, linked_binary));
  ASSERT_EQ(0x3FFFFEu, linked_binary[3]);
}

TEST_F(IdsLimit, OverLimit) {
  binaries.push_back({
    SpvMagicNumber,
    SpvVersion,
    SPV_GENERATOR_CODEPLAY,
    3u,  // NOTE: Bound
    0u,  // NOTE: Schema; reserved
  });

  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_ERROR_INVALID_ID, linker.Link(binaries, linked_binary));
}

}  // anonymous namespace
