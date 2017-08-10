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

class EntryPoints : public spvtools::LinkerTest {
 public:
  EntryPoints() { binaries.reserve(0xFFFF); }

  virtual void SetUp() {
      binaries.push_back({
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      10, // NOTE: Bound
      0,  // NOTE: Schema; reserved

      3 << SpvWordCountShift | SpvOpTypeFloat,
      1,  // NOTE: Result ID
      32, // NOTE: Width

      4 << SpvWordCountShift | SpvOpTypePointer,
      2,  // NOTE: Result ID
      SpvStorageClassInput,
      1,  // NOTE: Type ID

      2 << SpvWordCountShift | SpvOpTypeVoid,
      3,  // NOTE: Result ID

      3 << SpvWordCountShift | SpvOpTypeFunction,
      4,  // NOTE: Result ID
      3,  // NOTE: Return type

      5 << SpvWordCountShift | SpvOpFunction,
      3,  // NOTE: Result type
      5,  // NOTE: Result ID
      SpvFunctionControlMaskNone,
      4,  // NOTE: Function type

      2 << SpvWordCountShift | SpvOpLabel,
      6,  // NOTE: Result ID

      4 << SpvWordCountShift | SpvOpVariable,
      2,  // NOTE: Type ID
      7,  // NOTE: Result ID
      SpvStorageClassFunction,

      4 << SpvWordCountShift | SpvOpVariable,
      2,  // NOTE: Type ID
      8,  // NOTE: Result ID
      SpvStorageClassFunction,

      4 << SpvWordCountShift | SpvOpVariable,
      2,  // NOTE: Type ID
      9,  // NOTE: Result ID
      SpvStorageClassFunction,

      1 << SpvWordCountShift | SpvOpReturn,

      1 << SpvWordCountShift | SpvOpFunctionEnd
      });
    for (size_t i = 0u; i < 2; ++i) {
      spvtools::Binary binary = {
        SpvMagicNumber,
        SpvVersion,
        SPV_GENERATOR_CODEPLAY,
        103, // NOTE: Bound
        0,  // NOTE: Schema; reserved

        3 << SpvWordCountShift | SpvOpTypeFloat,
        1,  // NOTE: Result ID
        32, // NOTE: Width

        4 << SpvWordCountShift | SpvOpTypePointer,
        2,  // NOTE: Result ID
        SpvStorageClassInput,
        1  // NOTE: Type ID
      };

      for (uint32_t j = 0u; j < 0xFFFF / 2; ++j) {
        binary.push_back(4 << SpvWordCountShift | SpvOpVariable);
        binary.push_back(2);  // NOTE: Type ID
        binary.push_back(j + 3);  // NOTE: Result ID
        binary.push_back(SpvStorageClassInput);
      }
      binaries.push_back(binary);
    }
  }
  spvtools::Binaries& get_binaries() {
    return binaries;
  }
  virtual void TearDown() { binaries.clear(); }

  spvtools::Binaries binaries;
};

TEST_F(EntryPoints, Default) {
  const spvtools::Binaries& binaries = get_binaries();

  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_SUCCESS, linker.Link(binaries, linked_binary));
}

TEST_F(EntryPoints, OverLimit) {
  spvtools::Binaries& binaries = get_binaries();

  binaries.push_back({
    SpvMagicNumber,
    SpvVersion,
    SPV_GENERATOR_CODEPLAY,
    5,  // NOTE: Bound
    0,  // NOTE: Schema; reserved

    3 << SpvWordCountShift | SpvOpTypeFloat,
    1,  // NOTE: Result ID
    32, // NOTE: Width

    4 << SpvWordCountShift | SpvOpTypePointer,
    2,  // NOTE: Result ID
    SpvStorageClassInput,
    1,  // NOTE: Type ID

    4 << SpvWordCountShift | SpvOpVariable,
    2,  // NOTE: Type ID
    3,  // NOTE: Result ID
    SpvStorageClassInput,

    4 << SpvWordCountShift | SpvOpVariable,
    2,  // NOTE: Type ID
    4,  // NOTE: Result ID
    SpvStorageClassInput
  });

  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_ERROR_INTERNAL, linker.Link(binaries, linked_binary));
}

}  // anonymous namespace
