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

class MemoryModel : public spvtools::LinkerTest {
 public:
  MemoryModel() { }

  virtual void SetUp() { }
  virtual void TearDown() { }
};

TEST_F(MemoryModel, Default) {
  const spvtools::Binaries binaries = {
    {
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      1,  // NOTE: Bound
      0,  // NOTE: Schema; reserved
      3u << SpvWordCountShift | SpvOpMemoryModel,
        SpvAddressingModelLogical,
        SpvMemoryModelSimple
    },
    {
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      1,  // NOTE: Bound
      0,  // NOTE: Schema; reserved
      3u << SpvWordCountShift | SpvOpMemoryModel,
        SpvAddressingModelLogical,
        SpvMemoryModelSimple
      }
  };
  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_SUCCESS, linker.Link(binaries, linked_binary));

  ASSERT_EQ(SpvAddressingModelLogical, linked_binary[6]);
  ASSERT_EQ(SpvMemoryModelSimple,      linked_binary[7]);
}

TEST_F(MemoryModel, AddressingMismatch) {
  const spvtools::Binaries binaries = {
    {
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      1,  // NOTE: Bound
      0,  // NOTE: Schema; reserved
      3u << SpvWordCountShift | SpvOpMemoryModel,
        SpvAddressingModelLogical,
        SpvMemoryModelSimple
    },
    {
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      1,  // NOTE: Bound
      0,  // NOTE: Schema; reserved
      3u << SpvWordCountShift | SpvOpMemoryModel,
        SpvAddressingModelPhysical32,
        SpvMemoryModelSimple
      }
  };
  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_ERROR_INTERNAL, linker.Link(binaries, linked_binary));
}

TEST_F(MemoryModel, MemoryMismatch) {
  const spvtools::Binaries binaries = {
    {
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      1,  // NOTE: Bound
      0,  // NOTE: Schema; reserved
      3u << SpvWordCountShift | SpvOpMemoryModel,
        SpvAddressingModelLogical,
        SpvMemoryModelSimple
    },
    {
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      1,  // NOTE: Bound
      0,  // NOTE: Schema; reserved
      3u << SpvWordCountShift | SpvOpMemoryModel,
        SpvAddressingModelLogical,
        SpvMemoryModelGLSL450
      }
  };
  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_ERROR_INTERNAL, linker.Link(binaries, linked_binary));
}

}  // anonymous namespace
