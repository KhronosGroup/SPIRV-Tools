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
  EntryPoints() { binaries.reserve(2); }

  virtual void SetUp() {
    binaries.push_back({
      SpvMagicNumber,
      SpvVersion,
      SPV_GENERATOR_CODEPLAY,
      2u,  // NOTE: Bound
      0u   // NOTE: Schema; reserved
      });
    binaries.push_back(binaries.front());
  }
  virtual void TearDown() { binaries.clear(); }

  spvtools::Binaries binaries;
};

TEST_F(EntryPoints, Default) {
  spvtools::Binary& first_binary = binaries[0];
  first_binary.push_back(4u << SpvWordCountShift | SpvOpEntryPoint);
  first_binary.push_back(SpvExecutionModelGLCompute);
  first_binary.push_back(1u);          // NOTE: entry ID
  first_binary.push_back(0x006F6F66u); // NOTE: "foo"

  spvtools::Binary& second_binary = binaries[1];
  second_binary.push_back(4u << SpvWordCountShift | SpvOpEntryPoint);
  second_binary.push_back(SpvExecutionModelGLCompute);
  second_binary.push_back(1u);          // NOTE: entry ID
  second_binary.push_back(0x00726162u); // NOTE: "bar"

  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_SUCCESS, linker.Link(binaries, linked_binary));

  ASSERT_EQ(0x006F6F66u, linked_binary[8]);
  ASSERT_EQ(0x00726162u, linked_binary[12]);
}

TEST_F(EntryPoints, DifferentModelSameName) {
  spvtools::Binary& first_binary = binaries[0];
  first_binary.push_back(4u << SpvWordCountShift | SpvOpEntryPoint);
  first_binary.push_back(SpvExecutionModelGLCompute);
  first_binary.push_back(1u);          // NOTE: entry ID
  first_binary.push_back(0x006F6F66u); // NOTE: "foo"

  spvtools::Binary& second_binary = binaries[1];
  second_binary.push_back(4u << SpvWordCountShift | SpvOpEntryPoint);
  second_binary.push_back(SpvExecutionModelVertex);
  second_binary.push_back(1u);          // NOTE: entry ID
  second_binary.push_back(0x006F6F66u); // NOTE: "foo"

  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_SUCCESS, linker.Link(binaries, linked_binary));

  ASSERT_EQ(SpvExecutionModelGLCompute, linked_binary[6]);
  ASSERT_EQ(0x006F6F66u, linked_binary[8]);
  ASSERT_EQ(SpvExecutionModelVertex, linked_binary[10]);
  ASSERT_EQ(0x006F6F66u, linked_binary[12]);
}

TEST_F(EntryPoints, SameModelAndName) {
  spvtools::Binary& first_binary = binaries[0];
  first_binary.push_back(4u << SpvWordCountShift | SpvOpEntryPoint);
  first_binary.push_back(SpvExecutionModelGLCompute);
  first_binary.push_back(1u);          // NOTE: entry ID
  first_binary.push_back(0x006F6F66u); // NOTE: "foo"

  spvtools::Binary& second_binary = binaries[1];
  second_binary.push_back(4u << SpvWordCountShift | SpvOpEntryPoint);
  second_binary.push_back(SpvExecutionModelGLCompute);
  second_binary.push_back(1u);          // NOTE: entry ID
  second_binary.push_back(0x006F6F66u); // NOTE: "foo"

  spvtools::Binary linked_binary;

  ASSERT_EQ(SPV_ERROR_INTERNAL, linker.Link(binaries, linked_binary));
}

}  // anonymous namespace
