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

#include <gtest/gtest.h>

#include "opt/libspirv.hpp"

namespace {

using namespace spvtools;

TEST(CppInterface, SuccessfulRoundTrip) {
  const std::string input_text = "%2 = OpSizeOf %1 %3\n";
  SpvTools t(SPV_ENV_UNIVERSAL_1_1);

  std::vector<uint32_t> binary;
  EXPECT_EQ(SPV_SUCCESS, t.Assemble(input_text, &binary));
  EXPECT_TRUE(binary.size() > 5u);
  EXPECT_EQ(SpvMagicNumber, binary[0]);
  EXPECT_EQ(SpvVersion, binary[1]);

  std::string output_text;
  EXPECT_EQ(SPV_SUCCESS, t.Disassemble(binary, &output_text));
  EXPECT_EQ(input_text, output_text);
}

TEST(CppInterface, AssembleWithWrongTargetEnv) {
  const std::string input_text = "%r = OpSizeOf %type %pointer";
  SpvTools t(SPV_ENV_UNIVERSAL_1_0);

  std::vector<uint32_t> binary;
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT, t.Assemble(input_text, &binary));
}

TEST(CppInterface, DisassembleWithWrongTargetEnv) {
  const std::string input_text = "%r = OpSizeOf %type %pointer";
  SpvTools t11(SPV_ENV_UNIVERSAL_1_1);
  SpvTools t10(SPV_ENV_UNIVERSAL_1_0);

  std::vector<uint32_t> binary;
  EXPECT_EQ(SPV_SUCCESS, t11.Assemble(input_text, &binary));

  std::string output_text;
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, t10.Disassemble(binary, &output_text));
}

}  // anonymous namespace
