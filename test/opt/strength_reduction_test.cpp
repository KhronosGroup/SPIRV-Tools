// Copyright (c) 2017 Google Inc.
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

#include "assembly_builder.h"
#include "gmock/gmock.h"
#include "pass_fixture.h"
#include "pass_utils.h"

#include <algorithm>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace {

using namespace spvtools;

using ::testing::HasSubstr;
using ::testing::MatchesRegex;

using StrengthReductionBasicTest = PassTest<::testing::Test>;

// Test to make sure we replace 5*8.
TEST_F(StrengthReductionBasicTest, BasicReplaceMulBy8a) {
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
               "OpCapability Float64",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpName %main \"main\"",
       "%void = OpTypeVoid",
          "%4 = OpTypeFunction %void",
       "%uint = OpTypeInt 32 0",
     "%uint_5 = OpConstant %uint 5",
     "%uint_8 = OpConstant %uint 8",
       "%main = OpFunction %void None %4",
          "%8 = OpLabel",
          "%9 = OpIMul %uint %uint_5 %uint_8",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<opt::StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true);

  EXPECT_EQ(opt::Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  EXPECT_THAT(output, Not(HasSubstr("OpIMul")));
  EXPECT_THAT(output, HasSubstr("OpShiftLeftLogical %uint %uint_5 %uint_3"));
}

// Test to make sure we replace 8*5.
TEST_F(StrengthReductionBasicTest, BasicReplaceMulBy8b) {
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
               "OpCapability Float64",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpName %main \"main\"",
       "%void = OpTypeVoid",
          "%4 = OpTypeFunction %void",
        "%int = OpTypeInt 32 1",
      "%int_16 = OpConstant %int 16",
      "%int_5 = OpConstant %int 5",
       "%main = OpFunction %void None %4",
          "%8 = OpLabel",
          "%9 = OpIMul %int %int_16 %int_5",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<opt::StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true);

  EXPECT_EQ(opt::Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  EXPECT_THAT(output, Not(HasSubstr("OpIMul")));
  EXPECT_THAT(output, HasSubstr("OpShiftLeftLogical %int %int_5 %uint_4"));
}

// Test to make sure we don't replace 0*5.
TEST_F(StrengthReductionBasicTest, BasicDontReplace0) {
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
               "OpCapability Float64",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpName %main \"main\"",
       "%void = OpTypeVoid",
          "%4 = OpTypeFunction %void",
        "%int = OpTypeInt 32 1",
      "%int_0 = OpConstant %int 0",
      "%int_5 = OpConstant %int 5",
       "%main = OpFunction %void None %4",
          "%8 = OpLabel",
          "%9 = OpIMul %int %int_0 %int_5",
               "OpReturn",
               "OpFunctionEnd"
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<opt::StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true);

  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

// Test to make sure we do not replace a multiple of 5 and 7.
TEST_F(StrengthReductionBasicTest, BasicNoChange) {
  const std::vector<const char*> text = {
      // clang-format off
             "OpCapability Shader",
             "OpCapability Float64",
        "%1 = OpExtInstImport \"GLSL.std.450\"",
             "OpMemoryModel Logical GLSL450",
             "OpEntryPoint Vertex %2 \"main\"",
             "OpName %2 \"main\"",
        "%3 = OpTypeVoid",
        "%4 = OpTypeFunction %3",
        "%5 = OpTypeInt 32 1",
        "%6 = OpTypeInt 32 0",
        "%7 = OpConstant %5 5",
        "%8 = OpConstant %5 7",
        "%2 = OpFunction %3 None %4",
        "%9 = OpLabel",
        "%10 = OpIMul %5 %7 %8",
             "OpReturn",
             "OpFunctionEnd",
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<opt::StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true);

  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

// Test to make sure constants and types are reused and not duplicated.
TEST_F(StrengthReductionBasicTest, NoDuplicateConstantsAndTypes) {
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
               "OpCapability Float64",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %main \"main\"",
               "OpName %main \"main\"",
       "%void = OpTypeVoid",
          "%4 = OpTypeFunction %void",
       "%uint = OpTypeInt 32 0",
     "%uint_8 = OpConstant %uint 8",
     "%uint_3 = OpConstant %uint 3",
       "%main = OpFunction %void None %4",
          "%8 = OpLabel",
          "%9 = OpIMul %uint %uint_8 %uint_3",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<opt::StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true);

  EXPECT_EQ(opt::Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  EXPECT_THAT(output, Not(MatchesRegex(".*OpConstant %uint 3.*OpConstant %uint 3.*")));
  EXPECT_THAT(output, Not(MatchesRegex(".*OpTypeInt 32 0.*OpTypeInt 32 0.*")));
}

}  // anonymous namespace
