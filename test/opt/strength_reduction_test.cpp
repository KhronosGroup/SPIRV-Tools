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
#include "pass_fixture.h"
#include "pass_utils.h"

#include <algorithm>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace {

using namespace spvtools;

using StrengthReductionBasicTest = PassTest<::testing::Test>;

// Test to make sure we replace 5*8.
TEST_F(StrengthReductionBasicTest, BasicReplaceMulBy8a) {
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
        "%8 = OpConstant %5 8",
        "%2 = OpFunction %3 None %4",
        "%9 = OpLabel",
        "%10 = OpIMul %5 %7 %8",
             "OpReturn",
             "OpFunctionEnd",
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<opt::StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true);

  EXPECT_EQ(opt::Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  std::string::size_type pos = output.find("OpIMul");
  EXPECT_EQ(pos, std::string::npos);
  pos = output.find("OpShiftLeftLogical");
  EXPECT_NE(pos, std::string::npos);
}

// Test to make sure we replace 8*5.
TEST_F(StrengthReductionBasicTest, BasicReplaceMulBy8b) {
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
        "%7 = OpConstant %5 8",
        "%8 = OpConstant %5 5",
        "%2 = OpFunction %3 None %4",
        "%9 = OpLabel",
        "%10 = OpIMul %5 %7 %8",
             "OpReturn",
             "OpFunctionEnd",
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<opt::StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true);

  EXPECT_EQ(opt::Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  std::string::size_type pos = output.find("OpIMul");
  EXPECT_EQ(pos, std::string::npos);
  pos = output.find("OpShiftLeftLogical");
  EXPECT_NE(pos, std::string::npos);
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

// Test to make sure constants are reused and not duplicated.
TEST_F(StrengthReductionBasicTest, NoDuplicateConstants) {
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
        "%6 = OpTypeInt 32 0",
        "%7 = OpConstant %6 8",
        "%8 = OpConstant %6 3",
        "%2 = OpFunction %3 None %4",
        "%9 = OpLabel",
        "%10 = OpIMul %6 %7 %8",
             "OpReturn",
             "OpFunctionEnd",
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<opt::StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true);

  EXPECT_EQ(opt::Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  std::string::size_type pos1 = output.find("OpConstant %uint 3");
  EXPECT_NE(pos1, std::string::npos);
  std::string::size_type pos2 = output.rfind("OpConstant %uint 3");
  EXPECT_EQ(pos1, pos2);
}

// Test to make sure types are reused and not duplicated.
TEST_F(StrengthReductionBasicTest, NoDuplicateTypes) {
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
        "%6 = OpTypeInt 32 0",
        "%7 = OpConstant %6 8",
        "%8 = OpConstant %6 3",
        "%2 = OpFunction %3 None %4",
        "%9 = OpLabel",
        "%10 = OpIMul %6 %7 %8",
             "OpReturn",
             "OpFunctionEnd",
      // clang-format on
  };

  auto result = SinglePassRunAndDisassemble<opt::StrengthReductionPass>(
      JoinAllInsts(text), /* skip_nop = */ true);

  EXPECT_EQ(opt::Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& output = std::get<0>(result);
  std::string::size_type pos1 = output.find("OpTypeInt 32 0");
  EXPECT_NE(pos1, std::string::npos);
  std::string::size_type pos2 = output.rfind("OpTypeInt 32 0");
  EXPECT_EQ(pos1, pos2);
}
}  // anonymous namespace
