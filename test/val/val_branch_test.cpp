// Copyright (c) 2017 NVIDIA Corporation
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

// Tests for branch instruction validator

#include <string>

#include "gmock/gmock.h"
#include "unit_spirv.h"
#include "val_fixtures.h"

namespace {

using ::testing::HasSubstr;
using ::testing::Not;

using std::string;

using ValidateBranches = spvtest::ValidateBase<bool>;

std::string GenerateCode(const std::string& main_body) {
    const std::string prefix =
R"(
OpCapability Shader
OpCapability Int64
OpCapability Float64
%ext_inst = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%f64 = OpTypeFloat 64
%u64 = OpTypeInt 64 0
%s64 = OpTypeInt 64 1
%boolvec2 = OpTypeVector %bool 2
%s32vec2 = OpTypeVector %s32 2
%u32vec2 = OpTypeVector %u32 2
%u64vec2 = OpTypeVector %u64 2
%f32vec2 = OpTypeVector %f32 2
%f64vec2 = OpTypeVector %f64 2
%boolvec3 = OpTypeVector %bool 3
%u32vec3 = OpTypeVector %u32 3
%u64vec3 = OpTypeVector %u64 3
%s32vec3 = OpTypeVector %s32 3
%f32vec3 = OpTypeVector %f32 3
%f64vec3 = OpTypeVector %f64 3
%boolvec4 = OpTypeVector %bool 4
%u32vec4 = OpTypeVector %u32 4
%u64vec4 = OpTypeVector %u64 4
%s32vec4 = OpTypeVector %s32 4
%f32vec4 = OpTypeVector %f32 4
%f64vec4 = OpTypeVector %f64 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_2 = OpConstant %f32 2
%f32_3 = OpConstant %f32 3
%f32_4 = OpConstant %f32 4
%f32_pi = OpConstant %f32 3.14159

%s32_0 = OpConstant %s32 0
%s32_1 = OpConstant %s32 1
%s32_2 = OpConstant %s32 2
%s32_3 = OpConstant %s32 3
%s32_4 = OpConstant %s32 4
%s32_m1 = OpConstant %s32 -1

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3
%u32_4 = OpConstant %u32 4

%f64_0 = OpConstant %f64 0
%f64_1 = OpConstant %f64 1
%f64_2 = OpConstant %f64 2
%f64_3 = OpConstant %f64 3
%f64_4 = OpConstant %f64 4

%s64_0 = OpConstant %s64 0
%s64_1 = OpConstant %s64 1
%s64_2 = OpConstant %s64 2
%s64_3 = OpConstant %s64 3
%s64_4 = OpConstant %s64 4
%s64_m1 = OpConstant %s64 -1

%u64_0 = OpConstant %u64 0
%u64_1 = OpConstant %u64 1
%u64_2 = OpConstant %u64 2
%u64_3 = OpConstant %u64 3
%u64_4 = OpConstant %u64 4

%main = OpFunction %void None %func
%main_entry = OpLabel)";

  const std::string suffix =
R"(
OpReturn
OpFunctionEnd)";

  return prefix + main_body + suffix;
}

std::string GenerateBranchCode(const std::string& branch)
{
    const std::string branch_prefix = R"(
        OpSelectionMerge %end None
    )";

    const std::string branch_suffix = R"(
      %target_true = OpLabel
      OpNop
      OpBranch %end

      %target_false = OpLabel
      OpNop
      OpBranch %end

      %end = OpLabel
  )";

  return GenerateCode(branch_prefix + branch + branch_suffix);
}

TEST_F(ValidateBranches, CondBranchSuccess) {
    const auto code = GenerateBranchCode(R"(
        %branch_cond = OpINotEqual %bool %s32_0 %s32_1
        OpBranchConditional %branch_cond %target_true %target_false
    )");

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateBranches, CondBranchConditionS32) {
    const auto code = GenerateBranchCode(R"(
        OpBranchConditional %s32_0 %target_true %target_false
    )");

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
}

TEST_F(ValidateBranches, CondBranchConditionU32) {
    const auto code = GenerateBranchCode(R"(
        OpBranchConditional %u32_0 %target_true %target_false
    )");

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
}

TEST_F(ValidateBranches, CondBranchConditionS64) {
    const auto code = GenerateBranchCode(R"(
        OpBranchConditional %s64_0 %target_true %target_false
    )");

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
}

TEST_F(ValidateBranches, CondBranchConditionU64) {
    const auto code = GenerateBranchCode(R"(
        OpBranchConditional %u64_0 %target_true %target_false
    )");

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
}

TEST_F(ValidateBranches, CondBranchConditionF32) {
    const auto code = GenerateBranchCode(R"(
        OpBranchConditional %f32_0 %target_true %target_false
    )");

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
}

TEST_F(ValidateBranches, CondBranchConditionF64) {
    const auto code = GenerateBranchCode(R"(
        OpBranchConditional %f64_0 %target_true %target_false
    )");

  CompileSuccessfully(code.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
}

} // anonymous namespace
