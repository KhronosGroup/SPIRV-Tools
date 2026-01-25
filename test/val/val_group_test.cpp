// Copyright 2026 LunarG Inc.
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
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;

using ValidateGroup = spvtest::ValidateBase<bool>;

std::string GenerateShaderCode(const std::string& body) {
  std::ostringstream ss;
  ss << R"(
OpCapability Kernel
OpCapability Addresses
OpCapability Linkage
OpCapability Groups
OpCapability Float64
OpCapability Int64
OpMemoryModel Physical64 OpenCL
OpEntryPoint Kernel %main "main"
%float = OpTypeFloat 32
%float64 = OpTypeFloat 64
%uint = OpTypeInt 32 0
%uint64 = OpTypeInt 64 0
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%float_2 = OpConstant %float 2
%uint_array = OpTypeArray %uint %uint_2
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%main = OpFunction %void None %fn
%label = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";
  return ss.str();
}

TEST_F(ValidateGroup, AllAnyGood) {
  const std::string ss = R"(
    %x = OpGroupAll %bool %uint_2 %true
    %y = OpGroupAny %bool %uint_2 %true
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateGroup, FloatGood) {
  const std::string ss = R"(
    %a = OpGroupFAdd %float %uint_2 Reduce %float_2
    %b = OpGroupFMin %float %uint_2 Reduce %float_2
    %c = OpGroupFMax %float %uint_2 Reduce %float_2
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateGroup, IntGood) {
  const std::string ss = R"(
    %a = OpGroupIAdd %uint %uint_2 Reduce %uint_2
    %b = OpGroupSMin %uint %uint_2 Reduce %uint_2
    %c = OpGroupSMax %uint %uint_2 Reduce %uint_2
    %d = OpGroupUMin %uint %uint_2 Reduce %uint_2
    %e = OpGroupUMax %uint %uint_2 Reduce %uint_2
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateGroup, BroadcastGood) {
  const std::string ss = R"(
    %a = OpGroupBroadcast %uint %uint_2 %uint_2 %uint_0
    %b = OpGroupBroadcast %float %uint_2 %float_2 %uint_0
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateGroup, AllResult) {
  const std::string ss = R"(
    %x = OpGroupAll %uint %uint_2 %true
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result must be a boolean scalar type"));
}

TEST_F(ValidateGroup, AllPredicate) {
  const std::string ss = R"(
    %x = OpGroupAll %bool %uint_2 %uint_2
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Predicate must be a boolean scalar type"));
}

TEST_F(ValidateGroup, AnyResult) {
  const std::string ss = R"(
    %x = OpGroupAny %uint %uint_2 %true
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result must be a boolean scalar type"));
}

TEST_F(ValidateGroup, AnyPredicate) {
  const std::string ss = R"(
    %x = OpGroupAny %bool %uint_2 %uint_2
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Predicate must be a boolean scalar type"));
}

TEST_F(ValidateGroup, FAddWithInt) {
  const std::string ss = R"(
    %a = OpGroupFAdd %uint %uint_2 Reduce %uint_2
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result must be a scalar or vector of float type"));
}

TEST_F(ValidateGroup, FMaxWidthMismatch) {
  const std::string ss = R"(
    %a = OpGroupFAdd %float64 %uint_2 Reduce %float_2
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The type of X must match the Result type"));
}

TEST_F(ValidateGroup, IAddWithFloat) {
  const std::string ss = R"(
     %a = OpGroupIAdd %float %uint_2 Reduce %float_2
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result must be a scalar or vector of integer type"));
}

TEST_F(ValidateGroup, UMinWithArray) {
  const std::string ss = R"(
    %a = OpGroupUMin %uint_array %uint_2 Reduce %float_2
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result must be a scalar or vector of integer type"));
}

TEST_F(ValidateGroup, SMaxWidthMismatch) {
  const std::string ss = R"(
    %c = OpGroupSMax %uint64 %uint_2 Reduce %uint_2
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The type of X must match the Result type"));
}

TEST_F(ValidateGroup, BroadcastArray) {
  const std::string ss = R"(
    %a = OpGroupBroadcast %uint_array %uint_2 %uint_2 %uint_0
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result must be a scalar or vector of integer, "
                        "floating-point, or boolean type"));
}

TEST_F(ValidateGroup, BroadcastMismatch) {
  const std::string ss = R"(
    %b = OpGroupBroadcast %uint %uint_2 %float_2 %uint_0
  )";
  CompileSuccessfully(GenerateShaderCode(ss));
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The type of Value must match the Result type"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
