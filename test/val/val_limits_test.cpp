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

// Validation tests for Universal Limits. (Section 2.17 of the SPIR-V Spec)

#include <sstream>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "unit_spirv.h"
#include "val_fixtures.h"

using ::testing::HasSubstr;
using ::testing::MatchesRegex;

using std::string;

using ValidateLimits = spvtest::ValidateBase<bool>;

string header = R"(
     OpCapability Shader
     OpMemoryModel Logical GLSL450
)";

TEST_F(ValidateLimits, idLargerThanBoundBad) {
  string str = header + R"(
;  %i32 has ID 1
%i32    = OpTypeInt 32 1
%c      = OpConstant %i32 100

; Fake an instruction with 64 as the result id.
; !64 = OpConstantNull %i32
!0x3002e !1 !64
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Result <id> '64' must be less than the ID bound '3'."));
}

TEST_F(ValidateLimits, idEqualToBoundBad) {
  string str = header + R"(
;  %i32 has ID 1
%i32    = OpTypeInt 32 1
%c      = OpConstant %i32 100

; Fake an instruction with 64 as the result id.
; !64 = OpConstantNull %i32
!0x3002e !1 !64
)";

  CompileSuccessfully(str);

  // The largest ID used in this program is 64. Let's overwrite the ID bound in
  // the header to be 64. This should result in an error because all IDs must
  // satisfy: 0 < id < bound.
  OverwriteAssembledBinary(3, 64);

  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Result <id> '64' must be less than the ID bound '64'."));
}

TEST_F(ValidateLimits, structNumMembersGood) {
  string str = header + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeStruct)";
  for (int i = 0; i < 16383; ++i) {
    str += " %1";
  }
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLimits, structNumMembersExceededBad) {
  string str = header + R"(
%1 = OpTypeInt 32 0
%2 = OpTypeStruct)";
  for (int i = 0; i < 16384; ++i) {
    str += " %1";
  }
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of OpTypeStruct members (16384) has exceeded "
                        "the limit (16,383)."));
}

// Valid: Switch statement has 16,383 branches.
TEST_F(ValidateLimits, switchNumBranchesGood) {
  string str = header + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpConstant %3 1234
%5 = OpFunction %1 None %2
%7 = OpLabel
%8 = OpIAdd %4 %4 %4
%9 = OpSwitch %4 %10)";

  // Now add the (literal, label) pairs
  for (int i = 0; i < 16383; ++i) {
    str += " 1 %10";
  }

  str += R"(
%10 = OpLabel
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Switch statement has 16,384 branches.
TEST_F(ValidateLimits, switchNumBranchesBad) {
  string str = header + R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpConstant %3 1234
%5 = OpFunction %1 None %2
%7 = OpLabel
%8 = OpIAdd %4 %4 %4
%9 = OpSwitch %4 %10)";

  // Now add the (literal, label) pairs
  for (int i = 0; i < 16384; ++i) {
    str += " 1 %10";
  }

  str += R"(
%10 = OpLabel
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of (literal, label) pairs in OpSwitch (16384) "
                        "exceeds the limit (16,383)."));
}

// Valid: OpFunctionCall with 255 arguments.
TEST_F(ValidateLimits, OpFunctionCallGood) {
  int num_args = 255;
  string spirv = header + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2)";
  for (int i = 0; i < num_args; ++i) {
    spirv += " %2";
  }
  spirv += R"(
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
     OpReturnValue %7
     OpFunctionEnd

%9  = OpFunction %1 None %4
%10 = OpLabel
%11 = OpFunctionCall %2 %6)";
  for (int i = 0; i < num_args; ++i) {
    spirv += " %5";
  }
  spirv += R"(
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: OpFunctionCall with 256 arguments. (limit is 255 according to the
// spec Universal Limits (2.17).
TEST_F(ValidateLimits, OpFunctionCallBad) {
  int num_args = 256;
  string spirv = header + R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2)";
  for (int i = 0; i < num_args; ++i) {
    spirv += " %2";
  }
  spirv += R"(
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
     OpReturnValue %7
     OpFunctionEnd

%9  = OpFunction %1 None %4
%10 = OpLabel
%11 = OpFunctionCall %2 %6)";
  for (int i = 0; i < num_args; ++i) {
    spirv += " %5";
  }
  spirv += R"(
      OpReturn
      OpFunctionEnd)";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of actual arguments passed to OpFunctionCall "
                        "may not be larger than 255. OpFunctionCall <id> '11' "
                        "has 256 actual arguments."));
}
