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
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeStruct)";
  for (int i = 0; i < 16383; ++i) {
    spirv << " %1";
  }
  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLimits, structNumMembersExceededBad) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeStruct)";
  for (int i = 0; i < 16384; ++i) {
    spirv << " %1";
  }
  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of OpTypeStruct members (16384) has exceeded "
                        "the limit (16383)."));
}

// Valid: Switch statement has 16,383 branches.
TEST_F(ValidateLimits, switchNumBranchesGood) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpConstant %3 1234
%5 = OpFunction %1 None %2
%7 = OpLabel
%8 = OpIAdd %3 %4 %4
%9 = OpSwitch %4 %10)";

  // Now add the (literal, label) pairs
  for (int i = 0; i < 16383; ++i) {
    spirv << " 1 %10";
  }

  spirv << R"(
%10 = OpLabel
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Switch statement has 16,384 branches.
TEST_F(ValidateLimits, switchNumBranchesBad) {
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeInt 32 0
%4 = OpConstant %3 1234
%5 = OpFunction %1 None %2
%7 = OpLabel
%8 = OpIAdd %3 %4 %4
%9 = OpSwitch %4 %10)";

  // Now add the (literal, label) pairs
  for (int i = 0; i < 16384; ++i) {
    spirv << " 1 %10";
  }

  spirv << R"(
%10 = OpLabel
OpReturn
OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of (literal, label) pairs in OpSwitch (16384) "
                        "exceeds the limit (16383)."));
}

// Valid: OpTypeFunction with 255 arguments.
TEST_F(ValidateLimits, OpTypeFunctionGood) {
  int num_args = 255;
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFunction %1)";
  // add parameters
  for (int i = 0; i < num_args; ++i) {
    spirv << " %1";
  }
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: OpTypeFunction with 256 arguments. (limit is 255 according to the
// spec Universal Limits (2.17).
TEST_F(ValidateLimits, OpTypeFunctionBad) {
  int num_args = 256;
  std::ostringstream spirv;
  spirv << header << R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFunction %1)";
  for (int i = 0; i < num_args; ++i) {
    spirv << " %1";
  }
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpTypeFunction may not take more than 255 arguments. "
                        "OpTypeFunction <id> '2' has 256 arguments."));
}

// Valid: module has 65,535 global variables.
TEST_F(ValidateLimits, NumGlobalVarsGood) {
  int num_globals = 65535;
  std::ostringstream spirv;
  spirv << header << R"(
     %int = OpTypeInt 32 0
%_ptr_int = OpTypePointer Input %int
  )";

  for (int i = 0; i < num_globals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Input\n";
  }

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: module has 65,536 global variables (limit is 65,535).
TEST_F(ValidateLimits, NumGlobalVarsBad) {
  int num_globals = 65536;
  std::ostringstream spirv;
  spirv << header << R"(
     %int = OpTypeInt 32 0
%_ptr_int = OpTypePointer Input %int
  )";

  for (int i = 0; i < num_globals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Input\n";
  }

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of Global Variables (Storage Class other than "
                        "'Function') exceeded the valid limit (65535)."));
}

// Valid: module has 524,287 local variables.
TEST_F(ValidateLimits, NumLocalVarsGood) {
  int num_locals = 524287;
  std::ostringstream spirv;
  spirv << header << R"(
 %int      = OpTypeInt 32 0
 %_ptr_int = OpTypePointer Function %int
 %voidt    = OpTypeVoid
 %funct    = OpTypeFunction %voidt
 %main     = OpFunction %voidt None %funct
 %entry    = OpLabel
  )";

  for (int i = 0; i < num_locals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Function\n";
  }

  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: module has 524,288 local variables (limit is 524,287).
TEST_F(ValidateLimits, NumLocalVarsBad) {
  int num_locals = 524288;
  std::ostringstream spirv;
  spirv << header << R"(
 %int      = OpTypeInt 32 0
 %_ptr_int = OpTypePointer Function %int
 %voidt    = OpTypeVoid
 %funct    = OpTypeFunction %voidt
 %main     = OpFunction %voidt None %funct
 %entry    = OpLabel
  )";

  for (int i = 0; i < num_locals; ++i) {
    spirv << "%var_" << i << " = OpVariable %_ptr_int Function\n";
  }

  spirv << R"(
    OpReturn
    OpFunctionEnd
  )";

  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Number of local variables ('Function' Storage Class) "
                        "exceeded the valid limit (524287)."));
}

// Valid: Structure nesting depth of 255.
TEST_F(ValidateLimits, StructNestingDepthGood) {
  std::ostringstream spirv;
  spirv << header << R"(
    %int = OpTypeInt 32 0
    %s_depth_1  = OpTypeStruct %int
  )";
  for(auto i=2; i<=255; ++i) {
    spirv << "%s_depth_" << i << " = OpTypeStruct %int %s_depth_" << i-1;
    spirv << "\n";
  }
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Invalid: Structure nesting depth of 256.
TEST_F(ValidateLimits, StructNestingDepthBad) {
  std::ostringstream spirv;
  spirv << header << R"(
    %int = OpTypeInt 32 0
    %s_depth_1  = OpTypeStruct %int
  )";
  for(auto i=2; i<=256; ++i) {
    spirv << "%s_depth_" << i << " = OpTypeStruct %int %s_depth_" << i-1;
    spirv << "\n";
  }
  CompileSuccessfully(spirv.str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Structure Nesting Depth may not be larger than 255. Found 256."));
}
