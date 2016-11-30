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
  spirv << header + R"(
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

