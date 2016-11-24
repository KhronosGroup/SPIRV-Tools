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

TEST_F(ValidateLimits, idBoundBad) {
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
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Result <id> '64' exceeds the ID bound '3'"));
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

