// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Validation tests for Data Rules.

#include "unit_spirv.h"
#include "ValidateFixtures.h"
#include "gmock/gmock.h"

#include <sstream>
#include <string>
#include <utility>

using ::testing::HasSubstr;
using ::testing::MatchesRegex;

using std::string;
using std::pair;
using std::stringstream;

using ValidateData = spvtest::ValidateBase<pair<string, bool>>;

string header = R"(
     OpCapability Shader
     OpMemoryModel Logical GLSL450
%1 = OpTypeFloat 32
)";
string header_with_vec16_cap = R"(
     OpCapability Shader
     OpCapability Vector16
     OpMemoryModel Logical GLSL450
%1 = OpTypeFloat 32
)";

string invalid_comp_error = "Illegal number of components";
string missing_cap_error = "requires the Vector16 capability";

TEST_F(ValidateData, vec0) {
  string str = header + "%2 = OpTypeVector %1 0";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_comp_error));
}

TEST_F(ValidateData, vec1) {
  string str = header + "%2 = OpTypeVector %1 1";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_comp_error));
}

TEST_F(ValidateData, vec2) {
  string str = header + "%2 = OpTypeVector %1 2";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec3) {
  string str = header + "%2 = OpTypeVector %1 3";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec4) {
  string str = header + "%2 = OpTypeVector %1 4";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec5) {
  string str = header + "%2 = OpTypeVector %1 5";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_comp_error));
}

TEST_F(ValidateData, vec8) {
  string str = header + "%2 = OpTypeVector %1 8";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(missing_cap_error));
}

TEST_F(ValidateData, vec8_with_capability) {
  string str = header_with_vec16_cap + "%2 = OpTypeVector %1 8";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec16) {
  string str = header + "%2 = OpTypeVector %1 16";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(missing_cap_error));
}

TEST_F(ValidateData, vec16_with_capability) {
  string str = header_with_vec16_cap + "%2 = OpTypeVector %1 16";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateData, vec15) {
  string str = header + "%2 = OpTypeVector %1 15";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(invalid_comp_error));
}
