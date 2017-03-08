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

// Tests for OpExtension validator rules.

#include <string>

#include "gmock/gmock.h"
#include "test_fixture.h"
#include "unit_spirv.h"
#include "val_fixtures.h"

namespace {

using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Values;

using std::string;

using ValidateKnownExtensions = spvtest::ValidateBase<string>;
using ValidateUnknownExtension = spvtest::ValidateBase<bool>;

// Returns expected error string if |extensions| is not recognized.
string GetErrorString(const std::string& extension) {
  return "Failed to parse OpExtension " + extension;
}

INSTANTIATE_TEST_CASE_P(ExpectSuccess, ValidateKnownExtensions, Values(
    "SPV_KHR_shader_ballot",
    "SPV_KHR_shader_draw_parameters",
    "SPV_KHR_subgroup_vote",
    "SPV_KHR_16bit_storage",
    "SPV_KHR_device_group",
    "SPV_KHR_multiview",
    "SPV_NV_sample_mask_override_coverage",
    "SPV_NV_geometry_shader_passthrough",
    "SPV_NV_viewport_array2",
    "SPV_NV_stereo_view_rendering",
    "SPV_NVX_multiview_per_view_attributes"
    ));

TEST_P(ValidateKnownExtensions, ExpectSuccess) {
  const std::string extension = GetParam();
  const string str =
      "OpCapability Shader\nOpCapability Linkage\nOpExtension \"" + extension +
      "\"\nOpMemoryModel Logical GLSL450";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  // TODO: Reanable once https://github.com/KhronosGroup/SPIRV-Tools/pull/577
  // is merged.
  // EXPECT_THAT(getDiagnosticString(), Not(HasSubstr(GetErrorString(extension))));
}

TEST_F(ValidateUnknownExtension, FailSilently) {
  const std::string extension = "ERROR_unknown_extension";
  const string str =
      "OpCapability Shader\nOpCapability Linkage\nOpExtension \"" + extension +
      "\"\nOpMemoryModel Logical GLSL450";
  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(GetErrorString(extension)));
}

}  // anonymous namespace
