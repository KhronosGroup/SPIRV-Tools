// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Unit tests for ValidationState_t.

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "spirv/1.1/spirv.h"
#include "unit_spirv.h"
#include "val_fixtures.h"

#include "enum_set.h"
#include "val/construct.h"
#include "val/function.h"
#include "val/validation_state.h"
#include "validate.h"

namespace {
using ::testing::HasSubstr;
using libspirv::CapabilitySet;
using libspirv::ValidationState_t;
using std::vector;

// A test with a ValidationState_t member transparently.
class ValidationStateTest : public testing::Test {
 public:
  ValidationStateTest()
      : context_(spvContextCreate(SPV_ENV_UNIVERSAL_1_0)), state_(context_) {}

 protected:
  spv_context context_;
  ValidationState_t state_;
};

// A test of ValidationState_t::HasAnyOf().
using ValidationState_HasAnyOfTest = ValidationStateTest;

TEST_F(ValidationState_HasAnyOfTest, EmptyMask) {
  EXPECT_TRUE(state_.HasAnyOf({}));
  state_.RegisterCapability(SpvCapabilityMatrix);
  EXPECT_TRUE(state_.HasAnyOf({}));
  state_.RegisterCapability(SpvCapabilityImageMipmap);
  EXPECT_TRUE(state_.HasAnyOf({}));
  state_.RegisterCapability(SpvCapabilityPipes);
  EXPECT_TRUE(state_.HasAnyOf({}));
  state_.RegisterCapability(SpvCapabilityStorageImageArrayDynamicIndexing);
  EXPECT_TRUE(state_.HasAnyOf({}));
  state_.RegisterCapability(SpvCapabilityClipDistance);
  EXPECT_TRUE(state_.HasAnyOf({}));
  state_.RegisterCapability(SpvCapabilityStorageImageWriteWithoutFormat);
  EXPECT_TRUE(state_.HasAnyOf({}));
}

TEST_F(ValidationState_HasAnyOfTest, SingleCapMask) {
  EXPECT_FALSE(state_.HasAnyOf({SpvCapabilityMatrix}));
  EXPECT_FALSE(state_.HasAnyOf({SpvCapabilityImageMipmap}));
  state_.RegisterCapability(SpvCapabilityMatrix);
  EXPECT_TRUE(state_.HasAnyOf({SpvCapabilityMatrix}));
  EXPECT_FALSE(state_.HasAnyOf({SpvCapabilityImageMipmap}));
  state_.RegisterCapability(SpvCapabilityImageMipmap);
  EXPECT_TRUE(state_.HasAnyOf({SpvCapabilityMatrix}));
  EXPECT_TRUE(state_.HasAnyOf({SpvCapabilityImageMipmap}));
}

TEST_F(ValidationState_HasAnyOfTest, MultiCapMask) {
  const auto set1 =
      CapabilitySet{SpvCapabilitySampledRect, SpvCapabilityImageBuffer};
  const auto set2 = CapabilitySet{SpvCapabilityStorageImageWriteWithoutFormat,
                                  SpvCapabilityStorageImageReadWithoutFormat,
                                  SpvCapabilityGeometryStreams};
  EXPECT_FALSE(state_.HasAnyOf(set1));
  EXPECT_FALSE(state_.HasAnyOf(set2));
  state_.RegisterCapability(SpvCapabilityImageBuffer);
  EXPECT_TRUE(state_.HasAnyOf(set1));
  EXPECT_FALSE(state_.HasAnyOf(set2));
}

using ValidatePermissive = spvtest::ValidateBase<bool>;
TEST_F(ValidatePermissive, properlyChecksPermissiveModeGood) {
  const std::string spirv = R"(
    OpCapability Shader
    OpExtension "unrecognized_extension"
    OpMemoryModel Logical GLSL450
    %int = OpTypeInt 32 0
  )";
  CompileSuccessfully(spirv);

  // This should fail because the above SPIR-V module is missing an
  // OpEntryPoint.
  bool permissive = false;
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_0, permissive));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("No OpEntryPoint instruction was found."));

  // This should pass because we are using the permissive mode, and we are using
  // an unrecognized extension.
  permissive = true;
  EXPECT_EQ(SPV_SUCCESS,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_0, permissive));
}

}  // anonymous namespace
