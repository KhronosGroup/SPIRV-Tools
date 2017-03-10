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

#include "gtest/gtest.h"
#include "spirv/1.1/spirv.h"

#include "enum_set.h"
#include "extensions.h"
#include "spirv_validator_options.h"
#include "val/construct.h"
#include "val/function.h"
#include "val/validation_state.h"
#include "validate.h"

namespace {
using libspirv::CapabilitySet;
using libspirv::Extension;
using libspirv::ExtensionSet;
using libspirv::ValidationState_t;
using std::vector;

// A test with a ValidationState_t member transparently.
class ValidationStateTest : public testing::Test {
 public:
  ValidationStateTest()
      : context_(spvContextCreate(SPV_ENV_UNIVERSAL_1_0)),
        options_(spvValidatorOptionsCreate()),
        state_(context_, options_) {}

  ~ValidationStateTest() {
    spvContextDestroy(context_);
    spvValidatorOptionsDestroy(options_);
  }
 protected:
  spv_context context_;
  spv_validator_options options_;
  ValidationState_t state_;
};

// A test of ValidationState_t::HasAnyOfCapabilities().
using ValidationState_HasAnyOfCapabilities = ValidationStateTest;

TEST_F(ValidationState_HasAnyOfCapabilities, EmptyMask) {
  EXPECT_TRUE(state_.HasAnyOfCapabilities({}));
  state_.RegisterCapability(SpvCapabilityMatrix);
  EXPECT_TRUE(state_.HasAnyOfCapabilities({}));
  state_.RegisterCapability(SpvCapabilityImageMipmap);
  EXPECT_TRUE(state_.HasAnyOfCapabilities({}));
  state_.RegisterCapability(SpvCapabilityPipes);
  EXPECT_TRUE(state_.HasAnyOfCapabilities({}));
  state_.RegisterCapability(SpvCapabilityStorageImageArrayDynamicIndexing);
  EXPECT_TRUE(state_.HasAnyOfCapabilities({}));
  state_.RegisterCapability(SpvCapabilityClipDistance);
  EXPECT_TRUE(state_.HasAnyOfCapabilities({}));
  state_.RegisterCapability(SpvCapabilityStorageImageWriteWithoutFormat);
  EXPECT_TRUE(state_.HasAnyOfCapabilities({}));
}

TEST_F(ValidationState_HasAnyOfCapabilities, SingleCapMask) {
  EXPECT_FALSE(state_.HasAnyOfCapabilities({SpvCapabilityMatrix}));
  EXPECT_FALSE(state_.HasAnyOfCapabilities({SpvCapabilityImageMipmap}));
  state_.RegisterCapability(SpvCapabilityMatrix);
  EXPECT_TRUE(state_.HasAnyOfCapabilities({SpvCapabilityMatrix}));
  EXPECT_FALSE(state_.HasAnyOfCapabilities({SpvCapabilityImageMipmap}));
  state_.RegisterCapability(SpvCapabilityImageMipmap);
  EXPECT_TRUE(state_.HasAnyOfCapabilities({SpvCapabilityMatrix}));
  EXPECT_TRUE(state_.HasAnyOfCapabilities({SpvCapabilityImageMipmap}));
}

TEST_F(ValidationState_HasAnyOfCapabilities, MultiCapMask) {
  const auto set1 =
      CapabilitySet{SpvCapabilitySampledRect, SpvCapabilityImageBuffer};
  const auto set2 = CapabilitySet{SpvCapabilityStorageImageWriteWithoutFormat,
                                  SpvCapabilityStorageImageReadWithoutFormat,
                                  SpvCapabilityGeometryStreams};
  EXPECT_FALSE(state_.HasAnyOfCapabilities(set1));
  EXPECT_FALSE(state_.HasAnyOfCapabilities(set2));
  state_.RegisterCapability(SpvCapabilityImageBuffer);
  EXPECT_TRUE(state_.HasAnyOfCapabilities(set1));
  EXPECT_FALSE(state_.HasAnyOfCapabilities(set2));
}

// A test of ValidationState_t::HasAnyOfExtensions().
using ValidationState_HasAnyOfExtensions = ValidationStateTest;

TEST_F(ValidationState_HasAnyOfExtensions, EmptyMask) {
  EXPECT_TRUE(state_.HasAnyOfExtensions({}));
  state_.RegisterExtension(Extension::kSPV_KHR_shader_ballot);
  EXPECT_TRUE(state_.HasAnyOfExtensions({}));
  state_.RegisterExtension(Extension::kSPV_KHR_16bit_storage);
  EXPECT_TRUE(state_.HasAnyOfExtensions({}));
  state_.RegisterExtension(Extension::kSPV_NV_viewport_array2);
  EXPECT_TRUE(state_.HasAnyOfExtensions({}));
}

TEST_F(ValidationState_HasAnyOfExtensions, SingleCapMask) {
  EXPECT_FALSE(state_.HasAnyOfExtensions({Extension::kSPV_KHR_shader_ballot}));
  EXPECT_FALSE(state_.HasAnyOfExtensions({Extension::kSPV_KHR_16bit_storage}));
  state_.RegisterExtension(Extension::kSPV_KHR_shader_ballot);
  EXPECT_TRUE(state_.HasAnyOfExtensions({Extension::kSPV_KHR_shader_ballot}));
  EXPECT_FALSE(state_.HasAnyOfExtensions({Extension::kSPV_KHR_16bit_storage}));
  state_.RegisterExtension(Extension::kSPV_KHR_16bit_storage);
  EXPECT_TRUE(state_.HasAnyOfExtensions({Extension::kSPV_KHR_shader_ballot}));
  EXPECT_TRUE(state_.HasAnyOfExtensions({Extension::kSPV_KHR_16bit_storage}));
}

TEST_F(ValidationState_HasAnyOfExtensions, MultiCapMask) {
  const auto set1 = ExtensionSet {
    Extension::kSPV_KHR_multiview,
    Extension::kSPV_KHR_16bit_storage
  };
  const auto set2 = ExtensionSet {
    Extension::kSPV_KHR_shader_draw_parameters,
    Extension::kSPV_NV_stereo_view_rendering,
    Extension::kSPV_KHR_shader_ballot
  };
  EXPECT_FALSE(state_.HasAnyOfExtensions(set1));
  EXPECT_FALSE(state_.HasAnyOfExtensions(set2));
  state_.RegisterExtension(Extension::kSPV_KHR_multiview);
  EXPECT_TRUE(state_.HasAnyOfExtensions(set1));
  EXPECT_FALSE(state_.HasAnyOfExtensions(set2));
}

}
