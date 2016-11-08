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
#include "val/construct.h"
#include "val/function.h"
#include "val/validation_state.h"
#include "validate.h"

namespace {
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
}
