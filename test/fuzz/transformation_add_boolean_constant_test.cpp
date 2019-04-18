// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_add_boolean_constant.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddBooleanConstantTest, NeitherPresentInitiallyAddBoth) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %6 = OpTypeBool
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  CheckValid(env, context.get());

  // True and false can both be added as neither is present.
  ASSERT_TRUE(
      TransformationAddBooleanConstant(7, true).IsApplicable(context.get()));
  ASSERT_TRUE(
      TransformationAddBooleanConstant(7, false).IsApplicable(context.get()));

  // Id 5 is already taken.
  ASSERT_FALSE(
      TransformationAddBooleanConstant(5, true).IsApplicable(context.get()));

  auto add_true = TransformationAddBooleanConstant(7, true);
  auto add_false = TransformationAddBooleanConstant(8, false);

  ASSERT_TRUE(add_true.IsApplicable(context.get()));
  add_true.Apply(context.get());
  CheckValid(env, context.get());

  // Having added true, we cannot add it again.
  ASSERT_FALSE(add_true.IsApplicable(context.get()));

  ASSERT_TRUE(add_false.IsApplicable(context.get()));
  add_false.Apply(context.get());
  CheckValid(env, context.get());

  // Having added false, we cannot add it again.
  ASSERT_FALSE(add_false.IsApplicable(context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %6 = OpTypeBool
          %3 = OpTypeFunction %2
          %7 = OpConstantTrue %6
          %8 = OpConstantFalse %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_transformation, context.get());
}

TEST(TransformationAddBooleanConstantTest, NoOpTypeBoolPresent) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  CheckValid(env, context.get());

  // Neither true nor false can be added as OpTypeBool is not present.
  ASSERT_FALSE(
      TransformationAddBooleanConstant(6, true).IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddBooleanConstant(6, false).IsApplicable(context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
