// Copyright (c) 2020 Stefano Milizia
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

#include "source/fuzz/transformation_toggle_constant_null.h"

#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationToggleConstantNullTest, IntConstantTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %17
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %12 "c"
               OpName %17 "color"
               OpDecorate %8 RelaxedPrecision
               OpDecorate %10 RelaxedPrecision
               OpDecorate %12 RelaxedPrecision
               OpDecorate %17 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %11 = OpConstantNull %6
         %13 = OpConstant %6 1
         %14 = OpTypeFloat 32
         %15 = OpTypeVector %14 4
         %16 = OpTypePointer Output %15
         %17 = OpVariable %16 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %13
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // %3 is not a constant declaration
  ASSERT_FALSE(TransformationToggleConstantNull(3).IsApplicable(
      context.get(), transformation_context));

  // The integer constant at %13 is non-zero
  ASSERT_FALSE(TransformationToggleConstantNull(13).IsApplicable(
      context.get(), transformation_context));

  // The transformation can be applied to instruction %9
  ASSERT_TRUE(TransformationToggleConstantNull(9).IsApplicable(
      context.get(), transformation_context));

  // The transformation can be applied to instruction %11
  ASSERT_TRUE(TransformationToggleConstantNull(11).IsApplicable(
      context.get(), transformation_context));
}

TEST(TransformationToggleConstantNullTest, BoolConstantTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %19
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "b"
               OpName %19 "color"
               OpDecorate %19 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpTypePointer Function %6
          %9 = OpConstantFalse %6
         %11 = OpConstantTrue %6
         %16 = OpTypeFloat 32
         %17 = OpTypeVector %16 4
         %18 = OpTypePointer Output %17
         %19 = OpVariable %18 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %12 = OpLogicalEqual %6 %10 %11
               OpSelectionMerge %14 None
               OpBranchConditional %12 %13 %14
         %13 = OpLabel
               OpReturn
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // The boolean constant at %11 is true, but the transformation is only
  // applicable to false boolean constants
  ASSERT_FALSE(TransformationToggleConstantNull(11).IsApplicable(
      context.get(), transformation_context));

  // The transformation can be applied to instruction %9 (OpConstantFalse)
  ASSERT_TRUE(TransformationToggleConstantNull(9).IsApplicable(
      context.get(), transformation_context));
}

TEST(TransformationToggleConstantNullTest, FloatAndOtherConstantTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %22
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %12 "c"
               OpName %22 "color"
               OpDecorate %22 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %11 = OpConstantNull %6
         %13 = OpConstant %6 2
         %16 = OpTypeBool
         %20 = OpTypeVector %6 4
         %21 = OpTypePointer Output %20
         %22 = OpVariable %21 Output
         %23 = OpConstantComposite %20 %9 %11 %9 %11
         %25 = OpConstantComposite %20 %11 %9 %9 %11
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %13
         %14 = OpLoad %6 %8
         %15 = OpLoad %6 %10
         %17 = OpFOrdEqual %16 %14 %15
               OpSelectionMerge %19 None
               OpBranchConditional %17 %18 %24
         %18 = OpLabel
               OpStore %22 %23
               OpBranch %19
         %24 = OpLabel
               OpStore %22 %25
               OpBranch %19
         %19 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // The integer constant at %13 is non-zero
  ASSERT_FALSE(TransformationToggleConstantNull(13).IsApplicable(
      context.get(), transformation_context));

  // The constant at %23 is not scalar nor null
  ASSERT_FALSE(TransformationToggleConstantNull(23).IsApplicable(
      context.get(), transformation_context));

  // The transformation can be applied to instruction %9
  ASSERT_TRUE(TransformationToggleConstantNull(9).IsApplicable(
      context.get(), transformation_context));

  // The transformation can be applied to instruction %11
  ASSERT_TRUE(TransformationToggleConstantNull(11).IsApplicable(
      context.get(), transformation_context));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools