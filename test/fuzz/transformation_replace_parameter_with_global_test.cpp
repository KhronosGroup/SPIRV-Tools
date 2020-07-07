// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_replace_parameter_with_global.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationReplaceParameterWithGlobalTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %13 0 RelaxedPrecision
               OpDecorate %16 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Private %6
          %8 = OpTypeFloat 32
          %9 = OpTypePointer Private %8
         %10 = OpTypeVector %8 2
         %11 = OpTypePointer Private %10
         %12 = OpTypeBool
         %40 = OpTypePointer Function %12
         %13 = OpTypeStruct %6 %8
         %14 = OpTypePointer Private %13
         %15 = OpTypeFunction %2 %6 %8 %10 %13 %40 %12
         %22 = OpConstant %6 0
         %23 = OpConstant %8 0
         %26 = OpConstantComposite %10 %23 %23
         %27 = OpConstantTrue %12
         %28 = OpConstantComposite %13 %22 %23
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %41 = OpVariable %40 Function %27
         %33 = OpFunctionCall %2 %20 %22 %23 %26 %28 %41 %27
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %15
         %16 = OpFunctionParameter %6
         %17 = OpFunctionParameter %8
         %18 = OpFunctionParameter %10
         %19 = OpFunctionParameter %13
         %42 = OpFunctionParameter %40
         %43 = OpFunctionParameter %12
         %21 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer =
      [](spv_message_level_t /* level */, const char* /* source */,
         const spv_position_t& /* position */,
         const char* message) { std::cerr << message << std::endl; };
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Parameter id is invalid.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(50, 50, 51)
                   .IsApplicable(context.get(), transformation_context));

  // Parameter id is not a result id of an OpFunctionParameter instruction.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(50, 21, 51)
                   .IsApplicable(context.get(), transformation_context));

  // Parameter has unsupported type.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(50, 42, 51)
                   .IsApplicable(context.get(), transformation_context));

  // Initializer for a global variable doesn't exist in the module.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(50, 43, 51)
                   .IsApplicable(context.get(), transformation_context));

  // Pointer type for a global variable doesn't exist in the module.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(50, 43, 51)
                   .IsApplicable(context.get(), transformation_context));

  // Function type id is not fresh.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(16, 16, 51)
                   .IsApplicable(context.get(), transformation_context));

  // Global variable id is not fresh.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(50, 16, 16)
                   .IsApplicable(context.get(), transformation_context));

  // Function type fresh id and global variable fresh id are equal.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(50, 16, 50)
                   .IsApplicable(context.get(), transformation_context));

  {
    TransformationReplaceParameterWithGlobal transformation(50, 16, 51);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    TransformationReplaceParameterWithGlobal transformation(52, 17, 53);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    TransformationReplaceParameterWithGlobal transformation(54, 18, 55);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    TransformationReplaceParameterWithGlobal transformation(56, 19, 57);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }

  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %13 0 RelaxedPrecision
               OpDecorate %16 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Private %6
          %8 = OpTypeFloat 32
          %9 = OpTypePointer Private %8
         %10 = OpTypeVector %8 2
         %11 = OpTypePointer Private %10
         %12 = OpTypeBool
         %40 = OpTypePointer Function %12
         %13 = OpTypeStruct %6 %8
         %14 = OpTypePointer Private %13
         %15 = OpTypeFunction %2 %40 %12
         %22 = OpConstant %6 0
         %23 = OpConstant %8 0
         %26 = OpConstantComposite %10 %23 %23
         %27 = OpConstantTrue %12
         %28 = OpConstantComposite %13 %22 %23
         %51 = OpVariable %7 Private %22
         %53 = OpVariable %9 Private %23
         %55 = OpVariable %11 Private %26
         %57 = OpVariable %14 Private %28
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %41 = OpVariable %40 Function %27
               OpStore %51 %22
               OpStore %53 %23
               OpStore %55 %26
               OpStore %57 %28
         %33 = OpFunctionCall %2 %20 %41 %27
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %15
         %42 = OpFunctionParameter %40
         %43 = OpFunctionParameter %12
         %21 = OpLabel
         %19 = OpLoad %13 %57
         %18 = OpLoad %10 %55
         %17 = OpLoad %8 %53
         %16 = OpLoad %6 %51
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
