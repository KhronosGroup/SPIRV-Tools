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

#include "source/fuzz/transformation_add_parameters.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddParametersTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %7 = OpTypeBool
         %11 = OpTypeInt 32 1
          %3 = OpTypeFunction %2
         %20 = OpTypeFunction %2 %7
          %6 = OpTypeFunction %7 %7
         %12 = OpTypeFunction %7 %7 %11
         %13 = OpTypeFunction %7 %7 %7
         %14 = OpTypeFunction %11 %7 %11
         %15 = OpTypeFunction %7 %11 %11
         %16 = OpTypeFunction %7 %7 %11 %11
          %8 = OpConstant %11 23
         %17 = OpConstantTrue %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpFunctionCall %7 %9 %17
               OpReturn
               OpFunctionEnd
          %9 = OpFunction %7 None %6
         %19 = OpFunctionParameter %7
         %10 = OpLabel
               OpReturnValue %17
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Can't modify entry point function.
  ASSERT_FALSE(TransformationAddParameters(4, 6, {20}, {21}, {17})
                   .IsApplicable(context.get(), transformation_context));

  // There is no function with result id 10.
  ASSERT_FALSE(TransformationAddParameters(29, 12, {11}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // There is no OpTypeFunction instruction with result id 21.
  ASSERT_FALSE(TransformationAddParameters(9, 21, {11}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // Function type with id 6 has fewer parameters than required.
  ASSERT_FALSE(TransformationAddParameters(9, 6, {11}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // Function type with id 16 has more parameters than required.
  ASSERT_FALSE(TransformationAddParameters(9, 16, {11}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // New function type is not OpTypeFunction instruction.
  ASSERT_FALSE(TransformationAddParameters(9, 11, {11}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // Return type is invalid.
  ASSERT_FALSE(TransformationAddParameters(9, 14, {11}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // Types of original parameters are invalid.
  ASSERT_FALSE(TransformationAddParameters(9, 15, {11}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // Types of new parameters are invalid.
  ASSERT_FALSE(TransformationAddParameters(9, 13, {11}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // OpTypeVoid can't be the type of function parameter.
  ASSERT_FALSE(TransformationAddParameters(9, 12, {2}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // Vectors, that describe parameters, have different sizes.
  ASSERT_FALSE(TransformationAddParameters(9, 12, {}, {21}, {8})
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationAddParameters(9, 12, {11}, {}, {8})
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationAddParameters(9, 12, {11}, {21}, {})
                   .IsApplicable(context.get(), transformation_context));

  // Vectors cannot be empty.
  ASSERT_FALSE(TransformationAddParameters(9, 12, {}, {}, {})
                   .IsApplicable(context.get(), transformation_context));

  // Parameters' ids are not fresh.
  ASSERT_FALSE(TransformationAddParameters(9, 12, {11}, {20}, {8})
                   .IsApplicable(context.get(), transformation_context));

  // Constants for parameters don't exist.
  ASSERT_FALSE(TransformationAddParameters(9, 12, {11}, {21}, {21})
                   .IsApplicable(context.get(), transformation_context));

  // Constants for parameters have invalid type.
  ASSERT_FALSE(TransformationAddParameters(9, 12, {11}, {21}, {17})
                   .IsApplicable(context.get(), transformation_context));

  // Correct transformation.
  TransformationAddParameters correct(9, 12, {11}, {21}, {8});
  ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
  correct.Apply(context.get(), &transformation_context);

  // The module remains valid.
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %7 = OpTypeBool
         %11 = OpTypeInt 32 1
          %3 = OpTypeFunction %2
         %20 = OpTypeFunction %2 %7
          %6 = OpTypeFunction %7 %7
         %12 = OpTypeFunction %7 %7 %11
         %13 = OpTypeFunction %7 %7 %7
         %14 = OpTypeFunction %11 %7 %11
         %15 = OpTypeFunction %7 %11 %11
         %16 = OpTypeFunction %7 %7 %11 %11
          %8 = OpConstant %11 23
         %17 = OpConstantTrue %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpFunctionCall %7 %9 %17 %8
               OpReturn
               OpFunctionEnd
          %9 = OpFunction %7 None %12
         %19 = OpFunctionParameter %7
         %21 = OpFunctionParameter %11
         %10 = OpLabel
               OpReturnValue %17
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
