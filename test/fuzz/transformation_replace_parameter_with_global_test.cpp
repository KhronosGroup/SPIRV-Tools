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
          %2 = OpTypeVoid
          %6 = OpTypeInt 32 1
          %3 = OpTypeFunction %2
          %7 = OpTypePointer Function %6
         %21 = OpTypePointer Private %6
         %22 = OpTypePointer Workgroup %6
          %8 = OpTypeFunction %2 %7 %21 %22 %6
         %15 = OpTypeBool
         %28 = OpConstantFalse %15
         %16 = OpTypePointer Function %15
         %18 = OpConstant %6 3
         %19 = OpConstant %6 4
         %26 = OpVariable %21 Private
         %27 = OpVariable %22 Workgroup
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %17 = OpVariable %16 Function
         %20 = OpVariable %7 Function
         %30 = OpFunctionCall %2 %10 %20 %26 %27 %18
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %23 = OpFunctionParameter %21
         %24 = OpFunctionParameter %22
         %25 = OpFunctionParameter %6
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = [](spv_message_level_t /* level */, const char* /* source */,
                           const spv_position_t& /* position */, const char* message) {
    std::cerr << message << std::endl;
  };
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Parameter id is invalid.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(40, 40, 41, 0)
               .IsApplicable(context.get(), transformation_context));

  // Parameter id is not a result id of an OpFunctionParameter instruction.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(40, 20, 41, 18)
                   .IsApplicable(context.get(), transformation_context));

  // Initializer is non-zero for a pointer parameter with Workgroup storage
  // class.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(40, 24, 41, 18)
                   .IsApplicable(context.get(), transformation_context));

  // Initializer id is 0 for a scalar parameter.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(40, 25, 41, 0)
                   .IsApplicable(context.get(), transformation_context));

  // Initializer id is 0 for a pointer parameter with Function storage class.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(40, 9, 41, 0)
                   .IsApplicable(context.get(), transformation_context));

  // Initializer id is 0 for a pointer parameter with Private storage class.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(40, 23, 41, 0)
                   .IsApplicable(context.get(), transformation_context));

  // Initializer has invalid type.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(40, 25, 41, 28)
                   .IsApplicable(context.get(), transformation_context));

  // Function type id is not fresh.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(28, 25, 40, 18)
                   .IsApplicable(context.get(), transformation_context));

  // Global variable id is not fresh.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(40, 25, 28, 18)
                   .IsApplicable(context.get(), transformation_context));

  // Function type fresh id and global variable fresh id are equal.
  ASSERT_FALSE(TransformationReplaceParameterWithGlobal(40, 25, 40, 18)
                   .IsApplicable(context.get(), transformation_context));

  {
    // Parameter is a pointer with Function storage class.
    TransformationReplaceParameterWithGlobal transformation(40, 9, 41, 18);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    // Parameter is a pointer with Private storage class.
    TransformationReplaceParameterWithGlobal transformation(42, 23, 43, 18);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    // Parameter is a pointer with Workgroup storage class.
    TransformationReplaceParameterWithGlobal transformation(42, 24, 43, 0);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    // Parameter is a scalar type.
    TransformationReplaceParameterWithGlobal transformation(42, 25, 43, 18);
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
          %2 = OpTypeVoid
          %6 = OpTypeInt 32 1
          %3 = OpTypeFunction %2
          %7 = OpTypePointer Function %6
         %21 = OpTypePointer Private %6
         %22 = OpTypePointer Workgroup %6
          %8 = OpTypeFunction %2
         %15 = OpTypeBool
         %28 = OpConstantFalse %15
         %16 = OpTypePointer Function %15
         %18 = OpConstant %6 3
         %19 = OpConstant %6 4
         %26 = OpVariable %21 Private
         %27 = OpVariable %22 Workgroup
         %41 = OpVariable %21 Private %18
         %23 = OpVariable %21 Private %18
         %24 = OpVariable %22 Workgroup
         %43 = OpVariable %21 Private %18
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %17 = OpVariable %16 Function
         %20 = OpVariable %7 Function
               OpCopyMemory %41 %20
               OpCopyMemory %23 %26
               OpCopyMemory %24 %27
               OpStore %43 %18
         %30 = OpFunctionCall %2 %10
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
         %11 = OpLabel
          %9 = OpVariable %7 Function %18
         %25 = OpLoad %6 %43
               OpCopyMemory %9 %41
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationReplaceParameterWithGlobalTest,
     GlobalVariableTypeDoesNotExist) {

}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
