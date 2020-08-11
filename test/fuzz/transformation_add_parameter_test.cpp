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

#include "source/fuzz/transformation_add_parameter.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddParameterTest, BasicTest) {
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
         %16 = OpTypeFloat 32
          %3 = OpTypeFunction %2
          %6 = OpTypeFunction %7 %7
          %8 = OpConstant %11 23
         %12 = OpConstantTrue %7
         %15 = OpTypeFunction %2 %16
         %24 = OpTypeFunction %2 %16 %7
         %31 = OpTypeStruct %7 %11
         %32 = OpConstant %16 23
         %33 = OpConstantComposite %31 %12 %8
         %41 = OpTypeStruct %11 %16
         %42 = OpConstantComposite %41 %8 %32
         %43 = OpTypeFunction %2 %41
         %44 = OpTypeFunction %2 %41 %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %13 = OpFunctionCall %7 %9 %12
               OpReturn
               OpFunctionEnd

          ; adjust type of the function in-place
          %9 = OpFunction %7 None %6
         %14 = OpFunctionParameter %7
         %10 = OpLabel
               OpReturnValue %12
               OpFunctionEnd

         ; reuse an existing function type
         %17 = OpFunction %2 None %15
         %18 = OpFunctionParameter %16
         %19 = OpLabel
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %15
         %21 = OpFunctionParameter %16
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
         %25 = OpFunction %2 None %24
         %26 = OpFunctionParameter %16
         %27 = OpFunctionParameter %7
         %28 = OpLabel
               OpReturn
               OpFunctionEnd

         ; create a new function type
         %29 = OpFunction %2 None %3
         %30 = OpLabel
               OpReturn
               OpFunctionEnd

         ; don't adjust the type of the function if it creates a duplicate
         %34 = OpFunction %2 None %43
         %35 = OpFunctionParameter %41
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
         %37 = OpFunction %2 None %44
         %38 = OpFunctionParameter %41
         %39 = OpFunctionParameter %7
         %40 = OpLabel
               OpReturn
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
  ASSERT_FALSE(TransformationAddParameter(4, 60, 12, 61)
                   .IsApplicable(context.get(), transformation_context));

  // There is no function with result id 29.
  ASSERT_FALSE(TransformationAddParameter(60, 60, 8, 61)
                   .IsApplicable(context.get(), transformation_context));

  // Parameter id is not fresh.
  ASSERT_FALSE(TransformationAddParameter(9, 14, 8, 61)
                   .IsApplicable(context.get(), transformation_context));

  // Function type id is not fresh.
  ASSERT_FALSE(TransformationAddParameter(9, 60, 8, 14)
                   .IsApplicable(context.get(), transformation_context));

  // Function type id and parameter type id are equal.
  ASSERT_FALSE(TransformationAddParameter(9, 60, 8, 60)
                   .IsApplicable(context.get(), transformation_context));

  // Parameter's initializer doesn't exist.
  ASSERT_FALSE(TransformationAddParameter(9, 60, 60, 61)
                   .IsApplicable(context.get(), transformation_context));

  // Correct transformations.
  {
    TransformationAddParameter correct(9, 60, 8, 61);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(60));
  }
  {
    TransformationAddParameter correct(17, 62, 12, 63);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(62));
  }
  {
    TransformationAddParameter correct(29, 64, 33, 65);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(64));
  }
  {
    TransformationAddParameter correct(34, 66, 12, 67);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(66));
  }

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
         %16 = OpTypeFloat 32
          %3 = OpTypeFunction %2
          %8 = OpConstant %11 23
         %12 = OpConstantTrue %7
         %15 = OpTypeFunction %2 %16
         %24 = OpTypeFunction %2 %16 %7
         %31 = OpTypeStruct %7 %11
         %32 = OpConstant %16 23
         %33 = OpConstantComposite %31 %12 %8
         %41 = OpTypeStruct %11 %16
         %42 = OpConstantComposite %41 %8 %32
         %44 = OpTypeFunction %2 %41 %7
          %6 = OpTypeFunction %7 %7 %11
         %65 = OpTypeFunction %2 %31
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %13 = OpFunctionCall %7 %9 %12 %8
               OpReturn
               OpFunctionEnd

          ; adjust type of the function in-place
          %9 = OpFunction %7 None %6
         %14 = OpFunctionParameter %7
         %60 = OpFunctionParameter %11
         %10 = OpLabel
               OpReturnValue %12
               OpFunctionEnd

         ; reuse an existing function type
         %17 = OpFunction %2 None %24
         %18 = OpFunctionParameter %16
         %62 = OpFunctionParameter %7
         %19 = OpLabel
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %15
         %21 = OpFunctionParameter %16
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
         %25 = OpFunction %2 None %24
         %26 = OpFunctionParameter %16
         %27 = OpFunctionParameter %7
         %28 = OpLabel
               OpReturn
               OpFunctionEnd

         ; create a new function type
         %29 = OpFunction %2 None %65
         %64 = OpFunctionParameter %31
         %30 = OpLabel
               OpReturn
               OpFunctionEnd

         ; don't adjust the type of the function if it creates a duplicate
         %34 = OpFunction %2 None %44
         %35 = OpFunctionParameter %41
         %66 = OpFunctionParameter %7
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
         %37 = OpFunction %2 None %44
         %38 = OpFunctionParameter %41
         %39 = OpFunctionParameter %7
         %40 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationAddParameterTest, PointerTypeNotApplicableTest) {
  // This types handles case of adding a new parameter of a pointer type where
  // the transformation is not applicable.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun1("
               OpName %12 "fun2(i1;"
               OpName %11 "a"
               OpName %21 "i1"
               OpName %22 "i2"
               OpName %23 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %40 = OpTypeFloat 32
         %41 = OpTypePointer Function %40
         %42 = OpTypePointer Private %40
         %43 = OpTypePointer Workgroup %40
         %10 = OpTypeFunction %8 %9
         %16 = OpConstant %8 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %21 = OpVariable %9 Function
         %22 = OpVariable %9 Function
         %23 = OpVariable %9 Function
         %20 = OpFunctionCall %2 %6
               OpStore %21 %16
         %24 = OpLoad %8 %21
               OpStore %23 %24
         %25 = OpFunctionCall %8 %12 %23
               OpStore %22 %25
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %8 None %10
         %11 = OpFunctionParameter %9
         %13 = OpLabel
         %15 = OpLoad %8 %11
         %17 = OpIAdd %8 %15 %16
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
  uint32_t function_id;
  uint32_t parameter_fresh_id;
  uint32_t pointer_type_id;
  uint32_t function_type_fresh_id;

  // Bad: There is no local variable in the caller function (main).
  function_id = 12;
  parameter_fresh_id = 50;
  pointer_type_id = 41;
  function_type_fresh_id = 51;

  TransformationAddParameter transformation_bad_1(
      function_id, parameter_fresh_id, pointer_type_id, function_type_fresh_id);

  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: There is no variable of type float and storage class Private.
  function_id = 12;
  parameter_fresh_id = 50;
  pointer_type_id = 42;
  function_type_fresh_id = 51;

  TransformationAddParameter transformation_bad_2(
      function_id, parameter_fresh_id, pointer_type_id, function_type_fresh_id);

  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: There is no variable of type float and storage class Workgroup.
  function_id = 12;
  parameter_fresh_id = 50;
  pointer_type_id = 43;
  function_type_fresh_id = 51;

  TransformationAddParameter transformation_bad_3(
      function_id, parameter_fresh_id, pointer_type_id, function_type_fresh_id);

  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddParameterTest, PointerTypeApplicableTest) {
  // This types handles case of adding a new parameter of a pointer type where
  // the transformation is applied.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun1("
               OpName %12 "fun2(i1;"
               OpName %11 "a"
               OpName %22 "f1"
               OpName %25 "f2"
               OpName %28 "i1"
               OpName %29 "i2"
               OpName %30 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %10 = OpTypeFunction %8 %9
         %16 = OpConstant %8 2
         %20 = OpTypeFloat 32
         %21 = OpTypePointer Private %20
         %22 = OpVariable %21 Private
         %23 = OpConstant %20 1
         %24 = OpTypePointer Function %20
         %26 = OpConstant %20 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %25 = OpVariable %24 Function
         %28 = OpVariable %9 Function
         %29 = OpVariable %9 Function
         %30 = OpVariable %9 Function
               OpStore %22 %23
               OpStore %25 %26
         %27 = OpFunctionCall %2 %6
               OpStore %28 %16
         %31 = OpLoad %8 %28
               OpStore %30 %31
         %32 = OpFunctionCall %8 %12 %30
               OpStore %29 %32
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %8 None %10
         %11 = OpFunctionParameter %9
         %13 = OpLabel
         %15 = OpLoad %8 %11
         %17 = OpIAdd %8 %15 %16
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
  uint32_t function_id;
  uint32_t parameter_fresh_id;
  uint32_t pointer_type_id;
  uint32_t function_type_fresh_id;

  // Good: In every caller of the function there is a local variable.
  function_id = 6;
  parameter_fresh_id = 50;
  pointer_type_id = 24;
  function_type_fresh_id = 51;

  TransformationAddParameter transformation_good_1(
      function_id, parameter_fresh_id, pointer_type_id, function_type_fresh_id);

  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Good: In every caller of the function there is a local variable.
  function_id = 12;
  parameter_fresh_id = 52;
  pointer_type_id = 24;
  function_type_fresh_id = 53;

  TransformationAddParameter transformation_good_2(
      function_id, parameter_fresh_id, pointer_type_id, function_type_fresh_id);

  ASSERT_TRUE(transformation_good_2.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_2.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Good: There is a global variable available.
  function_id = 6;
  parameter_fresh_id = 54;
  pointer_type_id = 21;
  function_type_fresh_id = 55;

  TransformationAddParameter transformation_good_3(
      function_id, parameter_fresh_id, pointer_type_id, function_type_fresh_id);

  ASSERT_TRUE(transformation_good_3.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_3.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Good: There is a global variable available.
  function_id = 12;
  parameter_fresh_id = 56;
  pointer_type_id = 21;
  function_type_fresh_id = 57;

  TransformationAddParameter transformation_good_4(
      function_id, parameter_fresh_id, pointer_type_id, function_type_fresh_id);

  ASSERT_TRUE(transformation_good_4.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_4.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformations = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun1("
               OpName %12 "fun2(i1;"
               OpName %11 "a"
               OpName %22 "f1"
               OpName %25 "f2"
               OpName %28 "i1"
               OpName %29 "i2"
               OpName %30 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %16 = OpConstant %8 2
         %20 = OpTypeFloat 32
         %21 = OpTypePointer Private %20
         %22 = OpVariable %21 Private
         %23 = OpConstant %20 1
         %24 = OpTypePointer Function %20
         %26 = OpConstant %20 2
         %51 = OpTypeFunction %2 %24 %21
         %10 = OpTypeFunction %8 %9 %24 %21
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %25 = OpVariable %24 Function
         %28 = OpVariable %9 Function
         %29 = OpVariable %9 Function
         %30 = OpVariable %9 Function
               OpStore %22 %23
               OpStore %25 %26
         %27 = OpFunctionCall %2 %6 %25 %22
               OpStore %28 %16
         %31 = OpLoad %8 %28
               OpStore %30 %31
         %32 = OpFunctionCall %8 %12 %30 %25 %22
               OpStore %29 %32
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %51
         %50 = OpFunctionParameter %24
         %54 = OpFunctionParameter %21
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %8 None %10
         %11 = OpFunctionParameter %9
         %52 = OpFunctionParameter %24
         %56 = OpFunctionParameter %21
         %13 = OpLabel
         %15 = OpLoad %8 %11
         %17 = OpIAdd %8 %15 %16
               OpReturnValue %17
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformations, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
