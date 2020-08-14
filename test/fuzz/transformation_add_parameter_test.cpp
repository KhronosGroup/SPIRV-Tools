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

TEST(TransformationAddParameterTest, NonPointerBasicTest) {
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
  ASSERT_FALSE(TransformationAddParameter(4, 60, {{0, 7}}, 61)
                   .IsApplicable(context.get(), transformation_context));

  // There is no function with result id 29.
  ASSERT_FALSE(TransformationAddParameter(60, 60, {{0, 11}}, 61)
                   .IsApplicable(context.get(), transformation_context));

  // Parameter id is not fresh.
  ASSERT_FALSE(TransformationAddParameter(9, 14, {{13, 8}}, 61)
                   .IsApplicable(context.get(), transformation_context));

  // Function type id is not fresh.
  ASSERT_FALSE(TransformationAddParameter(9, 60, {{13, 8}}, 14)
                   .IsApplicable(context.get(), transformation_context));

  // Function type id and parameter type id are equal.
  ASSERT_FALSE(TransformationAddParameter(9, 60, {{13, 8}}, 60)
                   .IsApplicable(context.get(), transformation_context));

  // Parameter's initializer doesn't exist.
  ASSERT_FALSE(TransformationAddParameter(9, 60, {{13, 60}}, 61)
                   .IsApplicable(context.get(), transformation_context));

  // Correct transformations.
  {
    TransformationAddParameter correct(9, 60, {{13, 8}}, 61);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(60));
  }
  {
    TransformationAddParameter correct(17, 62, {{0, 7}}, 63);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(62));
  }
  {
    TransformationAddParameter correct(29, 64, {{0, 31}}, 65);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(64));
  }
  {
    TransformationAddParameter correct(34, 66, {{0, 7}}, 67);
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

TEST(TransformationAddParameterTest, NonPointerNotApplicableTest) {
  // This types handles case of adding a new parameter of a non-pointer type
  // where the transformation is not applicable.
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
               OpName %14 "fun3("
               OpName %24 "f1"
               OpName %27 "f2"
               OpName %30 "i1"
               OpName %31 "i2"
               OpName %32 "param"
               OpName %35 "i3"
               OpName %36 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %10 = OpTypeFunction %8 %9
         %18 = OpConstant %8 2
         %22 = OpTypeFloat 32
         %23 = OpTypePointer Private %22
         %24 = OpVariable %23 Private
         %25 = OpConstant %22 1
         %26 = OpTypePointer Function %22
         %28 = OpConstant %22 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %27 = OpVariable %26 Function
         %30 = OpVariable %9 Function
         %31 = OpVariable %9 Function
         %32 = OpVariable %9 Function
         %35 = OpVariable %9 Function
         %36 = OpVariable %9 Function
               OpStore %24 %25
               OpStore %27 %28
         %29 = OpFunctionCall %2 %6
               OpStore %30 %18
         %33 = OpLoad %8 %30
               OpStore %32 %33
         %34 = OpFunctionCall %8 %12 %32
               OpStore %31 %34
         %37 = OpLoad %8 %31
               OpStore %36 %37
         %38 = OpFunctionCall %8 %12 %36
               OpStore %35 %38
         ; %39 = OpFunctionCall %2 %14
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %8 None %10
         %11 = OpFunctionParameter %9
         %13 = OpLabel
         %17 = OpLoad %8 %11
         %19 = OpIAdd %8 %17 %18
               OpReturnValue %19
               OpFunctionEnd
         %14 = OpFunction %2 None %3
         %15 = OpLabel
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

  // Bad: Id 19 is not available in the caller that has id 25.
  TransformationAddParameter transformation_bad_1(12, 50, {{34, 19}, {38, 19}},
                                                  51);

  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: Id 8 does not have a type.
  TransformationAddParameter transformation_bad_2(12, 50, {{34, 8}, {38, 8}},
                                                  51);

  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: Types of id 25 and id 18 are different.
  TransformationAddParameter transformation_bad_3(12, 50, {{34, 25}, {38, 18}},
                                                  51);
  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));

  // Function with id 14 does not have any callers.
  // Bad: Id 18 is not a vaild type.
  TransformationAddParameter transformation_bad_4(14, 50, {{0, 18}}, 51);
  ASSERT_FALSE(
      transformation_bad_4.IsApplicable(context.get(), transformation_context));

  // Function with id 14 does not have any callers.
  // Bad:  There is no type_id required for the new parameter.
  TransformationAddParameter transformation_bad_5(14, 50, {}, 51);
  ASSERT_FALSE(
      transformation_bad_5.IsApplicable(context.get(), transformation_context));

  // Function with id 14 does not have any callers.
  // Bad:  Id 3 refers to OpTypeVoid, which is not supported.
  TransformationAddParameter transformation_bad_6(14, 50, {{0, 3}}, 51);
  ASSERT_FALSE(
      transformation_bad_6.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddParameterTest, PointerTypeTest) {
  // This types handles case of adding a new parameter of a pointer type.
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
               OpName %14 "fun3("
               OpName %17 "s"
               OpName %24 "s"
               OpName %28 "f1"
               OpName %31 "f2"
               OpName %34 "i1"
               OpName %35 "i2"
               OpName %36 "param"
               OpName %39 "i3"
               OpName %40 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %10 = OpTypeFunction %8 %9
         %20 = OpConstant %8 2
         %25 = OpConstant %8 0
         %26 = OpTypeFloat 32
         %27 = OpTypePointer Private %26
         %28 = OpVariable %27 Private
         %60 = OpTypePointer Output %26
         %61 = OpVariable %60 Output
         %29 = OpConstant %26 1
         %30 = OpTypePointer Function %26
         %32 = OpConstant %26 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %31 = OpVariable %30 Function
         %34 = OpVariable %9 Function
         %35 = OpVariable %9 Function
         %36 = OpVariable %9 Function
         %39 = OpVariable %9 Function
         %40 = OpVariable %9 Function
               OpStore %28 %29
               OpStore %31 %32
         %33 = OpFunctionCall %2 %6
               OpStore %34 %20
         %37 = OpLoad %8 %34
               OpStore %36 %37
         %38 = OpFunctionCall %8 %12 %36
               OpStore %35 %38
         %41 = OpLoad %8 %35
               OpStore %40 %41
         %42 = OpFunctionCall %8 %12 %40
               OpStore %39 %42
         %43 = OpFunctionCall %2 %14
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %8 None %10
         %11 = OpFunctionParameter %9
         %13 = OpLabel
         %17 = OpVariable %9 Function
         %18 = OpLoad %8 %11
               OpStore %17 %18
         %19 = OpLoad %8 %17
         %21 = OpIAdd %8 %19 %20
               OpReturnValue %21
               OpFunctionEnd
         %14 = OpFunction %2 None %3
         %15 = OpLabel
         %24 = OpVariable %9 Function
               OpStore %24 %25
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

  // Bad: Pointer of id 61 has storage class Output, which is not supported.
  TransformationAddParameter transformation_bad_1(12, 50, {{38, 61}, {42, 61}},
                                                  51);

  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Good: Local variable of id 31 is defined in the caller (main).
  TransformationAddParameter transformation_good_1(12, 50, {{38, 31}, {42, 31}},
                                                   51);
  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Good: Local variable of id 34 is defined in the caller (main).
  TransformationAddParameter transformation_good_2(14, 52, {{43, 34}}, 53);
  ASSERT_TRUE(transformation_good_2.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_2.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Good: Local variable of id 39 is defined in the caller (main).
  TransformationAddParameter transformation_good_3(6, 54, {{33, 39}}, 55);
  ASSERT_TRUE(transformation_good_3.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_3.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Good: This adds another pointer parameter to the function of id 6.
  TransformationAddParameter transformation_good_4(6, 56, {{33, 31}}, 57);
  ASSERT_TRUE(transformation_good_4.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_4.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string expected_shader = R"(
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
               OpName %14 "fun3("
               OpName %17 "s"
               OpName %24 "s"
               OpName %28 "f1"
               OpName %31 "f2"
               OpName %34 "i1"
               OpName %35 "i2"
               OpName %36 "param"
               OpName %39 "i3"
               OpName %40 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %20 = OpConstant %8 2
         %25 = OpConstant %8 0
         %26 = OpTypeFloat 32
         %27 = OpTypePointer Private %26
         %28 = OpVariable %27 Private
         %60 = OpTypePointer Output %26
         %61 = OpVariable %60 Output
         %29 = OpConstant %26 1
         %30 = OpTypePointer Function %26
         %32 = OpConstant %26 2
         %10 = OpTypeFunction %8 %9 %30
         %53 = OpTypeFunction %2 %9
         %57 = OpTypeFunction %2 %9 %30
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %31 = OpVariable %30 Function
         %34 = OpVariable %9 Function
         %35 = OpVariable %9 Function
         %36 = OpVariable %9 Function
         %39 = OpVariable %9 Function
         %40 = OpVariable %9 Function
               OpStore %28 %29
               OpStore %31 %32
         %33 = OpFunctionCall %2 %6 %39 %31
               OpStore %34 %20
         %37 = OpLoad %8 %34
               OpStore %36 %37
         %38 = OpFunctionCall %8 %12 %36 %31
               OpStore %35 %38
         %41 = OpLoad %8 %35
               OpStore %40 %41
         %42 = OpFunctionCall %8 %12 %40 %31
               OpStore %39 %42
         %43 = OpFunctionCall %2 %14 %34
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %57
         %54 = OpFunctionParameter %9
         %56 = OpFunctionParameter %30
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %8 None %10
         %11 = OpFunctionParameter %9
         %50 = OpFunctionParameter %30
         %13 = OpLabel
         %17 = OpVariable %9 Function
         %18 = OpLoad %8 %11
               OpStore %17 %18
         %19 = OpLoad %8 %17
         %21 = OpIAdd %8 %19 %20
               OpReturnValue %21
               OpFunctionEnd
         %14 = OpFunction %2 None %53
         %52 = OpFunctionParameter %9
         %15 = OpLabel
         %24 = OpVariable %9 Function
               OpStore %24 %25
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
