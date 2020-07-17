// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_access_chain.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAccessChainTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %48 %54
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 2
         %50 = OpTypeMatrix %7 2
         %70 = OpTypePointer Function %7
         %71 = OpTypePointer Function %50
          %8 = OpTypeStruct %7 %6
          %9 = OpTypePointer Function %8
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %12 = OpTypeFunction %10 %9 %11
         %17 = OpConstant %10 0
         %18 = OpTypeInt 32 0
         %19 = OpConstant %18 0
         %20 = OpTypePointer Function %6
         %99 = OpTypePointer Private %6
         %29 = OpConstant %6 0
         %30 = OpConstant %6 1
         %31 = OpConstantComposite %7 %29 %30
         %32 = OpConstant %6 2
         %33 = OpConstantComposite %8 %31 %32
         %35 = OpConstant %10 10
         %51 = OpConstant %18 10
         %80 = OpConstant %18 0
         %81 = OpConstant %10 1
         %82 = OpConstant %18 2
         %83 = OpConstant %10 3
         %84 = OpConstant %18 4
         %85 = OpConstant %10 5
         %52 = OpTypeArray %50 %51
         %53 = OpTypePointer Private %52
         %45 = OpUndef %9
         %46 = OpConstantNull %9
         %47 = OpTypePointer Private %8
         %48 = OpVariable %47 Private
         %54 = OpVariable %53 Private
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %28 = OpVariable %9 Function
         %34 = OpVariable %11 Function
         %36 = OpVariable %9 Function
         %38 = OpVariable %11 Function
         %44 = OpCopyObject %9 %36
               OpStore %28 %33
               OpStore %34 %35
         %37 = OpLoad %8 %28
               OpStore %36 %37
         %39 = OpLoad %10 %34
               OpStore %38 %39
         %40 = OpFunctionCall %10 %15 %36 %38
         %41 = OpLoad %10 %34
         %42 = OpIAdd %10 %41 %40
               OpStore %34 %42
               OpReturn
               OpFunctionEnd
         %15 = OpFunction %10 None %12
         %13 = OpFunctionParameter %9
         %14 = OpFunctionParameter %11
         %16 = OpLabel
         %21 = OpAccessChain %20 %13 %17 %19
         %43 = OpCopyObject %9 %13
         %22 = OpLoad %6 %21
         %23 = OpConvertFToS %10 %22
         %24 = OpLoad %10 %14
         %25 = OpIAdd %10 %23 %24
               OpReturnValue %25
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Types:
  // Ptr | Pointee | Storage class | GLSL for pointee    | Ids of this type
  // ----+---------+---------------+---------------------+------------------
  //  9  |    8    | Function      | struct(vec2, float) | 28, 36, 44, 13, 43
  // 11  |   10    | Function      | int                 | 34, 38, 14
  // 20  |    6    | Function      | float               | -
  // 99  |    6    | Private       | float               | -
  // 53  |   52    | Private       | mat2x2[10]          | 54
  // 47  |    8    | Private       | struct(vec2, float) | 48
  // 70  |    7    | Function      | vec2                | -
  // 71  |   59    | Function      | mat2x2              | -

  // Indices 0-5 are in ids 80-85

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  transformation_context.GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
      54);

  // Bad: id is not fresh
  ASSERT_FALSE(TransformationAccessChain(
                   43, 43, {80}, MakeInstructionDescriptor(24, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: pointer id does not exist
  ASSERT_FALSE(TransformationAccessChain(
                   100, 1000, {80}, MakeInstructionDescriptor(24, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: pointer id is not a type
  ASSERT_FALSE(TransformationAccessChain(
                   100, 5, {80}, MakeInstructionDescriptor(24, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: pointer id is not a pointer
  ASSERT_FALSE(TransformationAccessChain(
                   100, 23, {80}, MakeInstructionDescriptor(24, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: index id does not exist
  ASSERT_FALSE(TransformationAccessChain(
                   100, 43, {1000}, MakeInstructionDescriptor(24, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: index id is not a constant and the pointer refers to a struct
  ASSERT_FALSE(TransformationAccessChain(
                   100, 43, {24}, MakeInstructionDescriptor(25, SpvOpIAdd, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: too many indices
  ASSERT_FALSE(
      TransformationAccessChain(100, 43, {80, 80, 80},
                                MakeInstructionDescriptor(24, SpvOpLoad, 0))
          .IsApplicable(context.get(), transformation_context));

  // Bad: index id is out of bounds when accessing a struct
  ASSERT_FALSE(
      TransformationAccessChain(100, 43, {83, 80},
                                MakeInstructionDescriptor(24, SpvOpLoad, 0))
          .IsApplicable(context.get(), transformation_context));

  // Bad: attempt to insert before variable
  ASSERT_FALSE(TransformationAccessChain(
                   100, 34, {}, MakeInstructionDescriptor(36, SpvOpVariable, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: pointer not available
  ASSERT_FALSE(
      TransformationAccessChain(
          100, 43, {80}, MakeInstructionDescriptor(21, SpvOpAccessChain, 0))
          .IsApplicable(context.get(), transformation_context));

  // Bad: instruction descriptor does not identify anything
  ASSERT_FALSE(TransformationAccessChain(
                   100, 43, {80}, MakeInstructionDescriptor(24, SpvOpLoad, 100))
                   .IsApplicable(context.get(), transformation_context));

#ifndef NDEBUG
  // Bad: pointer is null
  ASSERT_DEATH(
      TransformationAccessChain(100, 45, {80},
                                MakeInstructionDescriptor(24, SpvOpLoad, 0))
          .IsApplicable(context.get(), transformation_context),
      "Access chains should not be created from null/undefined pointers");
#endif

#ifndef NDEBUG
  // Bad: pointer is undef
  ASSERT_DEATH(
      TransformationAccessChain(100, 46, {80},
                                MakeInstructionDescriptor(24, SpvOpLoad, 0))
          .IsApplicable(context.get(), transformation_context),
      "Access chains should not be created from null/undefined pointers");
#endif

  // Bad: pointer to result type does not exist
  ASSERT_FALSE(TransformationAccessChain(
                   100, 52, {0}, MakeInstructionDescriptor(24, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  {
    TransformationAccessChain transformation(
        100, 43, {80}, MakeInstructionDescriptor(24, SpvOpLoad, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(100));
  }

  {
    TransformationAccessChain transformation(
        101, 28, {81}, MakeInstructionDescriptor(42, SpvOpReturn, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(101));
  }

  {
    TransformationAccessChain transformation(
        102, 36, {80, 81}, MakeInstructionDescriptor(37, SpvOpStore, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(102));
  }

  {
    TransformationAccessChain transformation(
        103, 44, {}, MakeInstructionDescriptor(44, SpvOpStore, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(103));
  }

  {
    TransformationAccessChain transformation(
        104, 13, {80}, MakeInstructionDescriptor(21, SpvOpAccessChain, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(104));
  }

  {
    TransformationAccessChain transformation(
        105, 34, {}, MakeInstructionDescriptor(44, SpvOpStore, 1));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(105));
  }

  {
    TransformationAccessChain transformation(
        106, 38, {}, MakeInstructionDescriptor(40, SpvOpFunctionCall, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(106));
  }

  {
    TransformationAccessChain transformation(
        107, 14, {}, MakeInstructionDescriptor(24, SpvOpLoad, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(107));
  }

  {
    TransformationAccessChain transformation(
        108, 54, {85, 81, 81}, MakeInstructionDescriptor(24, SpvOpLoad, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(108));
  }

  {
    TransformationAccessChain transformation(
        109, 48, {80, 80}, MakeInstructionDescriptor(24, SpvOpLoad, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(
        transformation_context.GetFactManager()->PointeeValueIsIrrelevant(109));
  }

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %48 %54
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 2
         %50 = OpTypeMatrix %7 2
         %70 = OpTypePointer Function %7
         %71 = OpTypePointer Function %50
          %8 = OpTypeStruct %7 %6
          %9 = OpTypePointer Function %8
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %12 = OpTypeFunction %10 %9 %11
         %17 = OpConstant %10 0
         %18 = OpTypeInt 32 0
         %19 = OpConstant %18 0
         %20 = OpTypePointer Function %6
         %99 = OpTypePointer Private %6
         %29 = OpConstant %6 0
         %30 = OpConstant %6 1
         %31 = OpConstantComposite %7 %29 %30
         %32 = OpConstant %6 2
         %33 = OpConstantComposite %8 %31 %32
         %35 = OpConstant %10 10
         %51 = OpConstant %18 10
         %80 = OpConstant %18 0
         %81 = OpConstant %10 1
         %82 = OpConstant %18 2
         %83 = OpConstant %10 3
         %84 = OpConstant %18 4
         %85 = OpConstant %10 5
         %52 = OpTypeArray %50 %51
         %53 = OpTypePointer Private %52
         %45 = OpUndef %9
         %46 = OpConstantNull %9
         %47 = OpTypePointer Private %8
         %48 = OpVariable %47 Private
         %54 = OpVariable %53 Private
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %28 = OpVariable %9 Function
         %34 = OpVariable %11 Function
         %36 = OpVariable %9 Function
         %38 = OpVariable %11 Function
         %44 = OpCopyObject %9 %36
        %103 = OpAccessChain %9 %44
               OpStore %28 %33
        %105 = OpAccessChain %11 %34
               OpStore %34 %35
         %37 = OpLoad %8 %28
        %102 = OpAccessChain %20 %36 %80 %81
               OpStore %36 %37
         %39 = OpLoad %10 %34
               OpStore %38 %39
        %106 = OpAccessChain %11 %38
         %40 = OpFunctionCall %10 %15 %36 %38
         %41 = OpLoad %10 %34
         %42 = OpIAdd %10 %41 %40
               OpStore %34 %42
        %101 = OpAccessChain %20 %28 %81
               OpReturn
               OpFunctionEnd
         %15 = OpFunction %10 None %12
         %13 = OpFunctionParameter %9
         %14 = OpFunctionParameter %11
         %16 = OpLabel
        %104 = OpAccessChain %70 %13 %80
         %21 = OpAccessChain %20 %13 %17 %19
         %43 = OpCopyObject %9 %13
         %22 = OpLoad %6 %21
         %23 = OpConvertFToS %10 %22
        %100 = OpAccessChain %70 %43 %80
        %107 = OpAccessChain %11 %14
        %108 = OpAccessChain %99 %54 %85 %81 %81
        %109 = OpAccessChain %99 %48 %80 %80
         %24 = OpLoad %10 %14
         %25 = OpIAdd %10 %23 %24
               OpReturnValue %25
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationAccessChainTest, IsomorphicStructs) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %11 %12
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeStruct %6
          %8 = OpTypePointer Private %7
          %9 = OpTypeStruct %6
         %10 = OpTypePointer Private %9
         %11 = OpVariable %8 Private
         %12 = OpVariable %10 Private
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  {
    TransformationAccessChain transformation(
        100, 11, {}, MakeInstructionDescriptor(5, SpvOpReturn, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
  }
  {
    TransformationAccessChain transformation(
        101, 12, {}, MakeInstructionDescriptor(5, SpvOpReturn, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
  }

  std::string after_transformation = R"(
; SPIR-V
; Version: 1.0
; Generator: Khronos Glslang Reference Front End; 8
; Bound: 49
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %48
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %9 "x"
               OpName %16 "index"
               OpName %17 "a"
               OpName %25 "A"
               OpMemberName %25 0 "x"
               OpName %27 "str"
               OpName %33 "index2"
               OpName %37 "vars"
               OpName %48 "color"
               OpDecorate %9 RelaxedPrecision
               OpDecorate %16 RelaxedPrecision
               OpDecorate %17 RelaxedPrecision
               OpDecorate %21 RelaxedPrecision
               OpDecorate %22 RelaxedPrecision
               OpDecorate %24 RelaxedPrecision
               OpMemberDecorate %25 0 RelaxedPrecision
               OpDecorate %28 RelaxedPrecision
               OpDecorate %30 RelaxedPrecision
               OpDecorate %32 RelaxedPrecision
               OpDecorate %33 RelaxedPrecision
               OpDecorate %38 RelaxedPrecision
               OpDecorate %41 RelaxedPrecision
               OpDecorate %42 RelaxedPrecision
               OpDecorate %44 RelaxedPrecision
               OpDecorate %48 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeVector %6 4
          %8 = OpTypePointer Function %7
         %10 = OpConstant %6 0
         %11 = OpConstant %6 1
         %12 = OpConstant %6 3
         %13 = OpConstant %6 2
         %14 = OpConstantComposite %7 %10 %11 %12 %13
         %15 = OpTypePointer Function %6
         %18 = OpTypeInt 32 0
         %19 = OpConstant %18 1
         %25 = OpTypeStruct %7
         %26 = OpTypePointer Function %25
         %34 = OpConstant %18 10
         %35 = OpTypeArray %25 %34
         %36 = OpTypePointer Function %35
         %45 = OpTypeFloat 32
         %46 = OpTypeVector %45 4
         %47 = OpTypePointer Output %46
         %48 = OpVariable %47 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %9 = OpVariable %8 Function
         %16 = OpVariable %15 Function
         %17 = OpVariable %15 Function
         %27 = OpVariable %26 Function
         %33 = OpVariable %15 Function
         %37 = OpVariable %36 Function
               OpStore %9 %14
               OpStore %16 %10
         %20 = OpAccessChain %15 %9 %19
         %21 = OpLoad %6 %20
               OpStore %17 %21
         %22 = OpLoad %6 %16
         %23 = OpAccessChain %15 %9 %22
         %24 = OpLoad %6 %23
               OpStore %17 %24
         %28 = OpLoad %7 %9
         %29 = OpCompositeConstruct %25 %28
               OpStore %27 %29
         %30 = OpLoad %6 %16
         %31 = OpAccessChain %15 %27 %10 %30
         %32 = OpLoad %6 %31
               OpStore %17 %32
               OpStore %33 %11
         %38 = OpLoad %7 %9
         %39 = OpCompositeConstruct %25 %38
         %40 = OpAccessChain %26 %37 %11
               OpStore %40 %39
         %41 = OpLoad %6 %16
         %42 = OpLoad %6 %33
         %43 = OpAccessChain %15 %37 %41 %10 %42
         %44 = OpLoad %6 %43
               OpStore %17 %44
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationAccessChainTest, ClampingVariables) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeBool
          %7 = OpTypeInt 32 1
          %8 = OpTypeVector %7 4
          %9 = OpTypePointer Function %8
         %10 = OpConstant %7 0
         %11 = OpConstant %7 1
         %12 = OpConstant %7 3
         %13 = OpConstant %7 2
         %14 = OpConstant %7 10
         %15 = OpConstantComposite %8 %10 %11 %12 %13
         %16 = OpTypePointer Function %7
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 1
         %19 = OpTypeFloat 32
         %20 = OpTypeVector %19 4
         %21 = OpTypePointer Output %20
          %3 = OpVariable %21 Output
          %2 = OpFunction %4 None %5
         %22 = OpLabel
         %23 = OpVariable %9 Function
         %24 = OpVariable %16 Function
         %25 = OpVariable %16 Function
               OpStore %23 %15
               OpStore %24 %10
         %26 = OpLoad %7 %24
         %27 = OpAccessChain %16 %23 %26
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  {
    // The out of bounds constant index is clamped to be in-bound
    TransformationAccessChain transformation(
        100, 23, {14}, MakeInstructionDescriptor(26, SpvOpReturn, 0),
        {200, 201});
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
  }

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeBool
          %7 = OpTypeInt 32 1
          %8 = OpTypeVector %7 4
          %9 = OpTypePointer Function %8
         %10 = OpConstant %7 0
         %11 = OpConstant %7 1
         %12 = OpConstant %7 3
         %13 = OpConstant %7 2
         %14 = OpConstant %7 10
         %15 = OpConstantComposite %8 %10 %11 %12 %13
         %16 = OpTypePointer Function %7
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 1
         %19 = OpTypeFloat 32
         %20 = OpTypeVector %19 4
         %21 = OpTypePointer Output %20
          %3 = OpVariable %21 Output
          %2 = OpFunction %4 None %5
         %22 = OpLabel
         %23 = OpVariable %9 Function
         %24 = OpVariable %16 Function
         %25 = OpVariable %16 Function
               OpStore %23 %15
               OpStore %24 %10
         %26 = OpLoad %7 %24
         %27 = OpAccessChain %16 %23 %26
        %200 = OpULessThanEqual %6 %14 %12
        %201 = OpSelect %7 %200 %14 %12
        %100 = OpAccessChain %16 %23 %201
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
