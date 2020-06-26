// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_replace_linear_algebra_instruction.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationReplaceLinearAlgebraInstructionTest, IsApplicable) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %22 "main"
               OpExecutionMode %22 OriginUpperLeft
               OpSource ESSL 310
               OpName %22 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeFloat 32
          %5 = OpTypeVector %4 2
          %6 = OpTypeVector %4 3
          %7 = OpTypeVector %4 4
          %8 = OpConstant %4 1
          %9 = OpConstant %4 2
         %10 = OpConstant %4 3
         %11 = OpConstant %4 4
         %12 = OpConstant %4 5
         %13 = OpConstant %4 6
         %14 = OpConstant %4 7
         %15 = OpConstant %4 8
         %16 = OpConstantComposite %5 %8 %9
         %17 = OpConstantComposite %5 %10 %11
         %18 = OpConstantComposite %6 %8 %9 %10
         %19 = OpConstantComposite %6 %11 %12 %13
         %20 = OpConstantComposite %7 %8 %9 %10 %11
         %21 = OpConstantComposite %7 %12 %13 %14 %15
         %22 = OpFunction %2 None %3
         %23 = OpLabel
         %24 = OpDot %4 %16 %17
         %25 = OpDot %4 %18 %19
         %26 = OpDot %4 %20 %21
         %27 = OpVectorTimesScalar %5 %16 %8
         %28 = OpVectorTimesScalar %6 %18 %9
         %29 = OpVectorTimesScalar %7 %20 %10
         %30 = OpCopyObject %4 %24
         %31 = OpFAdd %4 %8 %9
         %32 = OpFMul %4 %10 %11
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

  // Tests linear algebra instructions.
  auto instruction_descriptor = MakeInstructionDescriptor(24, SpvOpDot, 0);
  auto transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 35, 36, 37, 38}, instruction_descriptor);
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));

  instruction_descriptor =
      MakeInstructionDescriptor(27, SpvOpVectorTimesScalar, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 35, 36}, instruction_descriptor);
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests non-linear algebra instructions.
  instruction_descriptor = MakeInstructionDescriptor(30, SpvOpCopyObject, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 35, 36, 37, 38}, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  instruction_descriptor = MakeInstructionDescriptor(31, SpvOpFAdd, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 35, 36, 37}, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  instruction_descriptor = MakeInstructionDescriptor(32, SpvOpFMul, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 35, 36}, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests number of fresh ids is different than necessary.
  instruction_descriptor = MakeInstructionDescriptor(25, SpvOpDot, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 35, 36}, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  instruction_descriptor =
      MakeInstructionDescriptor(28, SpvOpVectorTimesScalar, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 35, 36, 37, 38, 39}, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests non-fresh ids.
  instruction_descriptor = MakeInstructionDescriptor(26, SpvOpDot, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 5, 36, 37, 8, 39, 40, 1, 42, 3, 44, 45, 46},
      instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  instruction_descriptor =
      MakeInstructionDescriptor(29, SpvOpVectorTimesScalar, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 35, 36, 7, 38, 9, 40}, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationReplaceLinearAlgebraInstructionTest,
     ReplaceOpVectorTimesScalar) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %15 "main"
               OpExecutionMode %15 OriginUpperLeft
               OpSource ESSL 310
               OpName %15 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeFloat 32
          %5 = OpTypeVector %4 2
          %6 = OpTypeVector %4 3
          %7 = OpTypeVector %4 4
          %8 = OpConstant %4 1
          %9 = OpConstant %4 2
         %10 = OpConstant %4 3
         %11 = OpConstant %4 4
         %12 = OpConstantComposite %5 %8 %9
         %13 = OpConstantComposite %6 %8 %9 %10
         %14 = OpConstantComposite %7 %8 %9 %10 %11
         %15 = OpFunction %2 None %3
         %16 = OpLabel
         %17 = OpVectorTimesScalar %5 %12 %8
         %18 = OpVectorTimesScalar %6 %13 %9
         %19 = OpVectorTimesScalar %7 %14 %10
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto instruction_descriptor =
      MakeInstructionDescriptor(17, SpvOpVectorTimesScalar, 0);
  auto transformation = TransformationReplaceLinearAlgebraInstruction(
      {20, 21, 22, 23}, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(18, SpvOpVectorTimesScalar, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {24, 25, 26, 27, 28, 29}, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(19, SpvOpVectorTimesScalar, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {30, 31, 32, 33, 34, 35, 36, 37}, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %15 "main"
               OpExecutionMode %15 OriginUpperLeft
               OpSource ESSL 310
               OpName %15 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeFloat 32
          %5 = OpTypeVector %4 2
          %6 = OpTypeVector %4 3
          %7 = OpTypeVector %4 4
          %8 = OpConstant %4 1
          %9 = OpConstant %4 2
         %10 = OpConstant %4 3
         %11 = OpConstant %4 4
         %12 = OpConstantComposite %5 %8 %9
         %13 = OpConstantComposite %6 %8 %9 %10
         %14 = OpConstantComposite %7 %8 %9 %10 %11
         %15 = OpFunction %2 None %3
         %16 = OpLabel
         %20 = OpCompositeExtract %4 %12 0
         %21 = OpFMul %4 %20 %8
         %22 = OpCompositeExtract %4 %12 1
         %23 = OpFMul %4 %22 %8
         %17 = OpCompositeConstruct %5 %21 %23
         %24 = OpCompositeExtract %4 %13 0
         %25 = OpFMul %4 %24 %9
         %26 = OpCompositeExtract %4 %13 1
         %27 = OpFMul %4 %26 %9
         %28 = OpCompositeExtract %4 %13 2
         %29 = OpFMul %4 %28 %9
         %18 = OpCompositeConstruct %6 %25 %27 %29
         %30 = OpCompositeExtract %4 %14 0
         %31 = OpFMul %4 %30 %10
         %32 = OpCompositeExtract %4 %14 1
         %33 = OpFMul %4 %32 %10
         %34 = OpCompositeExtract %4 %14 2
         %35 = OpFMul %4 %34 %10
         %36 = OpCompositeExtract %4 %14 3
         %37 = OpFMul %4 %36 %10
         %19 = OpCompositeConstruct %7 %31 %33 %35 %37
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
}

TEST(TransformationReplaceLinearAlgebraInstructionTest,
     ReplaceOpMatrixTimesScalar) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %54 "main"
               OpExecutionMode %54 OriginUpperLeft

; Types
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeFloat 32
          %5 = OpTypeVector %4 2
          %6 = OpTypeVector %4 3
          %7 = OpTypeVector %4 4
          %8 = OpTypeMatrix %5 2
          %9 = OpTypeMatrix %5 3
         %10 = OpTypeMatrix %5 4
         %11 = OpTypeMatrix %6 2
         %12 = OpTypeMatrix %6 3
         %13 = OpTypeMatrix %6 4
         %14 = OpTypeMatrix %7 2
         %15 = OpTypeMatrix %7 3
         %16 = OpTypeMatrix %7 4

; Constant scalars
         %17 = OpConstant %4 1
         %18 = OpConstant %4 2
         %19 = OpConstant %4 3
         %20 = OpConstant %4 4
         %21 = OpConstant %4 5
         %22 = OpConstant %4 6
         %23 = OpConstant %4 7
         %24 = OpConstant %4 8
         %25 = OpConstant %4 9
         %26 = OpConstant %4 10
         %27 = OpConstant %4 11
         %28 = OpConstant %4 12
         %29 = OpConstant %4 13
         %30 = OpConstant %4 14
         %31 = OpConstant %4 15
         %32 = OpConstant %4 16

; Constant vectors
         %33 = OpConstantComposite %5 %17 %18
         %34 = OpConstantComposite %5 %19 %20
         %35 = OpConstantComposite %5 %21 %22
         %36 = OpConstantComposite %5 %23 %24
         %37 = OpConstantComposite %6 %17 %18 %19
         %38 = OpConstantComposite %6 %20 %21 %22
         %39 = OpConstantComposite %6 %23 %24 %25
         %40 = OpConstantComposite %6 %26 %27 %28
         %41 = OpConstantComposite %7 %17 %18 %19 %20
         %42 = OpConstantComposite %7 %21 %22 %23 %24
         %43 = OpConstantComposite %7 %25 %26 %27 %28
         %44 = OpConstantComposite %7 %29 %30 %31 %32

; Constant matrices
         %45 = OpConstantComposite %8 %33 %34
         %46 = OpConstantComposite %9 %33 %34 %35
         %47 = OpConstantComposite %10 %33 %34 %35 %36
         %48 = OpConstantComposite %11 %37 %38
         %49 = OpConstantComposite %12 %37 %38 %39
         %50 = OpConstantComposite %13 %37 %38 %39 %40
         %51 = OpConstantComposite %14 %41 %42
         %52 = OpConstantComposite %15 %41 %42 %43
         %53 = OpConstantComposite %16 %41 %42 %43 %44

; main function
         %54 = OpFunction %2 None %3
         %55 = OpLabel

; Multiplying 2-row matrices by scalar
         %56 = OpMatrixTimesScalar %8 %45 %17
         %57 = OpMatrixTimesScalar %9 %46 %18
         %58 = OpMatrixTimesScalar %10 %47 %19

; Multiplying 3-row matrices by scalar
         %59 = OpMatrixTimesScalar %11 %48 %21
         %60 = OpMatrixTimesScalar %12 %49 %22
         %61 = OpMatrixTimesScalar %13 %50 %23

; Multiplying 4-row matrices by scalar
         %62 = OpMatrixTimesScalar %14 %51 %24
         %63 = OpMatrixTimesScalar %15 %52 %25
         %64 = OpMatrixTimesScalar %16 %53 %26
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto instruction_descriptor =
      MakeInstructionDescriptor(56, SpvOpMatrixTimesScalar, 0);
  auto transformation = TransformationReplaceLinearAlgebraInstruction(
      {65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76}, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(57, SpvOpMatrixTimesScalar, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(58, SpvOpMatrixTimesScalar, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106,
       107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %54 "main"
               OpExecutionMode %54 OriginUpperLeft

; Types
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeFloat 32
          %5 = OpTypeVector %4 2
          %6 = OpTypeVector %4 3
          %7 = OpTypeVector %4 4
          %8 = OpTypeMatrix %5 2
          %9 = OpTypeMatrix %5 3
         %10 = OpTypeMatrix %5 4
         %11 = OpTypeMatrix %6 2
         %12 = OpTypeMatrix %6 3
         %13 = OpTypeMatrix %6 4
         %14 = OpTypeMatrix %7 2
         %15 = OpTypeMatrix %7 3
         %16 = OpTypeMatrix %7 4

; Constant scalars
         %17 = OpConstant %4 1
         %18 = OpConstant %4 2
         %19 = OpConstant %4 3
         %20 = OpConstant %4 4
         %21 = OpConstant %4 5
         %22 = OpConstant %4 6
         %23 = OpConstant %4 7
         %24 = OpConstant %4 8
         %25 = OpConstant %4 9
         %26 = OpConstant %4 10
         %27 = OpConstant %4 11
         %28 = OpConstant %4 12
         %29 = OpConstant %4 13
         %30 = OpConstant %4 14
         %31 = OpConstant %4 15
         %32 = OpConstant %4 16

; Constant vectors
         %33 = OpConstantComposite %5 %17 %18
         %34 = OpConstantComposite %5 %19 %20
         %35 = OpConstantComposite %5 %21 %22
         %36 = OpConstantComposite %5 %23 %24
         %37 = OpConstantComposite %6 %17 %18 %19
         %38 = OpConstantComposite %6 %20 %21 %22
         %39 = OpConstantComposite %6 %23 %24 %25
         %40 = OpConstantComposite %6 %26 %27 %28
         %41 = OpConstantComposite %7 %17 %18 %19 %20
         %42 = OpConstantComposite %7 %21 %22 %23 %24
         %43 = OpConstantComposite %7 %25 %26 %27 %28
         %44 = OpConstantComposite %7 %29 %30 %31 %32

; Constant matrices
         %45 = OpConstantComposite %8 %33 %34
         %46 = OpConstantComposite %9 %33 %34 %35
         %47 = OpConstantComposite %10 %33 %34 %35 %36
         %48 = OpConstantComposite %11 %37 %38
         %49 = OpConstantComposite %12 %37 %38 %39
         %50 = OpConstantComposite %13 %37 %38 %39 %40
         %51 = OpConstantComposite %14 %41 %42
         %52 = OpConstantComposite %15 %41 %42 %43
         %53 = OpConstantComposite %16 %41 %42 %43 %44

; main function
         %54 = OpFunction %2 None %3
         %55 = OpLabel

; Multiplying 2x2 matrix by scalar
         %65 = OpCompositeExtract %5 %45 0
         %66 = OpCompositeExtract %4 %65 0
         %67 = OpFMul %4 %66 %17
         %68 = OpCompositeExtract %4 %65 1
         %69 = OpFMul %4 %68 %17
         %70 = OpCompositeConstruct %5 %67 %69
         %71 = OpCompositeExtract %5 %45 1
         %72 = OpCompositeExtract %4 %71 0
         %73 = OpFMul %4 %72 %17
         %74 = OpCompositeExtract %4 %71 1
         %75 = OpFMul %4 %74 %17
         %76 = OpCompositeConstruct %5 %73 %75
         %56 = OpCompositeConstruct %8 %70 %76

; Multiplying 2x3 matrix by scalar
         %77 = OpCompositeExtract %5 %46 0
         %78 = OpCompositeExtract %4 %77 0
         %79 = OpFMul %4 %78 %18
         %80 = OpCompositeExtract %4 %77 1
         %81 = OpFMul %4 %80 %18
         %82 = OpCompositeConstruct %5 %79 %81
         %83 = OpCompositeExtract %5 %46 1
         %84 = OpCompositeExtract %4 %83 0
         %85 = OpFMul %4 %84 %18
         %86 = OpCompositeExtract %4 %83 1
         %87 = OpFMul %4 %86 %18
         %88 = OpCompositeConstruct %5 %85 %87
         %89 = OpCompositeExtract %5 %46 2
         %90 = OpCompositeExtract %4 %89 0
         %91 = OpFMul %4 %90 %18
         %92 = OpCompositeExtract %4 %89 1
         %93 = OpFMul %4 %92 %18
         %94 = OpCompositeConstruct %5 %91 %93
         %57 = OpCompositeConstruct %9 %82 %88 %94

; Multiplying 2x4 matrix by scalar
         %95 = OpCompositeExtract %5 %47 0
         %96 = OpCompositeExtract %4 %95 0
         %97 = OpFMul %4 %96 %19
         %98 = OpCompositeExtract %4 %95 1
         %99 = OpFMul %4 %98 %19
        %100 = OpCompositeConstruct %5 %97 %99
        %101 = OpCompositeExtract %5 %47 1
        %102 = OpCompositeExtract %4 %101 0
        %103 = OpFMul %4 %102 %19
        %104 = OpCompositeExtract %4 %101 1
        %105 = OpFMul %4 %104 %19
        %106 = OpCompositeConstruct %5 %103 %105
        %107 = OpCompositeExtract %5 %47 2
        %108 = OpCompositeExtract %4 %107 0
        %109 = OpFMul %4 %108 %19
        %110 = OpCompositeExtract %4 %107 1
        %111 = OpFMul %4 %110 %19
        %112 = OpCompositeConstruct %5 %109 %111
        %113 = OpCompositeExtract %5 %47 3
        %114 = OpCompositeExtract %4 %113 0
        %115 = OpFMul %4 %114 %19
        %116 = OpCompositeExtract %4 %113 1
        %117 = OpFMul %4 %116 %19
        %118 = OpCompositeConstruct %5 %115 %117
         %58 = OpCompositeConstruct %10 %100 %106 %112 %118

; Multiplying 3-row matrices by scalar
         %59 = OpMatrixTimesScalar %11 %48 %21
         %60 = OpMatrixTimesScalar %12 %49 %22
         %61 = OpMatrixTimesScalar %13 %50 %23

; Multiplying 4-row matrices by scalar
         %62 = OpMatrixTimesScalar %14 %51 %24
         %63 = OpMatrixTimesScalar %15 %52 %25
         %64 = OpMatrixTimesScalar %16 %53 %26
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
}

TEST(TransformationReplaceLinearAlgebraInstructionTest, ReplaceOpDot) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %22 "main"
               OpExecutionMode %22 OriginUpperLeft
               OpSource ESSL 310
               OpName %22 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeFloat 32
          %5 = OpTypeVector %4 2
          %6 = OpTypeVector %4 3
          %7 = OpTypeVector %4 4
          %8 = OpConstant %4 1
          %9 = OpConstant %4 2
         %10 = OpConstant %4 3
         %11 = OpConstant %4 4
         %12 = OpConstant %4 5
         %13 = OpConstant %4 6
         %14 = OpConstant %4 7
         %15 = OpConstant %4 8
         %16 = OpConstantComposite %5 %8 %9
         %17 = OpConstantComposite %5 %10 %11
         %18 = OpConstantComposite %6 %8 %9 %10
         %19 = OpConstantComposite %6 %11 %12 %13
         %20 = OpConstantComposite %7 %8 %9 %10 %11
         %21 = OpConstantComposite %7 %12 %13 %14 %15
         %22 = OpFunction %2 None %3
         %23 = OpLabel
         %24 = OpDot %4 %16 %17
         %25 = OpDot %4 %18 %19
         %26 = OpDot %4 %20 %21
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto instruction_descriptor = MakeInstructionDescriptor(24, SpvOpDot, 0);
  auto transformation = TransformationReplaceLinearAlgebraInstruction(
      {27, 28, 29, 30, 31, 32}, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor = MakeInstructionDescriptor(25, SpvOpDot, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {33, 34, 35, 36, 37, 38, 39, 40, 41, 42}, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor = MakeInstructionDescriptor(26, SpvOpDot, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %22 "main"
               OpExecutionMode %22 OriginUpperLeft
               OpSource ESSL 310
               OpName %22 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeFloat 32
          %5 = OpTypeVector %4 2
          %6 = OpTypeVector %4 3
          %7 = OpTypeVector %4 4
          %8 = OpConstant %4 1
          %9 = OpConstant %4 2
         %10 = OpConstant %4 3
         %11 = OpConstant %4 4
         %12 = OpConstant %4 5
         %13 = OpConstant %4 6
         %14 = OpConstant %4 7
         %15 = OpConstant %4 8
         %16 = OpConstantComposite %5 %8 %9
         %17 = OpConstantComposite %5 %10 %11
         %18 = OpConstantComposite %6 %8 %9 %10
         %19 = OpConstantComposite %6 %11 %12 %13
         %20 = OpConstantComposite %7 %8 %9 %10 %11
         %21 = OpConstantComposite %7 %12 %13 %14 %15
         %22 = OpFunction %2 None %3
         %23 = OpLabel
         %27 = OpCompositeExtract %4 %16 0
         %28 = OpCompositeExtract %4 %17 0
         %29 = OpFMul %4 %27 %28
         %30 = OpCompositeExtract %4 %16 1
         %31 = OpCompositeExtract %4 %17 1
         %32 = OpFMul %4 %30 %31
         %24 = OpFAdd %4 %29 %32
         %33 = OpCompositeExtract %4 %18 0
         %34 = OpCompositeExtract %4 %19 0
         %35 = OpFMul %4 %33 %34
         %36 = OpCompositeExtract %4 %18 1
         %37 = OpCompositeExtract %4 %19 1
         %38 = OpFMul %4 %36 %37
         %39 = OpCompositeExtract %4 %18 2
         %40 = OpCompositeExtract %4 %19 2
         %41 = OpFMul %4 %39 %40
         %42 = OpFAdd %4 %35 %38
         %25 = OpFAdd %4 %41 %42
         %43 = OpCompositeExtract %4 %20 0
         %44 = OpCompositeExtract %4 %21 0
         %45 = OpFMul %4 %43 %44
         %46 = OpCompositeExtract %4 %20 1
         %47 = OpCompositeExtract %4 %21 1
         %48 = OpFMul %4 %46 %47
         %49 = OpCompositeExtract %4 %20 2
         %50 = OpCompositeExtract %4 %21 2
         %51 = OpFMul %4 %49 %50
         %52 = OpCompositeExtract %4 %20 3
         %53 = OpCompositeExtract %4 %21 3
         %54 = OpFMul %4 %52 %53
         %55 = OpFAdd %4 %45 %48
         %56 = OpFAdd %4 %51 %55
         %26 = OpFAdd %4 %54 %56
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
