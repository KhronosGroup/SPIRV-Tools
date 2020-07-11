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

TEST(TransformationReplaceLinearAlgebraInstructionTest,
     ReplaceOpVectorTimesMatrix) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %54 "main"
               OpExecutionMode %54 OriginUpperLeft
               OpSource ESSL 310
               OpName %54 "main"

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

; Multiplying 2-dimensional vector by 2x2 matrix
         %56 = OpVectorTimesMatrix %5 %33 %45

; Multiplying 2-dimensional vector by 2x3 matrix
         %57 = OpVectorTimesMatrix %6 %34 %46

; Multiplying 2-dimensional vector by 2x4 matrix
         %58 = OpVectorTimesMatrix %7 %35 %47

; Multiplying 3-dimensional vector by 3x2 matrix
         %59 = OpVectorTimesMatrix %5 %37 %48

; Multiplying 3-dimensional vector by 3x3 matrix
         %60 = OpVectorTimesMatrix %6 %38 %49

; Multiplying 3-dimensional vector by 3x4 matrix
         %61 = OpVectorTimesMatrix %7 %39 %50

; Multiplying 4-dimensional vector by 4x2 matrix
         %62 = OpVectorTimesMatrix %5 %41 %51

; Multiplying 4-dimensional vector by 4x3 matrix
         %63 = OpVectorTimesMatrix %6 %42 %52

; Multiplying 4-dimensional vector by 4x4 matrix
         %64 = OpVectorTimesMatrix %7 %43 %53
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
      MakeInstructionDescriptor(56, SpvOpVectorTimesMatrix, 0);
  auto transformation = TransformationReplaceLinearAlgebraInstruction(
      {65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(57, SpvOpVectorTimesMatrix, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
       89, 90, 91, 92, 93, 94, 95, 96, 97, 98},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(58, SpvOpVectorTimesMatrix, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(59, SpvOpVectorTimesMatrix, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
       136, 137, 138, 139, 140, 141, 142, 143, 144, 145},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %54 "main"
               OpExecutionMode %54 OriginUpperLeft
               OpSource ESSL 310
               OpName %54 "main"

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

; Multiplying 2-dimensional vector by 2x2 matrix
         %65 = OpCompositeExtract %4 %33 0
         %66 = OpCompositeExtract %4 %33 1
         %67 = OpCompositeExtract %5 %45 0
         %68 = OpCompositeExtract %4 %67 0
         %69 = OpFMul %4 %65 %68
         %70 = OpCompositeExtract %4 %67 1
         %71 = OpFMul %4 %66 %70
         %72 = OpFAdd %4 %69 %71
         %73 = OpCompositeExtract %5 %45 1
         %74 = OpCompositeExtract %4 %73 0
         %75 = OpFMul %4 %65 %74
         %76 = OpCompositeExtract %4 %73 1
         %77 = OpFMul %4 %66 %76
         %78 = OpFAdd %4 %75 %77
         %56 = OpCompositeConstruct %5 %72 %78

; Multiplying 2-dimensional vector by 2x3 matrix
         %79 = OpCompositeExtract %4 %34 0
         %80 = OpCompositeExtract %4 %34 1
         %81 = OpCompositeExtract %5 %46 0
         %82 = OpCompositeExtract %4 %81 0
         %83 = OpFMul %4 %79 %82
         %84 = OpCompositeExtract %4 %81 1
         %85 = OpFMul %4 %80 %84
         %86 = OpFAdd %4 %83 %85
         %87 = OpCompositeExtract %5 %46 1
         %88 = OpCompositeExtract %4 %87 0
         %89 = OpFMul %4 %79 %88
         %90 = OpCompositeExtract %4 %87 1
         %91 = OpFMul %4 %80 %90
         %92 = OpFAdd %4 %89 %91
         %93 = OpCompositeExtract %5 %46 2
         %94 = OpCompositeExtract %4 %93 0
         %95 = OpFMul %4 %79 %94
         %96 = OpCompositeExtract %4 %93 1
         %97 = OpFMul %4 %80 %96
         %98 = OpFAdd %4 %95 %97
         %57 = OpCompositeConstruct %6 %86 %92 %98

; Multiplying 2-dimensional vector by 2x4 matrix
         %99 = OpCompositeExtract %4 %35 0
        %100 = OpCompositeExtract %4 %35 1
        %101 = OpCompositeExtract %5 %47 0
        %102 = OpCompositeExtract %4 %101 0
        %103 = OpFMul %4 %99 %102
        %104 = OpCompositeExtract %4 %101 1
        %105 = OpFMul %4 %100 %104
        %106 = OpFAdd %4 %103 %105
        %107 = OpCompositeExtract %5 %47 1
        %108 = OpCompositeExtract %4 %107 0
        %109 = OpFMul %4 %99 %108
        %110 = OpCompositeExtract %4 %107 1
        %111 = OpFMul %4 %100 %110
        %112 = OpFAdd %4 %109 %111
        %113 = OpCompositeExtract %5 %47 2
        %114 = OpCompositeExtract %4 %113 0
        %115 = OpFMul %4 %99 %114
        %116 = OpCompositeExtract %4 %113 1
        %117 = OpFMul %4 %100 %116
        %118 = OpFAdd %4 %115 %117
        %119 = OpCompositeExtract %5 %47 3
        %120 = OpCompositeExtract %4 %119 0
        %121 = OpFMul %4 %99 %120
        %122 = OpCompositeExtract %4 %119 1
        %123 = OpFMul %4 %100 %122
        %124 = OpFAdd %4 %121 %123
         %58 = OpCompositeConstruct %7 %106 %112 %118 %124

; Multiplying 3-dimensional vector by 3x2 matrix
        %125 = OpCompositeExtract %4 %37 0
        %126 = OpCompositeExtract %4 %37 1
        %127 = OpCompositeExtract %4 %37 2
        %128 = OpCompositeExtract %6 %48 0
        %129 = OpCompositeExtract %4 %128 0
        %130 = OpFMul %4 %125 %129
        %131 = OpCompositeExtract %4 %128 1
        %132 = OpFMul %4 %126 %131
        %133 = OpCompositeExtract %4 %128 2
        %134 = OpFMul %4 %127 %133
        %135 = OpFAdd %4 %130 %132
        %136 = OpFAdd %4 %134 %135
        %137 = OpCompositeExtract %6 %48 1
        %138 = OpCompositeExtract %4 %137 0
        %139 = OpFMul %4 %125 %138
        %140 = OpCompositeExtract %4 %137 1
        %141 = OpFMul %4 %126 %140
        %142 = OpCompositeExtract %4 %137 2
        %143 = OpFMul %4 %127 %142
        %144 = OpFAdd %4 %139 %141
        %145 = OpFAdd %4 %143 %144
         %59 = OpCompositeConstruct %5 %136 %145

; Multiplying 3-dimensional vector by 3x3 matrix
         %60 = OpVectorTimesMatrix %6 %38 %49

; Multiplying 3-dimensional vector by 3x4 matrix
         %61 = OpVectorTimesMatrix %7 %39 %50

; Multiplying 4-dimensional vector by 4x2 matrix
         %62 = OpVectorTimesMatrix %5 %41 %51

; Multiplying 4-dimensional vector by 4x3 matrix
         %63 = OpVectorTimesMatrix %6 %42 %52

; Multiplying 4-dimensional vector by 4x4 matrix
         %64 = OpVectorTimesMatrix %7 %43 %53
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
}

TEST(TransformationReplaceLinearAlgebraInstructionTest,
     ReplaceOpMatrixTimesVector) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %54 "main"
               OpExecutionMode %54 OriginUpperLeft
               OpSource ESSL 310
               OpName %54 "main"

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

; Multiplying 2x2 matrix by 2-dimensional vector
         %56 = OpMatrixTimesVector %5 %45 %33

; Multiplying 3x2 matrix by 2-dimensional vector
         %57 = OpMatrixTimesVector %6 %48 %34

; Multiplying 4x2 matrix by 2-dimensional vector
         %58 = OpMatrixTimesVector %7 %51 %35

; Multiplying 2x3 matrix by 3-dimensional vector
         %59 = OpMatrixTimesVector %5 %46 %37

; Multiplying 3x3 matrix by 3-dimensional vector
         %60 = OpMatrixTimesVector %6 %49 %38

; Multiplying 4x3 matrix by 3-dimensional vector
         %61 = OpMatrixTimesVector %7 %52 %39

; Multiplying 2x4 matrix by 4-dimensional vector
         %62 = OpMatrixTimesVector %5 %47 %41

; Multiplying 3x4 matrix by 4-dimensional vector
         %63 = OpMatrixTimesVector %6 %50 %42

; Multiplying 4x4 matrix by 4-dimensional vector
         %64 = OpMatrixTimesVector %7 %53 %43
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
      MakeInstructionDescriptor(56, SpvOpMatrixTimesVector, 0);
  auto transformation = TransformationReplaceLinearAlgebraInstruction(
      {65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(57, SpvOpMatrixTimesVector, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
       97},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(58, SpvOpMatrixTimesVector, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor =
      MakeInstructionDescriptor(59, SpvOpMatrixTimesVector, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction(
      {122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
       133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143},
      instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %54 "main"
               OpExecutionMode %54 OriginUpperLeft
               OpSource ESSL 310
               OpName %54 "main"

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

; Multiplying 2x2 matrix by 2-dimensional vector
         %65 = OpCompositeExtract %5 %45 0
         %66 = OpCompositeExtract %5 %45 1
         %67 = OpCompositeExtract %4 %33 0
         %68 = OpCompositeExtract %4 %33 1
         %69 = OpCompositeExtract %4 %65 0
         %70 = OpFMul %4 %69 %67
         %71 = OpCompositeExtract %4 %66 0
         %72 = OpFMul %4 %71 %68
         %73 = OpFAdd %4 %70 %72
         %74 = OpCompositeExtract %4 %65 1
         %75 = OpFMul %4 %74 %67
         %76 = OpCompositeExtract %4 %66 1
         %77 = OpFMul %4 %76 %68
         %78 = OpFAdd %4 %75 %77
         %56 = OpCompositeConstruct %5 %73 %78

; Multiplying 3x2 matrix by 2-dimensional vector
         %79 = OpCompositeExtract %6 %48 0
         %80 = OpCompositeExtract %6 %48 1
         %81 = OpCompositeExtract %4 %34 0
         %82 = OpCompositeExtract %4 %34 1
         %83 = OpCompositeExtract %4 %79 0
         %84 = OpFMul %4 %83 %81
         %85 = OpCompositeExtract %4 %80 0
         %86 = OpFMul %4 %85 %82
         %87 = OpFAdd %4 %84 %86
         %88 = OpCompositeExtract %4 %79 1
         %89 = OpFMul %4 %88 %81
         %90 = OpCompositeExtract %4 %80 1
         %91 = OpFMul %4 %90 %82
         %92 = OpFAdd %4 %89 %91
         %93 = OpCompositeExtract %4 %79 2
         %94 = OpFMul %4 %93 %81
         %95 = OpCompositeExtract %4 %80 2
         %96 = OpFMul %4 %95 %82
         %97 = OpFAdd %4 %94 %96
         %57 = OpCompositeConstruct %6 %87 %92 %97

; Multiplying 4x2 matrix by 2-dimensional vector
         %98 = OpCompositeExtract %7 %51 0
         %99 = OpCompositeExtract %7 %51 1
        %100 = OpCompositeExtract %4 %35 0
        %101 = OpCompositeExtract %4 %35 1
        %102 = OpCompositeExtract %4 %98 0
        %103 = OpFMul %4 %102 %100
        %104 = OpCompositeExtract %4 %99 0
        %105 = OpFMul %4 %104 %101
        %106 = OpFAdd %4 %103 %105
        %107 = OpCompositeExtract %4 %98 1
        %108 = OpFMul %4 %107 %100
        %109 = OpCompositeExtract %4 %99 1
        %110 = OpFMul %4 %109 %101
        %111 = OpFAdd %4 %108 %110
        %112 = OpCompositeExtract %4 %98 2
        %113 = OpFMul %4 %112 %100
        %114 = OpCompositeExtract %4 %99 2
        %115 = OpFMul %4 %114 %101
        %116 = OpFAdd %4 %113 %115
        %117 = OpCompositeExtract %4 %98 3
        %118 = OpFMul %4 %117 %100
        %119 = OpCompositeExtract %4 %99 3
        %120 = OpFMul %4 %119 %101
        %121 = OpFAdd %4 %118 %120
         %58 = OpCompositeConstruct %7 %106 %111 %116 %121

; Multiplying 2x3 matrix by 3-dimensional vector
        %122 = OpCompositeExtract %5 %46 0
        %123 = OpCompositeExtract %5 %46 1
        %124 = OpCompositeExtract %5 %46 2
        %125 = OpCompositeExtract %4 %37 0
        %126 = OpCompositeExtract %4 %37 1
        %127 = OpCompositeExtract %4 %37 2
        %128 = OpCompositeExtract %4 %122 0
        %129 = OpFMul %4 %128 %125
        %130 = OpCompositeExtract %4 %123 0
        %131 = OpFMul %4 %130 %126
        %132 = OpCompositeExtract %4 %124 0
        %133 = OpFMul %4 %132 %127
        %134 = OpFAdd %4 %129 %131
        %135 = OpFAdd %4 %133 %134
        %136 = OpCompositeExtract %4 %122 1
        %137 = OpFMul %4 %136 %125
        %138 = OpCompositeExtract %4 %123 1
        %139 = OpFMul %4 %138 %126
        %140 = OpCompositeExtract %4 %124 1
        %141 = OpFMul %4 %140 %127
        %142 = OpFAdd %4 %137 %139
        %143 = OpFAdd %4 %141 %142
         %59 = OpCompositeConstruct %5 %135 %143

; Multiplying 3x3 matrix by 3-dimensional vector
         %60 = OpMatrixTimesVector %6 %49 %38

; Multiplying 4x3 matrix by 3-dimensional vector
         %61 = OpMatrixTimesVector %7 %52 %39

; Multiplying 2x4 matrix by 4-dimensional vector
         %62 = OpMatrixTimesVector %5 %47 %41

; Multiplying 3x4 matrix by 4-dimensional vector
         %63 = OpMatrixTimesVector %6 %50 %42

; Multiplying 4x4 matrix by 4-dimensional vector
         %64 = OpMatrixTimesVector %7 %53 %43
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
}

TEST(TransformationReplaceLinearAlgebraInstructionTest, ReplaceOpMatrixTimesMatrix) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %54 "main"
               OpExecutionMode %54 OriginUpperLeft
               OpSource ESSL 310
               OpName %54 "main"

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

; Multiplying 2x2 matrix by 2x2 matrix
         %56 = OpMatrixTimesMatrix %8 %45 %45

; Multiplying 2x2 matrix by 2x3 matrix
         %57 = OpMatrixTimesMatrix %9 %45 %46

; Multiplying 2x2 matrix by 2x4 matrix
         %58 = OpMatrixTimesMatrix %10 %45 %47

; Multiplying 2x3 matrix by 3x2 matrix
         %59 = OpMatrixTimesMatrix %8 %46 %48

; Multiplying 2x3 matrix by 3x3 matrix
         %60 = OpMatrixTimesMatrix %9 %46 %49

; Multiplying 2x3 matrix by 3x4 matrix
         %61 = OpMatrixTimesMatrix %10 %46 %50

; Multiplying 2x4 matrix by 4x2 matrix
         %62 = OpMatrixTimesMatrix %8 %47 %51

; Multiplying 2x4 matrix by 4x3 matrix
         %63 = OpMatrixTimesMatrix %9 %47 %52

; Multiplying 2x4 matrix by 4x4 matrix
         %64 = OpMatrixTimesMatrix %10 %47 %53

; Multiplying 3x2 matrix by 2x2 matrix
         %65 = OpMatrixTimesMatrix %11 %48 %45

; Multiplying 3x2 matrix by 2x3 matrix
         %66 = OpMatrixTimesMatrix %12 %48 %46

; Multiplying 3x2 matrix by 2x4 matrix
         %67 = OpMatrixTimesMatrix %13 %48 %47

; Multiplying 3x3 matrix by 3x2 matrix
         %68 = OpMatrixTimesMatrix %11 %49 %48

; Multiplying 3x3 matrix by 3x3 matrix
         %69 = OpMatrixTimesMatrix %12 %49 %49

; Multiplying 3x3 matrix by 3x4 matrix
         %70 = OpMatrixTimesMatrix %13 %49 %50

; Multiplying 3x4 matrix by 4x2 matrix
         %71 = OpMatrixTimesMatrix %11 %50 %51

; Multiplying 3x4 matrix by 4x3 matrix
         %72 = OpMatrixTimesMatrix %12 %50 %52

; Multiplying 3x4 matrix by 4x4 matrix
         %73 = OpMatrixTimesMatrix %13 %50 %53

; Multiplying 4x2 matrix by 2x2 matrix
         %74 = OpMatrixTimesMatrix %14 %51 %45

; Multiplying 4x2 matrix by 2x3 matrix
         %75 = OpMatrixTimesMatrix %15 %51 %46

; Multiplying 4x2 matrix by 2x4 matrix
         %76 = OpMatrixTimesMatrix %16 %51 %47

; Multiplying 4x3 matrix by 3x2 matrix
         %77 = OpMatrixTimesMatrix %14 %52 %48

; Multiplying 4x3 matrix by 3x3 matrix
         %78 = OpMatrixTimesMatrix %15 %52 %49

; Multiplying 4x3 matrix by 3x4 matrix
         %79 = OpMatrixTimesMatrix %16 %52 %50

; Multiplying 4x4 matrix by 4x2 matrix
         %80 = OpMatrixTimesMatrix %14 %53 %51

; Multiplying 4x4 matrix by 4x3 matrix
         %81 = OpMatrixTimesMatrix %15 %53 %52

; Multiplying 4x4 matrix by 4x4 matrix
         %82 = OpMatrixTimesMatrix %16 %53 %53
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager, validator_options);

  auto instruction_descriptor = MakeInstructionDescriptor(56, SpvOpMatrixTimesMatrix, 0);
  auto transformation = TransformationReplaceLinearAlgebraInstruction({83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116}, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor = MakeInstructionDescriptor(57, SpvOpMatrixTimesMatrix, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction({117, 118,  119,  120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166}, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  instruction_descriptor = MakeInstructionDescriptor(58, SpvOpMatrixTimesMatrix, 0);
  transformation = TransformationReplaceLinearAlgebraInstruction({167, 168,  169,  170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232}, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %54 "main"
               OpExecutionMode %54 OriginUpperLeft
               OpSource ESSL 310
               OpName %54 "main"

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

; Multiplying 2x2 matrix by 2x2 matrix
         %83 = OpCompositeExtract %5 %45 0 ; matrix 1 column 0
         %84 = OpCompositeExtract %5 %45 1 ; matrix 1 column 1
         %85 = OpCompositeExtract %5 %45 0 ; matrix 2 column 0
         %86 = OpCompositeExtract %5 %45 1 ; matrix 2 column 1
         %87 = OpCompositeExtract %4 %83 0 ; matrix 1 row 0 column 0
         %88 = OpCompositeExtract %4 %85 0 ; matrix 2 row 0 column 0
         %89 = OpFMul %4 %87 %88
         %90 = OpCompositeExtract %4 %84 0 ; matrix 1 row 0 column 1
         %91 = OpCompositeExtract %4 %85 1 ; matrix 2 row 1 column 0
         %92 = OpFMul %4 %90 %91
         %93 = OpFAdd %4 %89 %92
         %94 = OpCompositeExtract %4 %83 1 ; matrix 1 row 1 column 0
         %95 = OpCompositeExtract %4 %85 0 ; matrix 2 row 0 column 0
         %96 = OpFMul %4 %94 %95
         %97 = OpCompositeExtract %4 %84 1 ; matrix 1 row 1 column 1
         %98 = OpCompositeExtract %4 %85 1 ; matrix 2 row 1 column 0
         %99 = OpFMul %4 %97 %98
        %100 = OpFAdd %4 %96 %99
        %101 = OpCompositeConstruct %5 %93 %100 ; resulting matrix column 0
        %102 = OpCompositeExtract %4 %83 0 ; matrix 1 row 0 column 0
        %103 = OpCompositeExtract %4 %86 0 ; matrix 2 row 0 column 1
        %104 = OpFMul %4 %102 %103
        %105 = OpCompositeExtract %4 %84 0 ; matrix 1 row 0 column 1
        %106 = OpCompositeExtract %4 %86 1 ; matrix 2 row 1 column 1
        %107 = OpFMul %4 %105 %106
        %108 = OpFAdd %4 %104 %107
        %109 = OpCompositeExtract %4 %83 1 ; matrix 1 row 1 column 0
        %110 = OpCompositeExtract %4 %86 0 ; matrix 2 row 0 column 1
        %111 = OpFMul %4 %109 %110
        %112 = OpCompositeExtract %4 %84 1 ; matrix 1 row 1 column 1
        %113 = OpCompositeExtract %4 %86 1 ; matrix 2 row 1 column 1
        %114 = OpFMul %4 %112 %113
        %115 = OpFAdd %4 %111 %114
        %116 = OpCompositeConstruct %5 %108 %115 ; resulting matrix column 1
         %56 = OpCompositeConstruct %8 %101 %116 ; resulting matrix

; Multiplying 2x2 matrix by 2x3 matrix
        %117 = OpCompositeExtract %5 %45 0 ; matrix 1 column 0
        %118 = OpCompositeExtract %5 %45 1 ; matrix 1 column 1
        %119 = OpCompositeExtract %5 %46 0 ; matrix 2 column 0
        %120 = OpCompositeExtract %5 %46 1 ; matrix 2 column 1
        %121 = OpCompositeExtract %5 %46 2 ; matrix 2 column 2
        %122 = OpCompositeExtract %4 %117 0 ; matrix 1 row 0 column 0
        %123 = OpCompositeExtract %4 %119 0 ; matrix 2 row 0 column 0
        %124 = OpFMul %4 %122 %123
        %125 = OpCompositeExtract %4 %118 0 ; matrix 1 row 0 column 1
        %126 = OpCompositeExtract %4 %119 1 ; matrix 2 row 1 column 0
        %127 = OpFMul %4 %125 %126
        %128 = OpFAdd %4 %124 %127
        %129 = OpCompositeExtract %4 %117 1 ; matrix 1 row 1 column 0
        %130 = OpCompositeExtract %4 %119 0 ; matrix 2 row 0 column 0
        %131 = OpFMul %4 %129 %130
        %132 = OpCompositeExtract %4 %118 1 ; matrix 1 row 1 column 1
        %133 = OpCompositeExtract %4 %119 1 ; matrix 2 row 1 column 0
        %134 = OpFMul %4 %132 %133
        %135 = OpFAdd %4 %131 %134
        %136 = OpCompositeConstruct %5 %128 %135 ; resulting matrix column 0
        %137 = OpCompositeExtract %4 %117 0 ; matrix 1 row 0 column 0
        %138 = OpCompositeExtract %4 %120 0 ; matrix 2 row 0 column 1
        %139 = OpFMul %4 %137 %138
        %140 = OpCompositeExtract %4 %118 0 ; matrix 1 row 0 column 1
        %141 = OpCompositeExtract %4 %120 1 ; matrix 2 row 1 column 1
        %142 = OpFMul %4 %140 %141
        %143 = OpFAdd %4 %139 %142
        %144 = OpCompositeExtract %4 %117 1 ; matrix 1 row 1 column 0
        %145 = OpCompositeExtract %4 %120 0 ; matrix 2 row 0 column 1
        %146 = OpFMul %4 %144 %145
        %147 = OpCompositeExtract %4 %118 1 ; matrix 1 row 1 column 1
        %148 = OpCompositeExtract %4 %120 1 ; matrix 2 row 1 column 1
        %149 = OpFMul %4 %147 %148
        %150 = OpFAdd %4 %146 %149
        %151 = OpCompositeConstruct %5 %143 %150 ; resulting matrix column 1
        %152 = OpCompositeExtract %4 %117 0 ; matrix 1 row 0 column 0
        %153 = OpCompositeExtract %4 %121 0 ; matrix 2 row 0 column 2
        %154 = OpFMul %4 %152 %153
        %155 = OpCompositeExtract %4 %118 0 ; matrix 1 row 0 column 1
        %156 = OpCompositeExtract %4 %121 1 ; matrix 2 row 1 column 2
        %157 = OpFMul %4 %155 %156
        %158 = OpFAdd %4 %154 %157
        %159 = OpCompositeExtract %4 %117 1 ; matrix 1 row 1 column 0
        %160 = OpCompositeExtract %4 %121 0 ; matrix 2 row 0 column 2
        %161 = OpFMul %4 %159 %160
        %162 = OpCompositeExtract %4 %118 1 ; matrix 1 row 1 column 1
        %163 = OpCompositeExtract %4 %121 1 ; matrix 2 row 1 column 2
        %164 = OpFMul %4 %162 %163
        %165 = OpFAdd %4 %161 %164
        %166 = OpCompositeConstruct %5 %158 %165 ; resulting matrix column 2
         %57 = OpCompositeConstruct %9 %136 %151 %166 ; resulting matrix

; Multiplying 2x2 matrix by 2x4 matrix
        %167 = OpCompositeExtract %5 %45 0 ; matrix 1 column 0
        %168 = OpCompositeExtract %5 %45 1 ; matrix 1 column 1
        %169 = OpCompositeExtract %5 %47 0 ; matrix 2 column 0
        %170 = OpCompositeExtract %5 %47 1 ; matrix 2 column 1
        %171 = OpCompositeExtract %5 %47 2 ; matrix 2 column 2
        %172 = OpCompositeExtract %5 %47 3 ; matrix 2 column 3
        %173 = OpCompositeExtract %4 %167 0 ; matrix 1 row 0 column 0
        %174 = OpCompositeExtract %4 %169 0 ; matrix 2 row 0 column 0
        %175 = OpFMul %4 %173 %174
        %176 = OpCompositeExtract %4 %168 0 ; matrix 1 row 0 column 1
        %177 = OpCompositeExtract %4 %169 1 ; matrix 2 row 1 column 0
        %178 = OpFMul %4 %176 %177
        %179 = OpFAdd %4 %175 %178
        %180 = OpCompositeExtract %4 %167 1 ; matrix 1 row 1 column 0
        %181 = OpCompositeExtract %4 %169 0 ; matrix 2 row 0 column 0
        %182 = OpFMul %4 %180 %181
        %183 = OpCompositeExtract %4 %168 1 ; matrix 1 row 1 column 1
        %184 = OpCompositeExtract %4 %169 1 ; matrix 2 row 1 column 0
        %185 = OpFMul %4 %183 %184
        %186 = OpFAdd %4 %182 %185
        %187 = OpCompositeConstruct %5 %179 %186 ; resulting matrix column 0
        %188 = OpCompositeExtract %4 %167 0 ; matrix 1 row 0 column 0
        %189 = OpCompositeExtract %4 %170 0 ; matrix 2 row 0 column 1
        %190 = OpFMul %4 %188 %189
        %191 = OpCompositeExtract %4 %168 0 ; matrix 1 row 0 column 1
        %192 = OpCompositeExtract %4 %170 1 ; matrix 2 row 1 column 1
        %193 = OpFMul %4 %191 %192
        %194 = OpFAdd %4 %190 %193
        %195 = OpCompositeExtract %4 %167 1 ; matrix 1 row 1 column 0
        %196 = OpCompositeExtract %4 %170 0 ; matrix 2 row 0 column 1
        %197 = OpFMul %4 %195 %196
        %198 = OpCompositeExtract %4 %168 1 ; matrix 1 row 1 column 1
        %199 = OpCompositeExtract %4 %170 1 ; matrix 2 row 1 column 1
        %200 = OpFMul %4 %198 %199
        %201 = OpFAdd %4 %197 %200
        %202 = OpCompositeConstruct %5 %194 %201 ; resulting matrix column 1
        %203 = OpCompositeExtract %4 %167 0 ; matrix 1 row 0 column 0
        %204 = OpCompositeExtract %4 %171 0 ; matrix 2 row 0 column 2
        %205 = OpFMul %4 %203 %204
        %206 = OpCompositeExtract %4 %168 0 ; matrix 1 row 0 column 1
        %207 = OpCompositeExtract %4 %171 1 ; matrix 2 row 1 column 2
        %208 = OpFMul %4 %206 %207
        %209 = OpFAdd %4 %205 %208
        %210 = OpCompositeExtract %4 %167 1 ; matrix 1 row 1 column 0
        %211 = OpCompositeExtract %4 %171 0 ; matrix 2 row 0 column 2
        %212 = OpFMul %4 %210 %211
        %213 = OpCompositeExtract %4 %168 1 ; matrix 1 row 1 column 1
        %214 = OpCompositeExtract %4 %171 1 ; matrix 2 row 1 column 2
        %215 = OpFMul %4 %213 %214
        %216 = OpFAdd %4 %212 %215
        %217 = OpCompositeConstruct %5 %209 %216 ; resulting matrix column 2
        %218 = OpCompositeExtract %4 %167 0 ; matrix 1 row 0 column 0
        %219 = OpCompositeExtract %4 %172 0 ; matrix 2 row 0 column 3
        %220 = OpFMul %4 %218 %219
        %221 = OpCompositeExtract %4 %168 0 ; matrix 1 row 0 column 1
        %222 = OpCompositeExtract %4 %172 1 ; matrix 2 row 1 column 3
        %223 = OpFMul %4 %221 %222
        %224 = OpFAdd %4 %220 %223
        %225 = OpCompositeExtract %4 %167 1 ; matrix 1 row 1 column 0
        %226 = OpCompositeExtract %4 %172 0 ; matrix 2 row 0 column 3
        %227 = OpFMul %4 %225 %226
        %228 = OpCompositeExtract %4 %168 1 ; matrix 1 row 1 column 1
        %229 = OpCompositeExtract %4 %172 1 ; matrix 2 row 1 column 3
        %230 = OpFMul %4 %228 %229
        %231 = OpFAdd %4 %227 %230
        %232 = OpCompositeConstruct %5 %224 %231 ; resulting matrix column 3
         %58 = OpCompositeConstruct %10 %187 %202 %217 %232 ; resulting matrix

; Multiplying 2x3 matrix by 3x2 matrix
         %59 = OpMatrixTimesMatrix %8 %46 %48

; Multiplying 2x3 matrix by 3x3 matrix
         %60 = OpMatrixTimesMatrix %9 %46 %49

; Multiplying 2x3 matrix by 3x4 matrix
         %61 = OpMatrixTimesMatrix %10 %46 %50

; Multiplying 2x4 matrix by 4x2 matrix
         %62 = OpMatrixTimesMatrix %8 %47 %51

; Multiplying 2x4 matrix by 4x3 matrix
         %63 = OpMatrixTimesMatrix %9 %47 %52

; Multiplying 2x4 matrix by 4x4 matrix
         %64 = OpMatrixTimesMatrix %10 %47 %53

; Multiplying 3x2 matrix by 2x2 matrix
         %65 = OpMatrixTimesMatrix %11 %48 %45

; Multiplying 3x2 matrix by 2x3 matrix
         %66 = OpMatrixTimesMatrix %12 %48 %46

; Multiplying 3x2 matrix by 2x4 matrix
         %67 = OpMatrixTimesMatrix %13 %48 %47

; Multiplying 3x3 matrix by 3x2 matrix
         %68 = OpMatrixTimesMatrix %11 %49 %48

; Multiplying 3x3 matrix by 3x3 matrix
         %69 = OpMatrixTimesMatrix %12 %49 %49

; Multiplying 3x3 matrix by 3x4 matrix
         %70 = OpMatrixTimesMatrix %13 %49 %50

; Multiplying 3x4 matrix by 4x2 matrix
         %71 = OpMatrixTimesMatrix %11 %50 %51

; Multiplying 3x4 matrix by 4x3 matrix
         %72 = OpMatrixTimesMatrix %12 %50 %52

; Multiplying 3x4 matrix by 4x4 matrix
         %73 = OpMatrixTimesMatrix %13 %50 %53

; Multiplying 4x2 matrix by 2x2 matrix
         %74 = OpMatrixTimesMatrix %14 %51 %45

; Multiplying 4x2 matrix by 2x3 matrix
         %75 = OpMatrixTimesMatrix %15 %51 %46

; Multiplying 4x2 matrix by 2x4 matrix
         %76 = OpMatrixTimesMatrix %16 %51 %47

; Multiplying 4x3 matrix by 3x2 matrix
         %77 = OpMatrixTimesMatrix %14 %52 %48

; Multiplying 4x3 matrix by 3x3 matrix
         %78 = OpMatrixTimesMatrix %15 %52 %49

; Multiplying 4x3 matrix by 3x4 matrix
         %79 = OpMatrixTimesMatrix %16 %52 %50

; Multiplying 4x4 matrix by 4x2 matrix
         %80 = OpMatrixTimesMatrix %14 %53 %51

; Multiplying 4x4 matrix by 4x3 matrix
         %81 = OpMatrixTimesMatrix %15 %53 %52

; Multiplying 4x4 matrix by 4x4 matrix
         %82 = OpMatrixTimesMatrix %16 %53 %53
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
