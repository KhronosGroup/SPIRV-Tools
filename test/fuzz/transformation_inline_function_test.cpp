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

#include "source/fuzz/transformation_inline_function.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationInlineFunctionTest, IsApplicable) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %39 "main"

; Types
          %2 = OpTypeFloat 32
          %3 = OpTypeVector %2 4
          %4 = OpTypePointer Function %3
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeFunction %2 %4 %4

; Constant scalars
          %8 = OpConstant %2 1
          %9 = OpConstant %2 2
         %10 = OpConstant %2 3
         %11 = OpConstant %2 4
         %12 = OpConstant %2 5
         %13 = OpConstant %2 6
         %14 = OpConstant %2 7
         %15 = OpConstant %2 8

; Constant vectors
         %16 = OpConstantComposite %3 %8 %9 %10 %11
         %17 = OpConstantComposite %3 %12 %13 %14 %15

; dot product function
         %18 = OpFunction %2 None %7
         %19 = OpFunctionParameter %4
         %20 = OpFunctionParameter %4
         %21 = OpLabel
         %22 = OpLoad %3 %19
         %23 = OpLoad %3 %20
         %24 = OpCompositeExtract %2 %22 0
         %25 = OpCompositeExtract %2 %23 0
         %26 = OpFMul %2 %24 %25
         %27 = OpCompositeExtract %2 %22 1
         %28 = OpCompositeExtract %2 %23 1
         %29 = OpFMul %2 %27 %28
         %30 = OpCompositeExtract %2 %22 2
         %31 = OpCompositeExtract %2 %23 2
         %32 = OpFMul %2 %30 %31
         %33 = OpCompositeExtract %2 %22 3
         %34 = OpCompositeExtract %2 %23 3
         %35 = OpFMul %2 %33 %34
         %36 = OpFAdd %2 %26 %29
         %37 = OpFAdd %2 %32 %36
         %38 = OpFAdd %2 %35 %37
               OpReturnValue %38
               OpFunctionEnd

; main function
         %39 = OpFunction %5 None %6
         %40 = OpLabel
         %41 = OpVariable %4 Function
         %42 = OpVariable %4 Function
               OpStore %41 %16
               OpStore %42 %17
         %43 = OpFunctionCall %2 %18 %41 %42 ; dot product function call
               OpBranch %44
         %44 = OpLabel
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

  // Tests undefined function call instruction.
  auto transformation = TransformationInlineFunction(62, {{22, 45},
                                                          {23, 46},
                                                          {24, 47},
                                                          {25, 48},
                                                          {26, 49},
                                                          {27, 50},
                                                          {28, 51},
                                                          {29, 52},
                                                          {30, 53},
                                                          {31, 54},
                                                          {32, 55},
                                                          {33, 56},
                                                          {34, 57},
                                                          {35, 58},
                                                          {36, 59},
                                                          {37, 60},
                                                          {38, 61}});
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests false function call instruction.
  transformation = TransformationInlineFunction(42, {{22, 45},
                                                     {23, 46},
                                                     {24, 47},
                                                     {25, 48},
                                                     {26, 49},
                                                     {27, 50},
                                                     {28, 51},
                                                     {29, 52},
                                                     {30, 53},
                                                     {31, 54},
                                                     {32, 55},
                                                     {33, 56},
                                                     {34, 57},
                                                     {35, 58},
                                                     {36, 59},
                                                     {37, 60},
                                                     {38, 61}});
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests applicable transformation.
  transformation = TransformationInlineFunction(43, {{22, 45},
                                                     {23, 46},
                                                     {24, 47},
                                                     {25, 48},
                                                     {26, 49},
                                                     {27, 50},
                                                     {28, 51},
                                                     {29, 52},
                                                     {30, 53},
                                                     {31, 54},
                                                     {32, 55},
                                                     {33, 56},
                                                     {34, 57},
                                                     {35, 58},
                                                     {36, 59},
                                                     {37, 60},
                                                     {38, 61}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationInlineFunctionTest, Apply) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %39 "main"

; Types
          %2 = OpTypeFloat 32
          %3 = OpTypeVector %2 4
          %4 = OpTypePointer Function %3
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeFunction %2 %4 %4

; Constant scalars
          %8 = OpConstant %2 1
          %9 = OpConstant %2 2
         %10 = OpConstant %2 3
         %11 = OpConstant %2 4
         %12 = OpConstant %2 5
         %13 = OpConstant %2 6
         %14 = OpConstant %2 7
         %15 = OpConstant %2 8

; Constant vectors
         %16 = OpConstantComposite %3 %8 %9 %10 %11
         %17 = OpConstantComposite %3 %12 %13 %14 %15

; dot product function
         %18 = OpFunction %2 None %7
         %19 = OpFunctionParameter %4
         %20 = OpFunctionParameter %4
         %21 = OpLabel
         %22 = OpLoad %3 %19
         %23 = OpLoad %3 %20
         %24 = OpCompositeExtract %2 %22 0
         %25 = OpCompositeExtract %2 %23 0
         %26 = OpFMul %2 %24 %25
         %27 = OpCompositeExtract %2 %22 1
         %28 = OpCompositeExtract %2 %23 1
         %29 = OpFMul %2 %27 %28
         %30 = OpCompositeExtract %2 %22 2
         %31 = OpCompositeExtract %2 %23 2
         %32 = OpFMul %2 %30 %31
         %33 = OpCompositeExtract %2 %22 3
         %34 = OpCompositeExtract %2 %23 3
         %35 = OpFMul %2 %33 %34
         %36 = OpFAdd %2 %26 %29
         %37 = OpFAdd %2 %32 %36
         %38 = OpFAdd %2 %35 %37
               OpReturnValue %38
               OpFunctionEnd

; main function
         %39 = OpFunction %5 None %6
         %40 = OpLabel
         %41 = OpVariable %4 Function
         %42 = OpVariable %4 Function
               OpStore %41 %16
               OpStore %42 %17
         %43 = OpFunctionCall %2 %18 %41 %42 ; dot product function call
               OpBranch %44
         %44 = OpLabel
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

  auto transformation = TransformationInlineFunction(43, {{22, 45},
                                                          {23, 46},
                                                          {24, 47},
                                                          {25, 48},
                                                          {26, 49},
                                                          {27, 50},
                                                          {28, 51},
                                                          {29, 52},
                                                          {30, 53},
                                                          {31, 54},
                                                          {32, 55},
                                                          {33, 56},
                                                          {34, 57},
                                                          {35, 58},
                                                          {36, 59},
                                                          {37, 60},
                                                          {38, 61}});
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %39 "main"

; Types
          %2 = OpTypeFloat 32
          %3 = OpTypeVector %2 4
          %4 = OpTypePointer Function %3
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeFunction %2 %4 %4

; Constant scalars
          %8 = OpConstant %2 1
          %9 = OpConstant %2 2
         %10 = OpConstant %2 3
         %11 = OpConstant %2 4
         %12 = OpConstant %2 5
         %13 = OpConstant %2 6
         %14 = OpConstant %2 7
         %15 = OpConstant %2 8

; Constant vectors
         %16 = OpConstantComposite %3 %8 %9 %10 %11
         %17 = OpConstantComposite %3 %12 %13 %14 %15

; dot product function
         %18 = OpFunction %2 None %7
         %19 = OpFunctionParameter %4
         %20 = OpFunctionParameter %4
         %21 = OpLabel
         %22 = OpLoad %3 %19
         %23 = OpLoad %3 %20
         %24 = OpCompositeExtract %2 %22 0
         %25 = OpCompositeExtract %2 %23 0
         %26 = OpFMul %2 %24 %25
         %27 = OpCompositeExtract %2 %22 1
         %28 = OpCompositeExtract %2 %23 1
         %29 = OpFMul %2 %27 %28
         %30 = OpCompositeExtract %2 %22 2
         %31 = OpCompositeExtract %2 %23 2
         %32 = OpFMul %2 %30 %31
         %33 = OpCompositeExtract %2 %22 3
         %34 = OpCompositeExtract %2 %23 3
         %35 = OpFMul %2 %33 %34
         %36 = OpFAdd %2 %26 %29
         %37 = OpFAdd %2 %32 %36
         %38 = OpFAdd %2 %35 %37
               OpReturnValue %38
               OpFunctionEnd

; main function
         %39 = OpFunction %5 None %6
         %40 = OpLabel
         %41 = OpVariable %4 Function
         %42 = OpVariable %4 Function
               OpStore %41 %16
               OpStore %42 %17
         %45 = OpLoad %3 %41
         %46 = OpLoad %3 %42
         %47 = OpCompositeExtract %2 %45 0
         %48 = OpCompositeExtract %2 %46 0
         %49 = OpFMul %2 %47 %48
         %50 = OpCompositeExtract %2 %45 1
         %51 = OpCompositeExtract %2 %46 1
         %52 = OpFMul %2 %50 %51
         %53 = OpCompositeExtract %2 %45 2
         %54 = OpCompositeExtract %2 %46 2
         %55 = OpFMul %2 %53 %54
         %56 = OpCompositeExtract %2 %45 3
         %57 = OpCompositeExtract %2 %46 3
         %58 = OpFMul %2 %56 %57
         %59 = OpFAdd %2 %49 %52
         %60 = OpFAdd %2 %55 %59
         %61 = OpFAdd %2 %58 %60
         %43 = OpCopyObject %2 %61
               OpBranch %44
         %44 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
}

TEST(TransformationInlineFunctionTest, Misc1) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %15 "main"

; Types
          %2 = OpTypeInt 32 1
          %3 = OpTypeBool
          %4 = OpTypePointer Private %2
          %5 = OpTypePointer Function %2
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeFunction %2 %5
          %9 = OpTypeFunction %2 %2

; Constants
         %10 = OpConstant %2 0
         %11 = OpConstant %2 1
         %12 = OpConstant %2 2
         %13 = OpConstant %2 3

; Global variable
         %14 = OpVariable %4 Private

; main function
         %15 = OpFunction %6 None %7
         %16 = OpLabel
         %17 = OpVariable %5 Function
         %18 = OpVariable %5 Function
         %19 = OpVariable %5 Function
               OpStore %17 %13
         %20 = OpLoad %2 %17
               OpStore %18 %20
         %21 = OpFunctionCall %2 %36 %18
               OpBranch %22
         %22 = OpLabel
         %23 = OpFunctionCall %2 %36 %18
               OpStore %17 %21
         %24 = OpLoad %2 %17
         %25 = OpFunctionCall %2 %54 %24
               OpBranch %26
         %26 = OpLabel
         %27 = OpFunctionCall %2 %54 %24
         %28 = OpLoad %2 %17
         %29 = OpIAdd %2 %28 %25
               OpStore %17 %29
         %30 = OpFunctionCall %6 %67
               OpBranch %31
         %31 = OpLabel
         %32 = OpFunctionCall %6 %67
         %33 = OpLoad %2 %14
         %34 = OpLoad %2 %17
         %35 = OpIAdd %2 %34 %33
               OpStore %17 %35
               OpReturn
               OpFunctionEnd

; Function %36
         %36 = OpFunction %2 None %8
         %37 = OpFunctionParameter %5
         %38 = OpLabel
         %39 = OpVariable %5 Function
         %40 = OpVariable %5 Function
               OpStore %39 %10
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %52 %49 None
               OpBranch %42
         %42 = OpLabel
         %43 = OpLoad %2 %39
         %44 = OpLoad %2 %37
         %45 = OpSLessThan %3 %43 %44
               OpBranchConditional %45 %46 %52
         %46 = OpLabel
         %47 = OpLoad %2 %40
         %48 = OpIAdd %2 %47 %11
               OpStore %40 %48
               OpBranch %49
         %49 = OpLabel
         %50 = OpLoad %2 %39
         %51 = OpIAdd %2 %50 %12
               OpStore %39 %51
               OpBranch %41
         %52 = OpLabel
         %53 = OpLoad %2 %40
               OpReturnValue %53
               OpFunctionEnd

; Function %54
         %54 = OpFunction %2 None %9
         %55 = OpFunctionParameter %2
         %56 = OpLabel
         %57 = OpVariable %5 Function
               OpStore %57 %10
         %58 = OpSGreaterThan %3 %55 %10
               OpSelectionMerge %62 None
               OpBranchConditional %58 %64 %59
         %59 = OpLabel
         %60 = OpLoad %2 %57
         %61 = OpISub %2 %60 %12
               OpStore %57 %61
               OpBranch %62
         %62 = OpLabel
         %63 = OpLoad %2 %57
               OpReturnValue %63
         %64 = OpLabel
         %65 = OpLoad %2 %57
         %66 = OpIAdd %2 %65 %11
               OpStore %57 %66
               OpBranch %62
               OpFunctionEnd

; Function %67
         %67 = OpFunction %6 None %7
         %68 = OpLabel
               OpStore %14 %12
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto transformation = TransformationInlineFunction(30, {});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);

  // Tests a parameter included in the id map.
  transformation = TransformationInlineFunction(25, {{55, 69},
                                                     {56, 70},
                                                     {57, 71},
                                                     {58, 72},
                                                     {59, 73},
                                                     {60, 74},
                                                     {61, 75},
                                                     {62, 76},
                                                     {63, 77},
                                                     {64, 78},
                                                     {65, 79},
                                                     {66, 80}});
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests the id of the returned value not included in the id map.
  transformation = TransformationInlineFunction(25, {{56, 69},
                                                     {57, 70},
                                                     {58, 71},
                                                     {59, 72},
                                                     {60, 73},
                                                     {61, 74},
                                                     {62, 75},
                                                     {64, 76},
                                                     {65, 77},
                                                     {66, 78}});
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  transformation = TransformationInlineFunction(25, {{57, 69},
                                                     {58, 70},
                                                     {59, 71},
                                                     {60, 72},
                                                     {61, 73},
                                                     {62, 74},
                                                     {63, 75},
                                                     {64, 76},
                                                     {65, 77},
                                                     {66, 78}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);

  transformation = TransformationInlineFunction(21, {{39, 79},
                                                     {40, 80},
                                                     {41, 81},
                                                     {42, 82},
                                                     {43, 83},
                                                     {44, 84},
                                                     {45, 85},
                                                     {46, 86},
                                                     {47, 87},
                                                     {48, 88},
                                                     {49, 89},
                                                     {50, 90},
                                                     {51, 91},
                                                     {52, 92},
                                                     {53, 93}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %15 "main"

; Types
          %2 = OpTypeInt 32 1
          %3 = OpTypeBool
          %4 = OpTypePointer Private %2
          %5 = OpTypePointer Function %2
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeFunction %2 %5
          %9 = OpTypeFunction %2 %2

; Constants
         %10 = OpConstant %2 0
         %11 = OpConstant %2 1
         %12 = OpConstant %2 2
         %13 = OpConstant %2 3

; Global variable
         %14 = OpVariable %4 Private

; main function
         %15 = OpFunction %6 None %7
         %16 = OpLabel
         %80 = OpVariable %5 Function
         %79 = OpVariable %5 Function
         %69 = OpVariable %5 Function
         %17 = OpVariable %5 Function
         %18 = OpVariable %5 Function
         %19 = OpVariable %5 Function
               OpStore %17 %13
         %20 = OpLoad %2 %17
               OpStore %18 %20
               OpStore %79 %10
               OpBranch %81
         %81 = OpLabel
               OpLoopMerge %92 %89 None
               OpBranch %82
         %82 = OpLabel
         %83 = OpLoad %2 %79
         %84 = OpLoad %2 %18
         %85 = OpSLessThan %3 %83 %84
               OpBranchConditional %85 %86 %92
         %86 = OpLabel
         %87 = OpLoad %2 %80
         %88 = OpIAdd %2 %87 %11
               OpStore %80 %88
               OpBranch %89
         %89 = OpLabel
         %90 = OpLoad %2 %79
         %91 = OpIAdd %2 %90 %12
               OpStore %79 %91
               OpBranch %81
         %92 = OpLabel
         %93 = OpLoad %2 %80
         %21 = OpCopyObject %2 %93
               OpBranch %22
         %22 = OpLabel
         %23 = OpFunctionCall %2 %36 %18
               OpStore %17 %21
         %24 = OpLoad %2 %17
               OpStore %69 %10
         %70 = OpSGreaterThan %3 %24 %10
               OpSelectionMerge %74 None
               OpBranchConditional %70 %76 %71
         %71 = OpLabel
         %72 = OpLoad %2 %69
         %73 = OpISub %2 %72 %12
               OpStore %69 %73
               OpBranch %74
         %74 = OpLabel
         %75 = OpLoad %2 %69
         %25 = OpCopyObject %2 %75
               OpBranch %26
         %76 = OpLabel
         %77 = OpLoad %2 %69
         %78 = OpIAdd %2 %77 %11
               OpStore %69 %78
               OpBranch %74
         %26 = OpLabel
         %27 = OpFunctionCall %2 %54 %24
         %28 = OpLoad %2 %17
         %29 = OpIAdd %2 %28 %25
               OpStore %17 %29
               OpStore %14 %12
               OpBranch %31
         %31 = OpLabel
         %32 = OpFunctionCall %6 %67
         %33 = OpLoad %2 %14
         %34 = OpLoad %2 %17
         %35 = OpIAdd %2 %34 %33
               OpStore %17 %35
               OpReturn
               OpFunctionEnd

; Function %36
         %36 = OpFunction %2 None %8
         %37 = OpFunctionParameter %5
         %38 = OpLabel
         %39 = OpVariable %5 Function
         %40 = OpVariable %5 Function
               OpStore %39 %10
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %52 %49 None
               OpBranch %42
         %42 = OpLabel
         %43 = OpLoad %2 %39
         %44 = OpLoad %2 %37
         %45 = OpSLessThan %3 %43 %44
               OpBranchConditional %45 %46 %52
         %46 = OpLabel
         %47 = OpLoad %2 %40
         %48 = OpIAdd %2 %47 %11
               OpStore %40 %48
               OpBranch %49
         %49 = OpLabel
         %50 = OpLoad %2 %39
         %51 = OpIAdd %2 %50 %12
               OpStore %39 %51
               OpBranch %41
         %52 = OpLabel
         %53 = OpLoad %2 %40
               OpReturnValue %53
               OpFunctionEnd

; Function %54
         %54 = OpFunction %2 None %9
         %55 = OpFunctionParameter %2
         %56 = OpLabel
         %57 = OpVariable %5 Function
               OpStore %57 %10
         %58 = OpSGreaterThan %3 %55 %10
               OpSelectionMerge %62 None
               OpBranchConditional %58 %64 %59
         %59 = OpLabel
         %60 = OpLoad %2 %57
         %61 = OpISub %2 %60 %12
               OpStore %57 %61
               OpBranch %62
         %62 = OpLabel
         %63 = OpLoad %2 %57
               OpReturnValue %63
         %64 = OpLabel
         %65 = OpLoad %2 %57
         %66 = OpIAdd %2 %65 %11
               OpStore %57 %66
               OpBranch %62
               OpFunctionEnd

; Function %67
         %67 = OpFunction %6 None %7
         %68 = OpLabel
               OpStore %14 %12
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
