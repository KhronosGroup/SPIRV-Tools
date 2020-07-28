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
  auto transformation = TransformationInlineFunction({{22, 44},
                                                      {23, 45},
                                                      {24, 46},
                                                      {25, 47},
                                                      {26, 48},
                                                      {27, 49},
                                                      {28, 50},
                                                      {29, 51},
                                                      {30, 52},
                                                      {31, 53},
                                                      {32, 54},
                                                      {33, 55},
                                                      {34, 56},
                                                      {35, 57},
                                                      {36, 58},
                                                      {37, 59}},
                                                     60);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests false function call instruction.
  transformation = TransformationInlineFunction({{22, 44},
                                                 {23, 45},
                                                 {24, 46},
                                                 {25, 47},
                                                 {26, 48},
                                                 {27, 49},
                                                 {28, 50},
                                                 {29, 51},
                                                 {30, 52},
                                                 {31, 53},
                                                 {32, 54},
                                                 {33, 55},
                                                 {34, 56},
                                                 {35, 57},
                                                 {36, 58},
                                                 {37, 59}},
                                                42);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests applicable transformation.
  transformation = TransformationInlineFunction({{22, 44},
                                                 {23, 45},
                                                 {24, 46},
                                                 {25, 47},
                                                 {26, 48},
                                                 {27, 49},
                                                 {28, 50},
                                                 {29, 51},
                                                 {30, 52},
                                                 {31, 53},
                                                 {32, 54},
                                                 {33, 55},
                                                 {34, 56},
                                                 {35, 57},
                                                 {36, 58},
                                                 {37, 59}},
                                                43);
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

  auto transformation = TransformationInlineFunction({{22, 44},
                                                      {23, 45},
                                                      {24, 46},
                                                      {25, 47},
                                                      {26, 48},
                                                      {27, 49},
                                                      {28, 50},
                                                      {29, 51},
                                                      {30, 52},
                                                      {31, 53},
                                                      {32, 54},
                                                      {33, 55},
                                                      {34, 56},
                                                      {35, 57},
                                                      {36, 58},
                                                      {37, 59},
                                                      {38, 60}},
                                                     43);
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
         %44 = OpLoad %3 %41
         %45 = OpLoad %3 %42
         %46 = OpCompositeExtract %2 %44 0
         %47 = OpCompositeExtract %2 %45 0
         %48 = OpFMul %2 %46 %47
         %49 = OpCompositeExtract %2 %44 1
         %50 = OpCompositeExtract %2 %45 1
         %51 = OpFMul %2 %49 %50
         %52 = OpCompositeExtract %2 %44 2
         %53 = OpCompositeExtract %2 %45 2
         %54 = OpFMul %2 %52 %53
         %55 = OpCompositeExtract %2 %44 3
         %56 = OpCompositeExtract %2 %45 3
         %57 = OpFMul %2 %55 %56
         %58 = OpFAdd %2 %48 %51
         %59 = OpFAdd %2 %54 %58
         %60 = OpFAdd %2 %57 %59
         %43 = OpCopyObject %2 %60
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
