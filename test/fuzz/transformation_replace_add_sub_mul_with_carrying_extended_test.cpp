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

#include "source/fuzz/transformation_replace_add_sub_mul_with_carrying_extended.h"

#include <source/fuzz/fuzzer_util.h>

#include "source/fuzz/fuzzer_pass.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationReplaceAddSubMulWithCarryingExtendedTest, BasicScenarios) {
  // This is a simple transformation and this test handles the main cases.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "i1"
               OpName %10 "i2"
               OpName %12 "i3"
               OpName %21 "u1"
               OpName %23 "u2"
               OpName %25 "u3"
               OpName %34 "v1"
               OpName %39 "v2"
               OpName %44 "v3_i"
               OpName %48 "v3_u"
               OpName %52 "uint2"
               OpMemberName %52 0 "a"
               OpMemberName %52 1 "b"
               OpName %54 "result_uint"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 3
         %19 = OpTypeInt 32 0
         %20 = OpTypePointer Function %19
         %22 = OpConstant %19 0
         %24 = OpConstant %19 1
         %32 = OpTypeVector %6 3
         %33 = OpTypePointer Function %32
         %35 = OpConstant %6 1
         %36 = OpConstantComposite %32 %35 %9 %11
         %37 = OpTypeVector %19 3
         %38 = OpTypePointer Function %37
         %40 = OpConstant %19 4
         %41 = OpConstant %19 5
         %42 = OpConstant %19 6
         %43 = OpConstantComposite %37 %40 %41 %42
         %52 = OpTypeStruct %19 %19
         %53 = OpTypePointer Private %52
         %54 = OpVariable %53 Private
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
         %21 = OpVariable %20 Function
         %23 = OpVariable %20 Function
         %25 = OpVariable %20 Function
         %34 = OpVariable %33 Function
         %39 = OpVariable %38 Function
         %44 = OpVariable %33 Function
         %48 = OpVariable %38 Function
               OpStore %8 %9
               OpStore %10 %11
         %13 = OpLoad %6 %10
         %14 = OpLoad %6 %8
         %15 = OpSDiv %6 %13 %14
               OpStore %12 %15
         %16 = OpLoad %6 %10
         %17 = OpLoad %6 %8
         %18 = OpIAdd %6 %16 %17
               OpStore %12 %18
               OpStore %21 %22
               OpStore %23 %24
         %26 = OpLoad %19 %21
         %27 = OpLoad %19 %23
         %28 = OpIMul %19 %26 %27
               OpStore %25 %28
         %29 = OpLoad %6 %10
         %30 = OpLoad %6 %8
         %31 = OpIMul %6 %29 %30
               OpStore %12 %31
               OpStore %34 %36
               OpStore %39 %43
         %45 = OpLoad %32 %34
         %46 = OpLoad %32 %34
         %47 = OpIAdd %32 %45 %46
               OpStore %44 %47
         %49 = OpLoad %37 %39
         %50 = OpLoad %37 %39
         %51 = OpIMul %37 %49 %50
               OpStore %48 %51
         %70 = OpIAdd %19 %16 %26
         %71 = OpISub %6 %26 %27
               OpReturn
               OpFunctionEnd

    )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Bad: |struct_fresh_id| must be fresh.
  auto transformation_bad_1 =
      TransformationReplaceAddSubMulWithCarryingExtended(34, 28);
  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: The transformation cannot be applied to an instruction OpSDiv.
  auto transformation_bad_2 =
      TransformationReplaceAddSubMulWithCarryingExtended(80, 15);
  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: The transformation cannot be applied to an instruction OpIAdd that has
  // signed variables as operands.
  auto transformation_bad_3 =
      TransformationReplaceAddSubMulWithCarryingExtended(80, 18);
  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));

  // Bad: The transformation cannot be applied to an instruction OpIAdd that has
  // different signedness of the types of operands.
  auto transformation_bad_4 =
      TransformationReplaceAddSubMulWithCarryingExtended(80, 70);
  ASSERT_FALSE(
      transformation_bad_4.IsApplicable(context.get(), transformation_context));

  // Bad: The transformation cannot be applied to an instruction OpISub that has
  // different signedness of the result type than the signedness of the types of
  // the operands.
  auto transformation_bad_5 =
      TransformationReplaceAddSubMulWithCarryingExtended(80, 71);
  ASSERT_FALSE(
      transformation_bad_5.IsApplicable(context.get(), transformation_context));

  // Bad: The instruction with result id 100 doesn't exist.
  auto transformation_bad_6 =
      TransformationReplaceAddSubMulWithCarryingExtended(80, 100);
  ASSERT_FALSE(
      transformation_bad_6.IsApplicable(context.get(), transformation_context));

  // Bad: The transformation cannot be applied to the instruction OpIAdd of two
  // vectors that have signed components.

  // Explicitly create the required struct type for the operation on two vectors
  // with unsigned integer components.
  std::vector<uint32_t> operand_type_ids = {37, 37};
  fuzzerutil::AddStructType(context.get(), 81, operand_type_ids);
  context.get()->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
  auto transformation_bad_7 =
      TransformationReplaceAddSubMulWithCarryingExtended(82, 47);
  ASSERT_FALSE(
      transformation_bad_7.IsApplicable(context.get(), transformation_context));

  // Transformations which are applicable:
  auto transformation_good_1 =
      TransformationReplaceAddSubMulWithCarryingExtended(83, 28);
  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));

  transformation_good_1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Explicitly create the required struct type for the operation on two signed
  // integers.
  operand_type_ids = {6, 6};
  fuzzerutil::AddStructType(context.get(), 84, operand_type_ids);
  context.get()->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
  auto transformation_good_2 =
      TransformationReplaceAddSubMulWithCarryingExtended(85, 31);
  ASSERT_TRUE(transformation_good_2.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_2.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  auto transformation_good_3 =
      TransformationReplaceAddSubMulWithCarryingExtended(86, 51);
  ASSERT_TRUE(transformation_good_3.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_3.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformations = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "i1"
               OpName %10 "i2"
               OpName %12 "i3"
               OpName %21 "u1"
               OpName %23 "u2"
               OpName %25 "u3"
               OpName %34 "v1"
               OpName %39 "v2"
               OpName %44 "v3_i"
               OpName %48 "v3_u"
               OpName %52 "uint2"
               OpMemberName %52 0 "a"
               OpMemberName %52 1 "b"
               OpName %54 "result_uint"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 3
         %19 = OpTypeInt 32 0
         %20 = OpTypePointer Function %19
         %22 = OpConstant %19 0
         %24 = OpConstant %19 1
         %32 = OpTypeVector %6 3
         %33 = OpTypePointer Function %32
         %35 = OpConstant %6 1
         %36 = OpConstantComposite %32 %35 %9 %11
         %37 = OpTypeVector %19 3
         %38 = OpTypePointer Function %37
         %40 = OpConstant %19 4
         %41 = OpConstant %19 5
         %42 = OpConstant %19 6
         %43 = OpConstantComposite %37 %40 %41 %42
         %52 = OpTypeStruct %19 %19
         %53 = OpTypePointer Private %52
         %54 = OpVariable %53 Private
         %81 = OpTypeStruct %37 %37
         %84 = OpTypeStruct %6 %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
         %21 = OpVariable %20 Function
         %23 = OpVariable %20 Function
         %25 = OpVariable %20 Function
         %34 = OpVariable %33 Function
         %39 = OpVariable %38 Function
         %44 = OpVariable %33 Function
         %48 = OpVariable %38 Function
               OpStore %8 %9
               OpStore %10 %11
         %13 = OpLoad %6 %10
         %14 = OpLoad %6 %8
         %15 = OpSDiv %6 %13 %14
               OpStore %12 %15
         %16 = OpLoad %6 %10
         %17 = OpLoad %6 %8
         %18 = OpIAdd %6 %16 %17
               OpStore %12 %18
               OpStore %21 %22
               OpStore %23 %24
         %26 = OpLoad %19 %21
         %27 = OpLoad %19 %23
         %83 = OpUMulExtended %52 %26 %27
         %28 = OpCompositeExtract %19 %83 0
               OpStore %25 %28
         %29 = OpLoad %6 %10
         %30 = OpLoad %6 %8
         %85 = OpSMulExtended %84 %29 %30
         %31 = OpCompositeExtract %6 %85 0
               OpStore %12 %31
               OpStore %34 %36
               OpStore %39 %43
         %45 = OpLoad %32 %34
         %46 = OpLoad %32 %34
         %47 = OpIAdd %32 %45 %46
               OpStore %44 %47
         %49 = OpLoad %37 %39
         %50 = OpLoad %37 %39
         %86 = OpUMulExtended %81 %49 %50
         %51 = OpCompositeExtract %37 %86 0
               OpStore %48 %51
         %70 = OpIAdd %19 %16 %26
         %71 = OpISub %6 %26 %27
               OpReturn
               OpFunctionEnd
           )";
  ASSERT_TRUE(IsEqual(env, after_transformations, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
