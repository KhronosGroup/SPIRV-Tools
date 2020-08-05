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
               OpName %38 "uint2"
               OpMemberName %38 0 "a"
               OpMemberName %38 1 "b"
               OpName %40 "result_uint"
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
         %38 = OpTypeStruct %19 %19
         %39 = OpTypePointer Private %38
         %40 = OpVariable %39 Private
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
         %21 = OpVariable %20 Function
         %23 = OpVariable %20 Function
         %25 = OpVariable %20 Function
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
         %28 = OpIAdd %19 %26 %27
               OpStore %25 %28
         %29 = OpLoad %19 %21
         %30 = OpLoad %19 %23
         %31 = OpISub %19 %29 %30
               OpStore %25 %31
         %32 = OpLoad %19 %21
         %33 = OpLoad %19 %23
         %34 = OpIMul %19 %32 %33
               OpStore %25 %34
         %35 = OpLoad %6 %10
         %36 = OpLoad %6 %8
         %37 = OpIMul %6 %35 %36
               OpStore %12 %37
         %60 = OpIAdd %19 %16 %26
         %61 = OpIAdd %6 %26 %27
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
      TransformationReplaceAddSubMulWithCarryingExtended(50, 15);
  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: The transformation cannot be applied to an instruction OpIAdd that has
  // signed variables as operands.
  auto transformation_bad_3 =
      TransformationReplaceAddSubMulWithCarryingExtended(50, 18);
  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));

  // Bad: The transformation cannot be applied to an instruction OpIAdd that has
  // different signedness of the types of operands.
  auto transformation_bad_4 =
      TransformationReplaceAddSubMulWithCarryingExtended(50, 60);
  ASSERT_FALSE(
      transformation_bad_4.IsApplicable(context.get(), transformation_context));

  // Bad: The transformation cannot be applied to an instruction OpISub that has
  // different signedness of the result type than the signedness of the types of
  // the operands.
  auto transformation_bad_5 =
      TransformationReplaceAddSubMulWithCarryingExtended(50, 61);
  ASSERT_FALSE(
      transformation_bad_5.IsApplicable(context.get(), transformation_context));

  // Bad: The instruction with result id 70 doesn't exist.
  auto transformation_bad_6 =
      TransformationReplaceAddSubMulWithCarryingExtended(50, 70);
  ASSERT_FALSE(
      transformation_bad_6.IsApplicable(context.get(), transformation_context));

  auto transformation_good_1 =
      TransformationReplaceAddSubMulWithCarryingExtended(50, 28);
  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));

  transformation_good_1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  auto transformation_good_2 =
      TransformationReplaceAddSubMulWithCarryingExtended(51, 31);
  ASSERT_TRUE(transformation_good_2.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_2.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  auto transformation_good_3 =
      TransformationReplaceAddSubMulWithCarryingExtended(52, 34);
  ASSERT_TRUE(transformation_good_3.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_3.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Explicitly create the required struct type for the OpSMulExtended.
  std::vector<uint32_t> operand_type_ids = {6, 6};
  fuzzerutil::AddStructType(context.get(), 54, operand_type_ids);
  context.get()->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  auto transformation_good_4 =
      TransformationReplaceAddSubMulWithCarryingExtended(53, 37);
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
               OpName %8 "i1"
               OpName %10 "i2"
               OpName %12 "i3"
               OpName %21 "u1"
               OpName %23 "u2"
               OpName %25 "u3"
               OpName %38 "uint2"
               OpMemberName %38 0 "a"
               OpMemberName %38 1 "b"
               OpName %40 "result_uint"
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
         %38 = OpTypeStruct %19 %19
         %39 = OpTypePointer Private %38
         %40 = OpVariable %39 Private
         %54 = OpTypeStruct %6 %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
         %21 = OpVariable %20 Function
         %23 = OpVariable %20 Function
         %25 = OpVariable %20 Function
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
         %50 = OpIAddCarry %38 %26 %27
         %28 = OpCompositeExtract %19 %50 0
               OpStore %25 %28
         %29 = OpLoad %19 %21
         %30 = OpLoad %19 %23
         %51 = OpISubBorrow %38 %29 %30
         %31 = OpCompositeExtract %19 %51 0
               OpStore %25 %31
         %32 = OpLoad %19 %21
         %33 = OpLoad %19 %23
         %52 = OpUMulExtended %38 %32 %33
         %34 = OpCompositeExtract %19 %52 0
               OpStore %25 %34
         %35 = OpLoad %6 %10
         %36 = OpLoad %6 %8
         %53 = OpSMulExtended %54 %35 %36
         %37 = OpCompositeExtract %6 %53 0
               OpStore %12 %37
         %60 = OpIAdd %19 %16 %26
         %61 = OpIAdd %6 %26 %27
               OpReturn
               OpFunctionEnd
    )";
  ASSERT_TRUE(IsEqual(env, after_transformations, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
