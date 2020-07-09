// Copyright (c) 2020 Stefano Milizia
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

#include "source/fuzz/transformation_record_synonymous_constants.h"

#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

// Apply the TransformationRecordSynonymousConstants defined by the given
// constant1_id and constant2_id and check that the fact that the two
// constants are synonym is recorded.
void ApplyTransformationAndCheckFactManager(
    uint32_t constant1_id, uint32_t constant2_id, opt::IRContext* ir_context,
    TransformationContext* transformation_context) {
  TransformationRecordSynonymousConstants(constant1_id, constant2_id)
      .Apply(ir_context, transformation_context);

  ASSERT_TRUE(transformation_context->GetFactManager()->IsSynonymous(
      MakeDataDescriptor(constant1_id, {}),
      MakeDataDescriptor(constant2_id, {})));
}

TEST(TransformationRecordSynonymousConstantsTest, IntConstants) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %17
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %12 "c"
               OpName %17 "color"
               OpDecorate %8 RelaxedPrecision
               OpDecorate %10 RelaxedPrecision
               OpDecorate %12 RelaxedPrecision
               OpDecorate %17 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 0
         %19 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %18 = OpConstant %6 0
         %11 = OpConstantNull %6
         %13 = OpConstant %6 1
         %20 = OpConstant %19 1
         %21 = OpConstant %19 -1
         %22 = OpConstant %6 1
         %14 = OpTypeFloat 32
         %15 = OpTypeVector %14 4
         %16 = OpTypePointer Output %15
         %17 = OpVariable %16 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %13
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
  ASSERT_TRUE(IsValid(env, context.get()));

  // %3 is not a constant declaration
  ASSERT_FALSE(TransformationRecordSynonymousConstants(3, 9).IsApplicable(
      context.get(), transformation_context));

  // Swapping the ids gives the same result
  ASSERT_FALSE(TransformationRecordSynonymousConstants(9, 3).IsApplicable(
      context.get(), transformation_context));

  // The two constants must be different
  ASSERT_FALSE(TransformationRecordSynonymousConstants(9, 9).IsApplicable(
      context.get(), transformation_context));

  // %9 and %13 are not equivalent
  ASSERT_FALSE(TransformationRecordSynonymousConstants(9, 13).IsApplicable(
      context.get(), transformation_context));

  // Swapping the ids gives the same result
  ASSERT_FALSE(TransformationRecordSynonymousConstants(13, 9).IsApplicable(
      context.get(), transformation_context));

  // %11 and %13 are not equivalent
  ASSERT_FALSE(TransformationRecordSynonymousConstants(11, 13).IsApplicable(
      context.get(), transformation_context));

  // Swapping the ids gives the same result
  ASSERT_FALSE(TransformationRecordSynonymousConstants(13, 11).IsApplicable(
      context.get(), transformation_context));

  // %20 and %21 have different values
  ASSERT_FALSE(TransformationRecordSynonymousConstants(20, 21).IsApplicable(
      context.get(), transformation_context));

  // %13 and %22 are equal and thus equivalent (having the same value and type)
  ASSERT_TRUE(TransformationRecordSynonymousConstants(13, 22).IsApplicable(
      context.get(), transformation_context));

  ApplyTransformationAndCheckFactManager(13, 22, context.get(),
                                         &transformation_context);

  // %13 and %20 are equal even if %13 is signed and %20 is unsigned
  ASSERT_TRUE(TransformationRecordSynonymousConstants(13, 20).IsApplicable(
      context.get(), transformation_context));

  ApplyTransformationAndCheckFactManager(13, 20, context.get(),
                                         &transformation_context);

  // %9 and %11 are equivalent (OpConstant with value 0 and OpConstantNull)
  ASSERT_TRUE(TransformationRecordSynonymousConstants(9, 11).IsApplicable(
      context.get(), transformation_context));

  ApplyTransformationAndCheckFactManager(9, 11, context.get(),
                                         &transformation_context);

  // Swapping the ids gives the same result
  ASSERT_TRUE(TransformationRecordSynonymousConstants(11, 9).IsApplicable(
      context.get(), transformation_context));

  ApplyTransformationAndCheckFactManager(11, 9, context.get(),
                                         &transformation_context);
}

TEST(TransformationRecordSynonymousConstantsTest, BoolConstants) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %19
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "b"
               OpName %19 "color"
               OpDecorate %19 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpTypePointer Function %6
          %9 = OpConstantFalse %6
         %20 = OpConstantNull %6
         %11 = OpConstantTrue %6
         %21 = OpConstantFalse %6
         %22 = OpConstantTrue %6
         %16 = OpTypeFloat 32
         %17 = OpTypeVector %16 4
         %18 = OpTypePointer Output %17
         %19 = OpVariable %18 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %12 = OpLogicalEqual %6 %10 %11
               OpSelectionMerge %14 None
               OpBranchConditional %12 %13 %14
         %13 = OpLabel
               OpReturn
         %14 = OpLabel
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
  ASSERT_TRUE(IsValid(env, context.get()));

  // %9 and %11 are not equivalent
  ASSERT_FALSE(TransformationRecordSynonymousConstants(9, 11).IsApplicable(
      context.get(), transformation_context));

  // %20 and %11 are not equivalent
  ASSERT_FALSE(TransformationRecordSynonymousConstants(20, 11).IsApplicable(
      context.get(), transformation_context));

  // %9 and %21 are equivalent (both OpConstantFalse)
  ASSERT_TRUE(TransformationRecordSynonymousConstants(9, 21).IsApplicable(
      context.get(), transformation_context));

  ApplyTransformationAndCheckFactManager(9, 21, context.get(),
                                         &transformation_context);

  // %11 and %22 are equivalent (both OpConstantTrue)
  ASSERT_TRUE(TransformationRecordSynonymousConstants(11, 22).IsApplicable(
      context.get(), transformation_context));

  ApplyTransformationAndCheckFactManager(11, 22, context.get(),
                                         &transformation_context);

  // %9 and %20 are equivalent (OpConstantFalse and boolean OpConstantNull)
  ASSERT_TRUE(TransformationRecordSynonymousConstants(9, 20).IsApplicable(
      context.get(), transformation_context));

  ApplyTransformationAndCheckFactManager(9, 20, context.get(),
                                         &transformation_context);
}

TEST(TransformationRecordSynonymousConstantsTest, FloatConstants) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %22
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %12 "c"
               OpName %22 "color"
               OpDecorate %22 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %11 = OpConstantNull %6
         %13 = OpConstant %6 2
         %26 = OpConstant %6 2
         %16 = OpTypeBool
         %20 = OpTypeVector %6 4
         %21 = OpTypePointer Output %20
         %22 = OpVariable %21 Output
         %23 = OpConstantComposite %20 %9 %11 %9 %11
         %25 = OpConstantComposite %20 %11 %9 %9 %11
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %13
         %14 = OpLoad %6 %8
         %15 = OpLoad %6 %10
         %17 = OpFOrdEqual %16 %14 %15
               OpSelectionMerge %19 None
               OpBranchConditional %17 %18 %24
         %18 = OpLabel
               OpStore %22 %23
               OpBranch %19
         %24 = OpLabel
               OpStore %22 %25
               OpBranch %19
         %19 = OpLabel
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
  ASSERT_TRUE(IsValid(env, context.get()));

  // %9 and %13 are not equivalent
  ASSERT_FALSE(TransformationRecordSynonymousConstants(9, 13).IsApplicable(
      context.get(), transformation_context));

  // %11 and %13 are not equivalent
  ASSERT_FALSE(TransformationRecordSynonymousConstants(11, 13).IsApplicable(
      context.get(), transformation_context));

  // %13 and %23 are not equivalent
  ASSERT_FALSE(TransformationRecordSynonymousConstants(13, 23).IsApplicable(
      context.get(), transformation_context));

  // %13 and %26 are identical float constants
  ASSERT_TRUE(TransformationRecordSynonymousConstants(13, 26).IsApplicable(
      context.get(), transformation_context));

  ApplyTransformationAndCheckFactManager(13, 26, context.get(),
                                         &transformation_context);

  // %9 and %11 are equivalent ()
  ASSERT_TRUE(TransformationRecordSynonymousConstants(9, 11).IsApplicable(
      context.get(), transformation_context));

  ApplyTransformationAndCheckFactManager(9, 11, context.get(),
                                         &transformation_context);
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
