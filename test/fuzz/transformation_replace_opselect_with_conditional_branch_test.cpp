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

#include "source/fuzz/transformation_replace_opselect_with_conditional_branch.h"

#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationReplaceOpSelectWithConditionalBranchTest, Inapplicable) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpConstant %5 1
          %7 = OpTypeVector %5 4
          %8 = OpTypeBool
          %9 = OpConstantTrue %8
         %10 = OpTypeSampler
         %11 = OpTypeImage %3 2D 2 0 0 1 Unknown
         %12 = OpTypeSampledImage %11
         %13 = OpTypePointer Function %11
         %14 = OpTypePointer Function %10
         %15 = OpTypeFloat 32
         %16 = OpTypeVector %15 2
         %17 = OpConstant %15 1
         %18 = OpConstantComposite %16 %17 %17
          %2 = OpFunction %3 None %4
         %19 = OpLabel
         %20 = OpVariable %13 Function
         %21 = OpVariable %14 Function
         %22 = OpLoad %11 %20
         %23 = OpLoad %10 %21
         %24 = OpCopyObject %5 %6
         %25 = OpSelect %5 %9 %24 %6
               OpBranch %26
         %26 = OpLabel
         %27 = OpSampledImage %12 %22 %23
         %28 = OpSelect %5 %9 %6 %24
         %29 = OpImageSampleImplicitLod %7 %27 %18
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

  // %24 is not an OpSelect instruction.
  ASSERT_FALSE(
      TransformationReplaceOpSelectWithConditionalBranch(24, {100, 101})
          .IsApplicable(context.get(), transformation_context));

  // The block containing %28 cannot be split before %28 because this would
  // separate an OpSampledImage instruction from its use.
  ASSERT_FALSE(
      TransformationReplaceOpSelectWithConditionalBranch(28, {100, 101})
          .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationReplaceOpSelectWithConditionalBranchTest, Simple) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %7 = OpTypeInt 32 1
          %8 = OpConstant %7 1
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpCopyObject %7 %8
         %11 = OpSelect %7 %6 %10 %8
         %12 = OpCopyObject %7 %10
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

  // 2 fresh ids are required.
  ASSERT_FALSE(TransformationReplaceOpSelectWithConditionalBranch(11, {100})
                   .IsApplicable(context.get(), transformation_context));

  // One of the ids are not fresh.
  ASSERT_FALSE(TransformationReplaceOpSelectWithConditionalBranch(11, {100, 11})
                   .IsApplicable(context.get(), transformation_context));

  // The ids are repeated.
  ASSERT_FALSE(
      TransformationReplaceOpSelectWithConditionalBranch(11, {100, 100})
          .IsApplicable(context.get(), transformation_context));

  ASSERT_TRUE(TransformationReplaceOpSelectWithConditionalBranch(11, {100, 101})
                  .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationReplaceOpSelectWithConditionalBranchTest, InsideMergeBlock) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %7 = OpTypeInt 32 1
          %8 = OpConstant %7 1
          %2 = OpFunction %3 None %4
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %11 %10 None
               OpBranchConditional %6 %10 %11
         %11 = OpLabel
         %12 = OpSelect %7 %6 %8 %8
               OpBranch %13
         %13 = OpLabel
               OpSelectionMerge %14 None
               OpBranchConditional %6 %15 %14
         %15 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %16 = OpSelect %7 %6 %8 %8
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

  // 3 fresh ids are required because %12 is in a merge block.
  ASSERT_FALSE(
      TransformationReplaceOpSelectWithConditionalBranch(12, {100, 101})
          .IsApplicable(context.get(), transformation_context));

  // 3 fresh ids are required because %16 is in a merge block.
  ASSERT_FALSE(
      TransformationReplaceOpSelectWithConditionalBranch(16, {100, 101})
          .IsApplicable(context.get(), transformation_context));

  auto transformation1 =
      TransformationReplaceOpSelectWithConditionalBranch(12, {100, 101, 102});
  ASSERT_TRUE(
      transformation1.IsApplicable(context.get(), transformation_context));

  auto transformation2 =
      TransformationReplaceOpSelectWithConditionalBranch(16, {103, 104, 105});
  ASSERT_TRUE(
      transformation2.IsApplicable(context.get(), transformation_context));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools