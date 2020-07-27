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

#include "source/fuzz/transformation_add_loop_preheader.h"

#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddLoopPreheaderTest, SimpleTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpConstantFalse %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpLoopMerge %10 %9 None
               OpBranch %9
          %9 = OpLabel
               OpBranchConditional %7 %8 %10
         %10 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %7 %11 %12
         %11 = OpLabel
               OpBranch %13
         %12 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpLoopMerge %15 %14 None
               OpBranch %14
         %14 = OpLabel
               OpBranchConditional %7 %13 %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  ASSERT_TRUE(IsValid(env, context.get()));

  // %9 is not a loop header
  ASSERT_FALSE(TransformationAddLoopPreheader(9, 13, {}).IsApplicable(
      context.get(), transformation_context));

  // The id %10 is not fresh
  ASSERT_FALSE(TransformationAddLoopPreheader(8, 10, {}).IsApplicable(
      context.get(), transformation_context));

  ASSERT_TRUE(TransformationAddLoopPreheader(8, 20, {}).IsApplicable(
      context.get(), transformation_context));

  ASSERT_TRUE(TransformationAddLoopPreheader(13, 21, {})
                  .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddLoopPreheaderTest, OpPhi) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpConstantFalse %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %20 = OpCopyObject %6 %7
               OpBranch %8
          %8 = OpLabel
         %31 = OpPhi %6 %20 %5 %21 %9
               OpLoopMerge %10 %9 None
               OpBranch %9
          %9 = OpLabel
         %21 = OpCopyObject %6 %7
               OpBranchConditional %7 %8 %10
         %10 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %7 %11 %12
         %11 = OpLabel
         %22 = OpCopyObject %6 %7
               OpBranch %13
         %12 = OpLabel
         %23 = OpCopyObject %6 %7
               OpBranch %13
         %13 = OpLabel
         %32 = OpPhi %6 %22 %11 %23 %12 %24 %14
         %33 = OpPhi %6 %7 %11 %7 %12 %24 %14
               OpLoopMerge %15 %14 None
               OpBranch %14
         %14 = OpLabel
         %24 = OpCopyObject %6 %7
               OpBranchConditional %7 %13 %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  ASSERT_TRUE(IsValid(env, context.get()));

  ASSERT_TRUE(TransformationAddLoopPreheader(8, 40, {}).IsApplicable(
      context.get(), transformation_context));

  // Not enough ids for the OpPhi instructions are given
  ASSERT_FALSE(TransformationAddLoopPreheader(13, 41, {})
                   .IsApplicable(context.get(), transformation_context));

  // Not enough ids for the OpPhi instructions are given
  ASSERT_FALSE(TransformationAddLoopPreheader(13, 41, {42})
                   .IsApplicable(context.get(), transformation_context));

  // One of the ids is not fresh
  ASSERT_FALSE(TransformationAddLoopPreheader(13, 41, {31, 42})
                   .IsApplicable(context.get(), transformation_context));

  // One of the ids is repeated
  ASSERT_FALSE(TransformationAddLoopPreheader(13, 41, {41, 42})
                   .IsApplicable(context.get(), transformation_context));

  ASSERT_TRUE(TransformationAddLoopPreheader(13, 41, {42, 43})
                  .IsApplicable(context.get(), transformation_context));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools