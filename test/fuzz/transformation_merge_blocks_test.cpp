// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_merge_blocks.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationMergeBlocksTest, BlockDoesNotExist) {
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
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  ASSERT_FALSE(
      TransformationMergeBlocks(3).IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(
      TransformationMergeBlocks(7).IsApplicable(context.get(), fact_manager));
}

TEST(TransformationMergeBlocksTest, DoNotMergeBlockHasMultipleSuccessors) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpSelectionMerge %10 None
               OpBranchConditional %8 %6 %9
          %6 = OpLabel
               OpBranch %10
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  ASSERT_FALSE(
      TransformationMergeBlocks(5).IsApplicable(context.get(), fact_manager));
}

TEST(TransformationMergeBlocksTest,
     DoNotMergeBlockSuccessorHasMultiplePredecessors) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpSelectionMerge %10 None
               OpBranchConditional %8 %6 %9
          %6 = OpLabel
               OpBranch %10
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  ASSERT_FALSE(
      TransformationMergeBlocks(6).IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(
      TransformationMergeBlocks(9).IsApplicable(context.get(), fact_manager));
}

TEST(TransformationMergeBlocksTest, DoNotMergeSuccessorIsSelectionMerge) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpSelectionMerge %10 None
               OpBranchConditional %8 %6 %9
          %6 = OpLabel
               OpBranch %11
          %9 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  ASSERT_FALSE(
      TransformationMergeBlocks(11).IsApplicable(context.get(), fact_manager));
}

TEST(TransformationMergeBlocksTest, DoNotMergeSuccessorIsLoopMerge) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
         %12 = OpLabel
               OpBranch %5
          %5 = OpLabel
               OpLoopMerge %10 %11 None
               OpBranch %6
          %6 = OpLabel
               OpBranchConditional %8 %9 %11
          %9 = OpLabel
               OpBranch %10
         %11 = OpLabel
               OpBranch %5
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  ASSERT_FALSE(
      TransformationMergeBlocks(9).IsApplicable(context.get(), fact_manager));
}

TEST(TransformationMergeBlocksTest, DoNotMergeSuccessorIsLoopContinue) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
         %13 = OpLabel
               OpBranch %5
          %5 = OpLabel
               OpLoopMerge %10 %11 None
               OpBranch %6
          %6 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %8 %9 %12
         %12 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpBranch %5
          %9 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  ASSERT_FALSE(
      TransformationMergeBlocks(12).IsApplicable(context.get(), fact_manager));
}

TEST(TransformationMergeBlocksTest, DoNotMergeSuccessorStartsWithOpPhi) {
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
          %7 = OpTypeBool
          %8 = OpUndef %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
          %9 = OpPhi %7 %8 %5
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  ASSERT_FALSE(
      TransformationMergeBlocks(5).IsApplicable(context.get(), fact_manager));
}

TEST(TransformationMergeBlocksTest, BasicMerge) {
  std::string shader = R"(
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  FAIL();
}

TEST(TransformationMergeBlocksTest, MergeWhenSuccessorIsSelectionHeader) {
  std::string shader = R"(
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  FAIL();
}

TEST(TransformationMergeBlocksTest,
     MergeWhenFirstBlockIsLoopMergeFollowedByUnconditionalBranch) {
  std::string shader = R"(
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  FAIL();
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
