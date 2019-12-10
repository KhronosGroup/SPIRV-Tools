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
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 3
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %100
        %100 = OpLabel
               OpStore %10 %11
         %12 = OpLoad %6 %10
         %13 = OpLoad %6 %8
               OpBranch %101
        %101 = OpLabel
         %14 = OpIAdd %6 %13 %12
               OpStore %8 %14
         %15 = OpLoad %6 %8
               OpBranch %102
        %102 = OpLabel
         %16 = OpLoad %6 %10
         %17 = OpIMul %6 %16 %15
               OpBranch %103
        %103 = OpLabel
               OpStore %10 %17
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  for (auto& transformation :
       {TransformationMergeBlocks(102), TransformationMergeBlocks(101),
        TransformationMergeBlocks(100), TransformationMergeBlocks(5)}) {
    ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
    transformation.Apply(context.get(), &fact_manager);
    ASSERT_TRUE(IsValid(env, context.get()));
  }

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 3
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %11
         %12 = OpLoad %6 %10
         %13 = OpLoad %6 %8
         %14 = OpIAdd %6 %13 %12
               OpStore %8 %14
         %15 = OpLoad %6 %8
         %16 = OpLoad %6 %10
         %17 = OpIMul %6 %16 %15
               OpStore %10 %17
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationMergeBlocksTest, MergeWhenSuccessorIsSelectionHeader) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 3
         %50 = OpTypeBool
         %51 = OpConstantTrue %50
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %100
        %100 = OpLabel
               OpStore %10 %11
         %12 = OpLoad %6 %10
         %13 = OpLoad %6 %8
               OpBranch %101
        %101 = OpLabel
               OpSelectionMerge %103 None
               OpBranchConditional %51 %102 %103
        %102 = OpLabel
         %14 = OpIAdd %6 %13 %12
               OpStore %8 %14
               OpBranch %103
        %103 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  for (auto& transformation :
       {TransformationMergeBlocks(100), TransformationMergeBlocks(5)}) {
    ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
    transformation.Apply(context.get(), &fact_manager);
    ASSERT_TRUE(IsValid(env, context.get()));
  }

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 3
         %50 = OpTypeBool
         %51 = OpConstantTrue %50
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %11
         %12 = OpLoad %6 %10
         %13 = OpLoad %6 %8
               OpSelectionMerge %103 None
               OpBranchConditional %51 %102 %103
        %102 = OpLabel
         %14 = OpIAdd %6 %13 %12
               OpStore %8 %14
               OpBranch %103
        %103 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationMergeBlocksTest,
     MergeWhenFirstBlockIsLoopMergeFollowedByUnconditionalBranch) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 3
         %50 = OpTypeBool
         %51 = OpConstantTrue %50
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %100
        %100 = OpLabel
               OpLoopMerge %102 %103 None
               OpBranch %101
        %101 = OpLabel
               OpBranchConditional %51 %102 %103
        %103 = OpLabel
               OpBranch %100
        %102 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  TransformationMergeBlocks transformation(100);
  ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
  transformation.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 3
         %50 = OpTypeBool
         %51 = OpConstantTrue %50
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %100
        %100 = OpLabel
               OpLoopMerge %102 %103 None
               OpBranchConditional %51 %102 %103
        %103 = OpLabel
               OpBranch %100
        %102 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
