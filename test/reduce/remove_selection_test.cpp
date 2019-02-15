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

#include "reduce_test_util.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "source/reduce/reduction_pass.h"
#include "source/reduce/remove_selection_reduction_opportunity_finder.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(RemoveSelectionTest, OneRemoval) {
  std::string shader = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpSelectionMerge %15 None
                 OpBranchConditional %13 %14 %14
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

  std::string after = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpBranch %14
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  // There should be no more opportunities after we have applied the reduction.
  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  // No more opportunities should remain.
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, ChooseBetweenTwoRemovals) {
  std::string shader = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpSelectionMerge %15 None
                 OpBranchConditional %13 %14 %17
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(2, ops.size());

  // Both opportunities should initially be available.
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ASSERT_TRUE(ops[1]->PreconditionHolds());

  // Taking the second opportunity should disable the first.
  ops[1]->TryToApply();
  CheckValid(env, context.get());
  ASSERT_FALSE(ops[0]->PreconditionHolds());

  std::string after = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpBranch %17
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  // There should be no more opportunities after we have applied the reduction.
  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  // No more opportunities should remain.
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, ManyRemovalsIdenticalTargets) {
  // There are multiple opportunities for removing selections, but in each case
  // the targets of the selection are the same.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 3
         %11 = OpConstant %6 4
         %12 = OpTypeBool
         %20 = OpConstant %6 1
         %22 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %13 = OpSGreaterThan %12 %10 %11
               OpSelectionMerge %15 None
               OpBranchConditional %13 %14 %14
         %14 = OpLabel
         %16 = OpLoad %6 %8
         %17 = OpSGreaterThan %12 %16 %11
               OpSelectionMerge %19 None
               OpBranchConditional %17 %18 %18
         %18 = OpLabel
               OpStore %8 %20
               OpBranch %19
         %21 = OpLabel
               OpStore %8 %22
               OpBranch %19
         %19 = OpLabel
         %23 = OpLoad %6 %8
         %24 = OpIMul %6 %23 %22
               OpStore %8 %24
               OpBranch %15
         %25 = OpLabel
         %26 = OpLoad %6 %8
         %27 = OpSGreaterThan %12 %26 %11
               OpSelectionMerge %29 None
               OpBranchConditional %27 %30 %30
         %28 = OpLabel
               OpStore %8 %20
               OpBranch %29
         %30 = OpLabel
               OpStore %8 %22
               OpBranch %29
         %29 = OpLabel
               OpStore %8 %22
               OpBranch %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(3, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());
  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ops[2]->TryToApply();
  CheckValid(env, context.get());

  std::string after = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 3
         %11 = OpConstant %6 4
         %12 = OpTypeBool
         %20 = OpConstant %6 1
         %22 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %13 = OpSGreaterThan %12 %10 %11
               OpBranch %14
         %14 = OpLabel
         %16 = OpLoad %6 %8
         %17 = OpSGreaterThan %12 %16 %11
               OpBranch %18
         %18 = OpLabel
               OpStore %8 %20
               OpBranch %19
         %21 = OpLabel
               OpStore %8 %22
               OpBranch %19
         %19 = OpLabel
         %23 = OpLoad %6 %8
         %24 = OpIMul %6 %23 %22
               OpStore %8 %24
               OpBranch %15
         %25 = OpLabel
         %26 = OpLoad %6 %8
         %27 = OpSGreaterThan %12 %26 %11
               OpBranch %30
         %28 = OpLabel
               OpStore %8 %20
               OpBranch %29
         %30 = OpLabel
               OpStore %8 %22
               OpBranch %29
         %29 = OpLabel
               OpStore %8 %22
               OpBranch %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after, context.get());

  // There should be no more opportunities left.
  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, ManyRemovalsDifferentTargets) {
  // There are multiple opportunities for removing selections, and in each case
  // the selection has distinct targets.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 3
         %11 = OpConstant %6 4
         %12 = OpTypeBool
         %20 = OpConstant %6 1
         %22 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %13 = OpSGreaterThan %12 %10 %11
               OpSelectionMerge %15 None
               OpBranchConditional %13 %14 %25
         %14 = OpLabel
         %16 = OpLoad %6 %8
         %17 = OpSGreaterThan %12 %16 %11
               OpSelectionMerge %19 None
               OpBranchConditional %17 %18 %21
         %18 = OpLabel
               OpStore %8 %20
               OpBranch %19
         %21 = OpLabel
               OpStore %8 %22
               OpBranch %19
         %19 = OpLabel
         %23 = OpLoad %6 %8
         %24 = OpIMul %6 %23 %22
               OpStore %8 %24
               OpBranch %15
         %25 = OpLabel
         %26 = OpLoad %6 %8
         %27 = OpSGreaterThan %12 %26 %11
               OpSelectionMerge %29 None
               OpBranchConditional %27 %28 %30
         %28 = OpLabel
               OpStore %8 %20
               OpBranch %29
         %30 = OpLabel
               OpStore %8 %22
               OpBranch %29
         %29 = OpLabel
               OpStore %8 %22
               OpBranch %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(6, ops.size());

  // Apply some of the opportunities, and check that the others end up getting
  // disabled.
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  ASSERT_TRUE(ops[4]->PreconditionHolds());
  ops[4]->TryToApply();
  CheckValid(env, context.get());
  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ops[2]->TryToApply();
  CheckValid(env, context.get());
  ASSERT_FALSE(ops[1]->PreconditionHolds());
  ASSERT_FALSE(ops[3]->PreconditionHolds());
  ASSERT_FALSE(ops[5]->PreconditionHolds());

  // The expected result from applying the above opportunities: each selection
  // construct has been eliminated and its former header modified to branch to
  // one of its previous targets.
  std::string after = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 3
         %11 = OpConstant %6 4
         %12 = OpTypeBool
         %20 = OpConstant %6 1
         %22 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %13 = OpSGreaterThan %12 %10 %11
               OpBranch %14
         %14 = OpLabel
         %16 = OpLoad %6 %8
         %17 = OpSGreaterThan %12 %16 %11
               OpBranch %21
         %18 = OpLabel
               OpStore %8 %20
               OpBranch %19
         %21 = OpLabel
               OpStore %8 %22
               OpBranch %19
         %19 = OpLabel
         %23 = OpLoad %6 %8
         %24 = OpIMul %6 %23 %22
               OpStore %8 %24
               OpBranch %15
         %25 = OpLabel
         %26 = OpLoad %6 %8
         %27 = OpSGreaterThan %12 %26 %11
               OpBranch %28
         %28 = OpLabel
               OpStore %8 %20
               OpBranch %29
         %30 = OpLabel
               OpStore %8 %22
               OpBranch %29
         %29 = OpLabel
               OpStore %8 %22
               OpBranch %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after, context.get());

  // There should be no more opportunities left.
  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, BreakOutOfSelection) {
  // This is designed to check an edge case where there is a break out of the
  // selection being removed.
  std::string shader = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %20 = OpConstantTrue %12
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpSelectionMerge %15 None
                 OpBranchConditional %13 %14 %17
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranchConditional %20 %15 %19 ; Early-exit from the selection
           %19 = OpLabel
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(2, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

  std::string after = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %20 = OpConstantTrue %12
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpBranch %14
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranch %19
           %19 = OpLabel
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  // There should be no more opportunities after we have applied the reduction.
  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  // No more opportunities should remain.
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, BreakOutOfSelection2) {
  // Another test designed to check correctness of this kind of reduction in the
  // presence of a break from a selection.
  std::string shader = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %20 = OpConstantTrue %12
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpSelectionMerge %15 None
                 OpBranchConditional %13 %14 %17
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranchConditional %20 %19 %15 ; Early-exit from the selection
           %19 = OpLabel
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(2, ops.size());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());

  std::string after = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %20 = OpConstantTrue %12
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpBranch %17
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranch %19
           %19 = OpLabel
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  // There should be no more opportunities after we have applied the reduction.
  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  // No more opportunities should remain.
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, BreakOutOfSelection3) {
  // Test case where an unconditional break out of a selection is disguised as a
  // conditional break, via a conditional for which both targets jump to the
  // merge.
  std::string shader = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %20 = OpConstantTrue %12
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpSelectionMerge %15 None
                 OpBranchConditional %13 %14 %17
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranchConditional %20 %15 %15 ; Both targets jump to the merge
           %19 = OpLabel
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(2, ops.size());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());

  std::string after = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %20 = OpConstantTrue %12
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpBranch %17
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranch %15
           %19 = OpLabel
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  // There should be no more opportunities after we have applied the reduction.
  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  // No more opportunities should remain.
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveSelectionTest, RedundantBranchInsideConditional) {
  // Check that the reduction succeeds in presence of the edge case where the
  // conditional being removed contains an inner branch, such that both targets
  // of the inner branch lead to the same node (so that the inner branch need
  // not be preceded by a merge).
  std::string shader = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %19 = OpConstantTrue %12
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpSelectionMerge %15 None
                 OpBranchConditional %13 %14 %17
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranchConditional %19 %20 %20
           %20 = OpLabel
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  // Check that the initial SPIR-V is valid, as this is an edge case.
  CheckValid(env, context.get());

  auto ops =
      RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(2, ops.size());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());

  std::string after = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 310
                 OpName %4 "main"
                 OpName %8 "x"
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
            %7 = OpTypePointer Function %6
            %9 = OpConstant %6 3
           %11 = OpConstant %6 4
           %12 = OpTypeBool
           %19 = OpConstantTrue %12
           %16 = OpConstant %6 1
           %18 = OpConstant %6 2
            %4 = OpFunction %2 None %3
            %5 = OpLabel
            %8 = OpVariable %7 Function
                 OpStore %8 %9
           %10 = OpLoad %6 %8
           %13 = OpSGreaterThan %12 %10 %11
                 OpBranch %17
           %14 = OpLabel
                 OpStore %8 %16
                 OpBranch %15
           %17 = OpLabel
                 OpStore %8 %18
                 OpBranchConditional %19 %20 %20
           %20 = OpLabel
                 OpBranch %15
           %15 = OpLabel
                 OpReturn
                 OpFunctionEnd
    )";
  CheckEqual(env, after, context.get());

  // There should be no more opportunities after we have applied the reduction.
  ops = RemoveSelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  // No more opportunities should remain.
  ASSERT_EQ(0, ops.size());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
