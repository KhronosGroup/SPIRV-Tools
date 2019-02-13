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
#include "source/reduce/simplify_selection_reduction_opportunity_finder.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(SimplifySelectionTest, OneSelectionToBeSimplified) {
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

  // Loop twice because we want to consider redirecting both branches of the
  // selection to the LHS, and both branches of the selection to the RHS.
  for (uint32_t i = 0; i < 2; i++) {
    const auto env = SPV_ENV_UNIVERSAL_1_3;
    const auto consumer = nullptr;
    const auto context =
        BuildModule(env, consumer, shader, kReduceAssembleOption);

    auto ops =
        SimplifySelectionReductionOpportunityFinder().GetAvailableOpportunities(
            context.get());

    // There is one selection that can be simplified, and simplification can
    // work in two ways: the LHS branch can be made to point to the RHS branch,
    // or vice versa.
    ASSERT_EQ(2, ops.size());

    // Initially, both reductions are possible.
    ASSERT_TRUE(ops[0]->PreconditionHolds());
    ASSERT_TRUE(ops[1]->PreconditionHolds());

    // After we apply the one of the reductions, the other reduction should no
    // longer be possible.
    ops[i]->TryToApply();
    CheckValid(env, context.get());
    ASSERT_FALSE(ops[1 - i]->PreconditionHolds());

    // This is what applying one of the reductions should give us:
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
                 OpSelectionMerge %15 None
)";
    // What this conditional branch looks like depends on which of its targets
    // we have redirected edges to.
    after += "OpBranchConditional %13 " +
             (i == 0 ? std::string("%14 %14") : std::string("%17 %17")) + "\n";
    after += R"(
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

    // There should be no more opportunities after we have applied the
    // reduction.
    ops =
        SimplifySelectionReductionOpportunityFinder().GetAvailableOpportunities(
            context.get());
    // No more opportunities should remain.
    ASSERT_EQ(0, ops.size());
  }
}

TEST(SimplifySelectionTest, ManySimplifications) {
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
      SimplifySelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(6, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

  // Redirecting the outer selection one way should disable redirecting of it
  // the other way.
  ASSERT_FALSE(ops[0]->PreconditionHolds());  // This opportunity was just taken
  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ASSERT_FALSE(ops[3]->PreconditionHolds());  // This opportunity corresponds to
                                              // the other way the outer
                                              // selection could have been
                                              // redirected.
  ASSERT_TRUE(ops[4]->PreconditionHolds());
  ASSERT_TRUE(ops[5]->PreconditionHolds());

  std::string after1 = R"(
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
  CheckEqual(env, after1, context.get());

  ops[5]->TryToApply();
  CheckValid(env, context.get());

  ASSERT_FALSE(ops[0]->PreconditionHolds());  // Previously taken.
  ASSERT_TRUE(ops[1]->PreconditionHolds());   // Still available.
  ASSERT_FALSE(ops[2]->PreconditionHolds());  // Just disabled by the
                                              // opportunity that has been
                                              // taken.
  ASSERT_FALSE(ops[3]->PreconditionHolds());  // Previously disabled.
  ASSERT_TRUE(ops[4]->PreconditionHolds());   // Still available.
  ASSERT_FALSE(ops[5]->PreconditionHolds());  // Just taken.

  std::string after2 = R"(
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
  CheckEqual(env, after2, context.get());

  ops[1]->TryToApply();
  CheckValid(env, context.get());

  ASSERT_FALSE(ops[0]->PreconditionHolds());  // Previously taken.
  ASSERT_FALSE(ops[1]->PreconditionHolds());  // Just taken.
  ASSERT_FALSE(ops[2]->PreconditionHolds());  // Previously disabled.
  ASSERT_FALSE(ops[3]->PreconditionHolds());  // Previously disabled.
  ASSERT_FALSE(ops[4]->PreconditionHolds());  // Just disabled by the
                                              // opportunity that has been
                                              // taken.
  ASSERT_FALSE(ops[5]->PreconditionHolds());  // Previously taken.

  std::string after3 = R"(
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
  CheckEqual(env, after3, context.get());

  // There should be no more opportunities left.
  ops = SimplifySelectionReductionOpportunityFinder().GetAvailableOpportunities(
      context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(SimplifySelectionTest, NoSimplifications) {
  // This SPIR-V provides no reduction opportunities of this kind.  It does have
  // a switch construct, which should be ignored by this pass.
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
         %16 = OpConstant %6 0
         %17 = OpTypeBool
         %24 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSGreaterThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %19 = OpLoad %6 %8
               OpSelectionMerge %22 None
               OpSwitch %19 %21 1 %20
         %21 = OpLabel
         %26 = OpLoad %6 %8
         %27 = OpIAdd %6 %26 %24
               OpStore %8 %27
               OpBranch %22
         %20 = OpLabel
         %23 = OpLoad %6 %8
         %25 = OpISub %6 %23 %24
               OpStore %8 %25
               OpBranch %21
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);

  auto ops =
      SimplifySelectionReductionOpportunityFinder().GetAvailableOpportunities(
          context.get());

  ASSERT_EQ(0, ops.size());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
