// Copyright (c) 2018 Google LLC
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

#include "source/reduce/cut_loop_reduction_pass.h"
#include "reduce_test_util.h"
#include "source/opt/build_module.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(CutLoopReductionPassTest, LoopyShader1) {
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
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %20 = OpConstant %6 1
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
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %19 = OpLoad %6 %8
         %21 = OpIAdd %6 %19 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());

  std::string after_op_0 = R"(
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
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %20 = OpConstant %6 1
         %22 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %22 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %19 = OpLoad %6 %8
         %21 = OpIAdd %6 %19 %20
               OpStore %8 %21
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
}

TEST(CutLoopReductionPassTest, LoopyShader2) {
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
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %43 %44 None
               OpBranch %45
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %44
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(4, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
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
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
         %52 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %52 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %43 %44 None
               OpBranch %45
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %44
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_1 = R"(
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
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
         %52 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %52 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpSelectionMerge %22 None
               OpBranchConditional %52 %24 %22
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %22
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %43 %44 None
               OpBranch %45
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %44
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_1, context.get());

  ASSERT_TRUE(ops[2]->PreconditionHolds());
  ops[2]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_2 = R"(
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
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
         %52 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %52 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpSelectionMerge %22 None
               OpBranchConditional %52 %24 %22
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %22
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpSelectionMerge %35 None
               OpBranchConditional %52 %37 %35
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpLoopMerge %43 %44 None
               OpBranch %45
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %44
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %35
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_2, context.get());

  ASSERT_TRUE(ops[3]->PreconditionHolds());
  ops[3]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_3 = R"(
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
          %9 = OpConstant %6 0
         %16 = OpConstant %6 100
         %17 = OpTypeBool
         %28 = OpConstant %6 1
         %52 = OpConstantTrue %17
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %40 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
               OpSelectionMerge %12 None
               OpBranchConditional %52 %14 %12
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpSelectionMerge %22 None
               OpBranchConditional %52 %24 %22
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %22
         %23 = OpLabel
         %27 = OpLoad %6 %19
         %29 = OpIAdd %6 %27 %28
               OpStore %19 %29
               OpBranch %20
         %22 = OpLabel
               OpBranch %12
         %13 = OpLabel
         %30 = OpLoad %6 %8
         %31 = OpIAdd %6 %30 %28
               OpStore %8 %31
               OpBranch %10
         %12 = OpLabel
               OpStore %32 %9
               OpBranch %33
         %33 = OpLabel
               OpSelectionMerge %35 None
               OpBranchConditional %52 %37 %35
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpSelectionMerge %43 None
               OpBranchConditional %52 %45 %43
         %45 = OpLabel
         %46 = OpLoad %6 %40
         %47 = OpSLessThan %17 %46 %16
               OpBranchConditional %47 %42 %43
         %42 = OpLabel
               OpBranch %43
         %44 = OpLabel
         %48 = OpLoad %6 %40
         %49 = OpIAdd %6 %48 %28
               OpStore %40 %49
               OpBranch %41
         %43 = OpLabel
               OpBranch %35
         %36 = OpLabel
         %50 = OpLoad %6 %32
         %51 = OpIAdd %6 %50 %28
               OpStore %32 %51
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_3, context.get());
}

TEST(CutLoopReductionPassTest, LoopyShader3) {
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
          %9 = OpConstant %6 10
         %16 = OpConstant %6 0
         %17 = OpTypeBool
         %20 = OpConstant %6 1
         %23 = OpConstant %6 3
         %40 = OpConstant %6 5
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
         %21 = OpISub %6 %19 %20
               OpStore %8 %21
         %22 = OpLoad %6 %8
         %24 = OpSLessThan %17 %22 %23
               OpSelectionMerge %26 None
               OpBranchConditional %24 %25 %26
         %25 = OpLabel
               OpBranch %13
         %26 = OpLabel
               OpBranch %28
         %28 = OpLabel
               OpLoopMerge %30 %31 None
               OpBranch %29
         %29 = OpLabel
         %32 = OpLoad %6 %8
         %33 = OpISub %6 %32 %20
               OpStore %8 %33
         %34 = OpLoad %6 %8
         %35 = OpIEqual %17 %34 %20
               OpSelectionMerge %37 None
               OpBranchConditional %35 %36 %37
         %36 = OpLabel
               OpReturn ; This return spoils everything: it means the merge does not post-dominate the header.
         %37 = OpLabel
               OpBranch %31
         %31 = OpLabel
         %39 = OpLoad %6 %8
         %41 = OpSGreaterThan %17 %39 %40
               OpBranchConditional %41 %28 %30
         %30 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(CutLoopReductionPassTest, LoopyShader4) {
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
          %8 = OpTypeFunction %6 %7
         %13 = OpConstant %6 0
         %22 = OpTypeBool
         %25 = OpConstant %6 1
         %39 = OpConstant %6 100
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %45 = OpVariable %7 Function
         %46 = OpVariable %7 Function
         %47 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %42 = OpVariable %7 Function
               OpStore %32 %13
               OpBranch %33
         %33 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %40 = OpSLessThan %22 %38 %39
               OpBranchConditional %40 %34 %35
         %34 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %41 = OpLoad %6 %32
               OpStore %42 %25
               OpStore %45 %13
               OpStore %46 %13
               OpBranch %48
         %48 = OpLabel
               OpLoopMerge %49 %50 None
               OpBranch %51
         %51 = OpLabel
         %52 = OpLoad %6 %46
         %53 = OpLoad %6 %42
         %54 = OpSLessThan %22 %52 %53
               OpBranchConditional %54 %55 %49
         %55 = OpLabel
         %56 = OpLoad %6 %45
         %57 = OpIAdd %6 %56 %25
               OpStore %45 %57
               OpBranch %50
         %50 = OpLabel
         %58 = OpLoad %6 %46
         %59 = OpIAdd %6 %58 %25
               OpStore %46 %59
               OpBranch %48
         %49 = OpLabel
         %60 = OpLoad %6 %45
               OpStore %47 %60
         %43 = OpLoad %6 %47
         %44 = OpIAdd %6 %41 %43
               OpStore %32 %44
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());

  // Initially there are two opportunities.
  ASSERT_EQ(2, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
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
          %8 = OpTypeFunction %6 %7
         %13 = OpConstant %6 0
         %22 = OpTypeBool
         %25 = OpConstant %6 1
         %39 = OpConstant %6 100
         %61 = OpConstantTrue %22
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %45 = OpVariable %7 Function
         %46 = OpVariable %7 Function
         %47 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %42 = OpVariable %7 Function
               OpStore %32 %13
               OpBranch %33
         %33 = OpLabel
               OpSelectionMerge %35 None
               OpBranchConditional %61 %37 %35
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %40 = OpSLessThan %22 %38 %39
               OpBranchConditional %40 %34 %35
         %34 = OpLabel
               OpBranch %35
         %36 = OpLabel
         %41 = OpLoad %6 %32
               OpStore %42 %25
               OpStore %45 %13
               OpStore %46 %13
               OpBranch %48
         %48 = OpLabel
               OpLoopMerge %49 %50 None
               OpBranch %51
         %51 = OpLabel
         %52 = OpLoad %6 %46
         %53 = OpLoad %6 %42
         %54 = OpSLessThan %22 %52 %53
               OpBranchConditional %54 %55 %49
         %55 = OpLabel
         %56 = OpLoad %6 %45
         %57 = OpIAdd %6 %56 %25
               OpStore %45 %57
               OpBranch %50
         %50 = OpLabel
         %58 = OpLoad %6 %46
         %59 = OpIAdd %6 %58 %25
               OpStore %46 %59
               OpBranch %48
         %49 = OpLabel
         %60 = OpLoad %6 %45
               OpStore %47 %60
         %43 = OpLoad %6 %47
         %44 = OpIAdd %6 %41 %43
               OpStore %32 %44
               OpBranch %33
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  // Applying the first opportunity has killed the second opportunity, because
  // there was a loop embedded in the continue target of the loop we have just
  // eliminated; the continue-embedded loop is now unreachable.
  ASSERT_FALSE(ops[1]->PreconditionHolds());
}

TEST(CutLoopReductionPassTest, ConditionalBreak1) {
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
         %10 = OpTypeBool
         %11 = OpConstantFalse %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpLoopMerge %8 %9 None
               OpBranch %7
          %7 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %11 %12 %13
         %12 = OpLabel
               OpBranch %8
         %13 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpBranchConditional %11 %6 %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeBool
         %11 = OpConstantFalse %10
         %14 = OpConstantTrue %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpSelectionMerge %8 None
               OpBranchConditional %14 %7 %8
          %7 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %11 %12 %13
         %12 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpBranch %8
          %9 = OpLabel
               OpBranchConditional %11 %6 %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
}

TEST(CutLoopReductionPassTest, ConditionalBreak2) {
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
         %10 = OpTypeBool
         %11 = OpConstantFalse %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpLoopMerge %8 %9 None
               OpBranch %7
          %7 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %11 %8 %13
         %13 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpBranchConditional %11 %6 %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeBool
         %11 = OpConstantFalse %10
         %14 = OpConstantTrue %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpSelectionMerge %8 None
               OpBranchConditional %14 %7 %8
          %7 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %11 %13 %13
         %13 = OpLabel
               OpBranch %8
          %9 = OpLabel
               OpBranchConditional %11 %6 %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
}

TEST(CutLoopReductionPassTest, UnconditionalBreak) {
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
               OpLoopMerge %8 %9 None
               OpBranch %7
          %7 = OpLabel
               OpBranch %8
          %9 = OpLabel
               OpBranch %6
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeBool
         %11 = OpConstantTrue %10
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpSelectionMerge %8 None
               OpBranchConditional %11 %7 %8
          %7 = OpLabel
               OpBranch %8
          %9 = OpLabel
               OpBranch %6
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
}

TEST(CutLoopReductionPassTest, Complex) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
          %9 = OpTypePointer Function %8
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %19 = OpTypePointer Function %10
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %25 = OpVariable %9 Function
         %26 = OpVariable %9 Function
         %27 = OpVariable %9 Function
         %28 = OpVariable %9 Function
         %29 = OpVariable %9 Function
         %30 = OpVariable %19 Function
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
               OpStore %25 %33
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
               OpStore %26 %36
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
               OpStore %27 %39
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpStore %28 %42
         %43 = OpLoad %8 %25
               OpStore %29 %43
               OpStore %30 %12
               OpBranch %44
         %44 = OpLabel
               OpLoopMerge %45 %46 None
               OpBranch %47
         %47 = OpLabel
         %48 = OpLoad %8 %29
               OpBranchConditional %48 %49 %45
         %49 = OpLabel
         %50 = OpLoad %8 %25
               OpSelectionMerge %51 None
               OpBranchConditional %50 %52 %51
         %52 = OpLabel
         %53 = OpLoad %8 %26
               OpStore %29 %53
         %54 = OpLoad %10 %30
         %55 = OpIAdd %10 %54 %16
               OpStore %30 %55
               OpBranch %51
         %51 = OpLabel
         %56 = OpLoad %8 %26
               OpSelectionMerge %57 None
               OpBranchConditional %56 %58 %57
         %58 = OpLabel
         %59 = OpLoad %10 %30
         %60 = OpIAdd %10 %59 %16
               OpStore %30 %60
         %61 = OpLoad %8 %29
         %62 = OpLoad %8 %25
         %63 = OpLogicalOr %8 %61 %62
               OpStore %29 %63
         %64 = OpLoad %8 %27
               OpSelectionMerge %65 None
               OpBranchConditional %64 %66 %65
         %66 = OpLabel
         %67 = OpLoad %10 %30
         %68 = OpIAdd %10 %67 %17
               OpStore %30 %68
         %69 = OpLoad %8 %29
         %70 = OpLogicalNot %8 %69
               OpStore %29 %70
               OpBranch %46
         %65 = OpLabel
         %71 = OpLoad %8 %29
         %72 = OpLogicalOr %8 %71 %20
               OpStore %29 %72
               OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
               OpLoopMerge %74 %75 None
               OpBranch %76
         %76 = OpLabel
         %77 = OpLoad %8 %28
               OpSelectionMerge %78 None
               OpBranchConditional %77 %79 %80
         %79 = OpLabel
         %81 = OpLoad %10 %30
               OpSelectionMerge %82 None
               OpSwitch %81 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %86 = OpLoad %8 %29
         %87 = OpSelect %10 %86 %16 %17
         %88 = OpLoad %10 %30
         %89 = OpIAdd %10 %88 %87
               OpStore %30 %89
               OpBranch %82
         %85 = OpLabel
               OpBranch %75
         %82 = OpLabel
         %90 = OpLoad %8 %27
               OpSelectionMerge %91 None
               OpBranchConditional %90 %92 %91
         %92 = OpLabel
               OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %74
         %78 = OpLabel
               OpBranch %75
         %75 = OpLabel
         %93 = OpLoad %8 %29
               OpBranchConditional %93 %73 %74
         %74 = OpLabel
               OpBranch %46
         %46 = OpLabel
               OpBranch %44
         %45 = OpLabel
         %94 = OpLoad %10 %30
         %95 = OpConvertSToF %21 %94
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());

  ASSERT_EQ(2, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
          %9 = OpTypePointer Function %8
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %19 = OpTypePointer Function %10
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
         %97 = OpConstantTrue %8
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %25 = OpVariable %9 Function
         %26 = OpVariable %9 Function
         %27 = OpVariable %9 Function
         %28 = OpVariable %9 Function
         %29 = OpVariable %9 Function
         %30 = OpVariable %19 Function
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
               OpStore %25 %33
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
               OpStore %26 %36
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
               OpStore %27 %39
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpStore %28 %42
         %43 = OpLoad %8 %25
               OpStore %29 %43
               OpStore %30 %12
               OpBranch %44
         %44 = OpLabel
               OpSelectionMerge %45 None ; Was OpLoopMerge %45 %46 None
               OpBranchConditional %97 %47 %45		 ; Was OpBranch %47
         %47 = OpLabel
         %48 = OpLoad %8 %29
               OpBranchConditional %48 %49 %45
         %49 = OpLabel
         %50 = OpLoad %8 %25
               OpSelectionMerge %51 None
               OpBranchConditional %50 %52 %51
         %52 = OpLabel
         %53 = OpLoad %8 %26
               OpStore %29 %53
         %54 = OpLoad %10 %30
         %55 = OpIAdd %10 %54 %16
               OpStore %30 %55
               OpBranch %51
         %51 = OpLabel
         %56 = OpLoad %8 %26
               OpSelectionMerge %57 None
               OpBranchConditional %56 %58 %57
         %58 = OpLabel
         %59 = OpLoad %10 %30
         %60 = OpIAdd %10 %59 %16
               OpStore %30 %60
         %61 = OpLoad %8 %29
         %62 = OpLoad %8 %25
         %63 = OpLogicalOr %8 %61 %62
               OpStore %29 %63
         %64 = OpLoad %8 %27
               OpSelectionMerge %65 None
               OpBranchConditional %64 %66 %65
         %66 = OpLabel
         %67 = OpLoad %10 %30
         %68 = OpIAdd %10 %67 %17
               OpStore %30 %68
         %69 = OpLoad %8 %29
         %70 = OpLogicalNot %8 %69
               OpStore %29 %70
               OpBranch %65 	; Was OpBranch %46
         %65 = OpLabel
         %71 = OpLoad %8 %29
         %72 = OpLogicalOr %8 %71 %20
               OpStore %29 %72
               OpBranch %57 	; Was OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
               OpLoopMerge %74 %75 None
               OpBranch %76
         %76 = OpLabel
         %77 = OpLoad %8 %28
               OpSelectionMerge %78 None
               OpBranchConditional %77 %79 %80
         %79 = OpLabel
         %81 = OpLoad %10 %30
               OpSelectionMerge %82 None
               OpSwitch %81 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %86 = OpLoad %8 %29
         %87 = OpSelect %10 %86 %16 %17
         %88 = OpLoad %10 %30
         %89 = OpIAdd %10 %88 %87
               OpStore %30 %89
               OpBranch %82
         %85 = OpLabel
               OpBranch %75
         %82 = OpLabel
         %90 = OpLoad %8 %27
               OpSelectionMerge %91 None
               OpBranchConditional %90 %92 %91
         %92 = OpLabel
               OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %74
         %78 = OpLabel
               OpBranch %75
         %75 = OpLabel
         %93 = OpLoad %8 %29
               OpBranchConditional %93 %73 %74
         %74 = OpLabel
               OpBranch %45 	; Was OpBranch %46
         %46 = OpLabel
               OpBranch %44
         %45 = OpLabel
         %94 = OpLoad %10 %30
         %95 = OpConvertSToF %21 %94
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());
  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());

  std::string after_op_1 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
          %9 = OpTypePointer Function %8
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %19 = OpTypePointer Function %10
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
         %97 = OpConstantTrue %8
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %25 = OpVariable %9 Function
         %26 = OpVariable %9 Function
         %27 = OpVariable %9 Function
         %28 = OpVariable %9 Function
         %29 = OpVariable %9 Function
         %30 = OpVariable %19 Function
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
               OpStore %25 %33
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
               OpStore %26 %36
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
               OpStore %27 %39
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpStore %28 %42
         %43 = OpLoad %8 %25
               OpStore %29 %43
               OpStore %30 %12
               OpBranch %44
         %44 = OpLabel
               OpSelectionMerge %45 None ; Was OpLoopMerge %45 %46 None
               OpBranchConditional %97 %47 %45		 ; Was OpBranch %47
         %47 = OpLabel
         %48 = OpLoad %8 %29
               OpBranchConditional %48 %49 %45
         %49 = OpLabel
         %50 = OpLoad %8 %25
               OpSelectionMerge %51 None
               OpBranchConditional %50 %52 %51
         %52 = OpLabel
         %53 = OpLoad %8 %26
               OpStore %29 %53
         %54 = OpLoad %10 %30
         %55 = OpIAdd %10 %54 %16
               OpStore %30 %55
               OpBranch %51
         %51 = OpLabel
         %56 = OpLoad %8 %26
               OpSelectionMerge %57 None
               OpBranchConditional %56 %58 %57
         %58 = OpLabel
         %59 = OpLoad %10 %30
         %60 = OpIAdd %10 %59 %16
               OpStore %30 %60
         %61 = OpLoad %8 %29
         %62 = OpLoad %8 %25
         %63 = OpLogicalOr %8 %61 %62
               OpStore %29 %63
         %64 = OpLoad %8 %27
               OpSelectionMerge %65 None
               OpBranchConditional %64 %66 %65
         %66 = OpLabel
         %67 = OpLoad %10 %30
         %68 = OpIAdd %10 %67 %17
               OpStore %30 %68
         %69 = OpLoad %8 %29
         %70 = OpLogicalNot %8 %69
               OpStore %29 %70
               OpBranch %65 	; Was OpBranch %46
         %65 = OpLabel
         %71 = OpLoad %8 %29
         %72 = OpLogicalOr %8 %71 %20
               OpStore %29 %72
               OpBranch %57 	; Was OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
               OpSelectionMerge %74 None ; Was OpLoopMerge %74 %75 None
               OpBranchConditional %97 %76 %74 ; Was OpBranch %76
         %76 = OpLabel
         %77 = OpLoad %8 %28
               OpSelectionMerge %78 None
               OpBranchConditional %77 %79 %80
         %79 = OpLabel
         %81 = OpLoad %10 %30
               OpSelectionMerge %82 None
               OpSwitch %81 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %86 = OpLoad %8 %29
         %87 = OpSelect %10 %86 %16 %17
         %88 = OpLoad %10 %30
         %89 = OpIAdd %10 %88 %87
               OpStore %30 %89
               OpBranch %82
         %85 = OpLabel
               OpBranch %82
         %82 = OpLabel
         %90 = OpLoad %8 %27
               OpSelectionMerge %91 None
               OpBranchConditional %90 %92 %91
         %92 = OpLabel
               OpBranch %91
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %78 ; Was OpBranch %74
         %78 = OpLabel
               OpBranch %74
         %75 = OpLabel
         %93 = OpLoad %8 %29
               OpBranchConditional %93 %73 %45 ; Was OpBranchConditional %93 %73 %74
         %74 = OpLabel
               OpBranch %45 	; Was OpBranch %46
         %46 = OpLabel
               OpBranch %44
         %45 = OpLabel
         %94 = OpLoad %10 %30
         %95 = OpConvertSToF %21 %94
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_1, context.get());
}

TEST(CutLoopReductionPassTest, ComplexOptimized) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpBranch %44
         %44 = OpLabel
         %98 = OpPhi %10 %12 %24 %107 %46
         %97 = OpPhi %8 %33 %24 %105 %46
               OpLoopMerge %45 %46 None
               OpBranchConditional %97 %49 %45
         %49 = OpLabel
               OpSelectionMerge %51 None
               OpBranchConditional %33 %52 %51
         %52 = OpLabel
         %55 = OpIAdd %10 %98 %16
               OpBranch %51
         %51 = OpLabel
        %100 = OpPhi %10 %98 %49 %55 %52
        %113 = OpSelect %8 %33 %36 %97
               OpSelectionMerge %57 None
               OpBranchConditional %36 %58 %57
         %58 = OpLabel
         %60 = OpIAdd %10 %100 %16
         %63 = OpLogicalOr %8 %113 %33
               OpSelectionMerge %65 None
               OpBranchConditional %39 %66 %65
         %66 = OpLabel
         %68 = OpIAdd %10 %100 %18
         %70 = OpLogicalNot %8 %63
               OpBranch %46
         %65 = OpLabel
         %72 = OpLogicalOr %8 %63 %20
               OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
         %99 = OpPhi %10 %100 %57 %109 %75
               OpLoopMerge %74 %75 None
               OpBranch %76
         %76 = OpLabel
               OpSelectionMerge %78 None
               OpBranchConditional %42 %79 %80
         %79 = OpLabel
               OpSelectionMerge %82 None
               OpSwitch %99 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %87 = OpSelect %10 %113 %16 %17
         %89 = OpIAdd %10 %99 %87
               OpBranch %82
         %85 = OpLabel
               OpBranch %75
         %82 = OpLabel
        %110 = OpPhi %10 %99 %83 %89 %84
               OpSelectionMerge %91 None
               OpBranchConditional %39 %92 %91
         %92 = OpLabel
               OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %74
         %78 = OpLabel
               OpBranch %75
         %75 = OpLabel
        %109 = OpPhi %10 %99 %85 %110 %92 %110 %78
               OpBranchConditional %113 %73 %74
         %74 = OpLabel
        %108 = OpPhi %10 %99 %80 %109 %75
               OpBranch %46
         %46 = OpLabel
        %107 = OpPhi %10 %68 %66 %60 %65 %108 %74
        %105 = OpPhi %8 %70 %66 %72 %65 %113 %74
               OpBranch %44
         %45 = OpLabel
         %95 = OpConvertSToF %21 %98
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto context = BuildModule(env, nullptr, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());

  ASSERT_EQ(2, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpBranch %44
         %44 = OpLabel
         %98 = OpPhi %10 %12 %24 %107 %46
         %97 = OpPhi %8 %33 %24 %105 %46
               OpSelectionMerge %45 None	; Was OpLoopMerge %45 %46 None
               OpBranchConditional %97 %49 %45
         %49 = OpLabel
               OpSelectionMerge %51 None
               OpBranchConditional %33 %52 %51
         %52 = OpLabel
         %55 = OpIAdd %10 %98 %16
               OpBranch %51
         %51 = OpLabel
        %100 = OpPhi %10 %98 %49 %55 %52
        %113 = OpSelect %8 %33 %36 %97
               OpSelectionMerge %57 None
               OpBranchConditional %36 %58 %57
         %58 = OpLabel
         %60 = OpIAdd %10 %100 %16
         %63 = OpLogicalOr %8 %113 %33
               OpSelectionMerge %65 None
               OpBranchConditional %39 %66 %65
         %66 = OpLabel
         %68 = OpIAdd %10 %100 %18
         %70 = OpLogicalNot %8 %63
               OpBranch %65 	; Was OpBranch %46
         %65 = OpLabel
         %72 = OpLogicalOr %8 %63 %20
               OpBranch %57     ; Was OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
         %99 = OpPhi %10 %100 %57 %109 %75
               OpLoopMerge %74 %75 None
               OpBranch %76
         %76 = OpLabel
               OpSelectionMerge %78 None
               OpBranchConditional %42 %79 %80
         %79 = OpLabel
               OpSelectionMerge %82 None
               OpSwitch %99 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %87 = OpSelect %10 %113 %16 %17
         %89 = OpIAdd %10 %99 %87
               OpBranch %82
         %85 = OpLabel
               OpBranch %75
         %82 = OpLabel
        %110 = OpPhi %10 %99 %83 %89 %84
               OpSelectionMerge %91 None
               OpBranchConditional %39 %92 %91
         %92 = OpLabel
               OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %74
         %78 = OpLabel
               OpBranch %75
         %75 = OpLabel
        %109 = OpPhi %10 %99 %85 %110 %92 %110 %78
               OpBranchConditional %113 %73 %74
         %74 = OpLabel
        %108 = OpPhi %10 %99 %80 %109 %75
               OpBranch %45 	; Was OpBranch %46
         %46 = OpLabel
        %107 = OpUndef %10      ; Was OpPhi %10 %68 %66 %60 %65 %108 %74
        %105 = OpUndef %8       ; Was OpPhi %8 %70 %66 %72 %65 %113 %74
               OpBranch %44
         %45 = OpLabel
         %95 = OpConvertSToF %21 %98
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_1 = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %4 0 Offset 0
               OpMemberDecorate %4 1 Offset 4
               OpMemberDecorate %4 2 Offset 8
               OpMemberDecorate %4 3 Offset 12
               OpDecorate %4 Block
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 0
               OpDecorate %3 Location 0
          %6 = OpTypeVoid
          %7 = OpTypeFunction %6
          %8 = OpTypeBool
         %10 = OpTypeInt 32 1
          %4 = OpTypeStruct %10 %10 %10 %10
         %11 = OpTypePointer Uniform %4
          %5 = OpVariable %11 Uniform
         %12 = OpConstant %10 0
         %13 = OpTypePointer Uniform %10
         %14 = OpTypeInt 32 0
         %15 = OpConstant %14 0
         %16 = OpConstant %10 1
         %17 = OpConstant %10 2
         %18 = OpConstant %10 3
         %20 = OpConstantFalse %8
         %21 = OpTypeFloat 32
         %22 = OpTypeVector %21 4
         %23 = OpTypePointer Output %22
          %3 = OpVariable %23 Output
        %114 = OpUndef %10
        %115 = OpConstantTrue %8
          %2 = OpFunction %6 None %7
         %24 = OpLabel
         %31 = OpAccessChain %13 %5 %12
         %32 = OpLoad %10 %31
         %33 = OpINotEqual %8 %32 %15
         %34 = OpAccessChain %13 %5 %16
         %35 = OpLoad %10 %34
         %36 = OpINotEqual %8 %35 %15
         %37 = OpAccessChain %13 %5 %17
         %38 = OpLoad %10 %37
         %39 = OpINotEqual %8 %38 %15
         %40 = OpAccessChain %13 %5 %18
         %41 = OpLoad %10 %40
         %42 = OpINotEqual %8 %41 %15
               OpBranch %44
         %44 = OpLabel
         %98 = OpPhi %10 %12 %24 %107 %46
         %97 = OpPhi %8 %33 %24 %105 %46
               OpSelectionMerge %45 None	; Was OpLoopMerge %45 %46 None
               OpBranchConditional %97 %49 %45
         %49 = OpLabel
               OpSelectionMerge %51 None
               OpBranchConditional %33 %52 %51
         %52 = OpLabel
         %55 = OpIAdd %10 %98 %16
               OpBranch %51
         %51 = OpLabel
        %100 = OpPhi %10 %98 %49 %55 %52
        %113 = OpSelect %8 %33 %36 %97
               OpSelectionMerge %57 None
               OpBranchConditional %36 %58 %57
         %58 = OpLabel
         %60 = OpIAdd %10 %100 %16
         %63 = OpLogicalOr %8 %113 %33
               OpSelectionMerge %65 None
               OpBranchConditional %39 %66 %65
         %66 = OpLabel
         %68 = OpIAdd %10 %100 %18
         %70 = OpLogicalNot %8 %63
               OpBranch %65 	; Was OpBranch %46
         %65 = OpLabel
         %72 = OpLogicalOr %8 %63 %20
               OpBranch %57     ; Was OpBranch %46
         %57 = OpLabel
               OpBranch %73
         %73 = OpLabel
         %99 = OpPhi %10 %100 %57 %109 %75
               OpSelectionMerge %74 None ; Was OpLoopMerge %74 %75 None
               OpBranchConditional %115 %76 %74
         %76 = OpLabel
               OpSelectionMerge %78 None
               OpBranchConditional %42 %79 %80
         %79 = OpLabel
               OpSelectionMerge %82 None
               OpSwitch %99 %83 1 %84 2 %85
         %83 = OpLabel
               OpBranch %82
         %84 = OpLabel
         %87 = OpSelect %10 %113 %16 %17
         %89 = OpIAdd %10 %99 %87
               OpBranch %82
         %85 = OpLabel
               OpBranch %82 	; Was OpBranch %75
         %82 = OpLabel
        %110 = OpPhi %10 %99 %83 %89 %84 %114 %85 ; Was OpPhi %10 %99 %83 %89 %84
               OpSelectionMerge %91 None
               OpBranchConditional %39 %92 %91
         %92 = OpLabel
               OpBranch %91 	; OpBranch %75
         %91 = OpLabel
               OpBranch %78
         %80 = OpLabel
               OpBranch %78 	; Was OpBranch %74
         %78 = OpLabel
               OpBranch %74     ; Was OpBranch %75
         %75 = OpLabel
        %109 = OpUndef %10 ; Was OpPhi %10 %99 %85 %110 %92 %110 %78
               OpBranchConditional %113 %73 %45 ; Was OpBranchConditional %113 %73 %74
         %74 = OpLabel
        %108 = OpPhi %10 %114 %78 %114 %73 ; Was OpPhi %10 %99 %80 %109 %75
               OpBranch %45 	; Was OpBranch %46
         %46 = OpLabel
        %107 = OpUndef %10      ; Was OpPhi %10 %68 %66 %60 %65 %108 %74
        %105 = OpUndef %8       ; Was OpPhi %8 %70 %66 %72 %65 %113 %74
               OpBranch %44
         %45 = OpLabel
         %95 = OpConvertSToF %21 %98
         %96 = OpCompositeConstruct %22 %95 %95 %95 %95
               OpStore %3 %96
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_1, context.get());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
