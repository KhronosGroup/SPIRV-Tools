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

#include "reduce_test_util.h"
#include "source/opt/build_module.h"
#include "source/reduce/cut_loop_reduction_pass.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(CutLoopReductionPassTest, LoopyShader1) {

  std::string prologue = R"(
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
  )";

  std::string shader = prologue + R"(
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
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = prologue + R"(
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %12
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

  std::string prologue = R"(
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
  )";

  std::string shader = prologue + R"(
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
  const auto consumer = nullptr;
  const auto context =
        BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(4, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = prologue + R"(
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
  std::string after_op_1 = prologue + R"(
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %22
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
  std::string after_op_2 = prologue + R"(
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %22
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
  std::string after_op_3 = prologue + R"(
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %6 %8
         %18 = OpSLessThan %17 %15 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %6 %19
         %26 = OpSLessThan %17 %25 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
               OpBranch %22
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
               OpBranch %37
         %37 = OpLabel
         %38 = OpLoad %6 %32
         %39 = OpSLessThan %17 %38 %16
               OpBranchConditional %39 %34 %35
         %34 = OpLabel
               OpStore %40 %9
               OpBranch %41
         %41 = OpLabel
               OpBranch %45
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
  std::string prologue = R"(
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
  )";

  std::string shader = prologue + R"(
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
               OpReturn
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
  const auto consumer = nullptr;
  const auto context =
        BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(2, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = prologue + R"(
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
               OpReturn
         %37 = OpLabel
               OpBranch %31
         %31 = OpLabel
         %39 = OpLoad %6 %8
         %41 = OpSGreaterThan %17 %39 %40
               OpBranchConditional %41 %28 %30
         %30 = OpLabel
               OpBranch %12
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_0, context.get());

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_1 = prologue + R"(
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
               OpReturn
         %37 = OpLabel
               OpBranch %30
         %39 = OpLoad %6 %8
         %41 = OpSGreaterThan %17 %39 %40
               OpBranchConditional %41 %28 %30
         %30 = OpLabel
               OpBranch %12
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_op_1, context.get());

}

TEST(CutLoopReductionPassTest, LoopyShader4) {
  std::string prologue = R"(
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
  )";

  std::string shader = prologue + R"(
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
  const auto consumer = nullptr;
  const auto context =
        BuildModule(env, consumer, shader, kReduceAssembleOption);
  const auto pass = TestSubclass<CutLoopReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(2, ops.size());

  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_0 = prologue + R"(
               OpBranch %37
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

  ASSERT_TRUE(ops[1]->PreconditionHolds());
  ops[1]->TryToApply();
  CheckValid(env, context.get());
  std::string after_op_1 = prologue + R"(
               OpBranch %37
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
               OpBranch %49
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
  CheckEqual(env, after_op_1, context.get());

}


}  // namespace
}  // namespace reduce
}  // namespace spvtools
