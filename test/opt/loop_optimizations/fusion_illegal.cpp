// Copyright (c) 2018 Google LLC.
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

#include <gmock/gmock.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "../pass_fixture.h"
#include "opt/loop_descriptor.h"
#include "opt/loop_fusion.h"

namespace {

using namespace spvtools;

using FusionIllegalTest = PassTest<::testing::Test>;

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Illegal, loop-independent dependence will become a
  // backward loop-carried antidependence
  for (int i = 0; i < 10; i++) {
    a[i] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    c[i] = a[i+1] + 2;
  }
}

*/
TEST_F(FusionIllegalTest, PositiveDistanceCreatedRAW) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %25 "b"
               OpName %34 "i"
               OpName %42 "c"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 10
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %29 = OpConstant %6 1
         %48 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %42 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %53 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %53 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %53
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %28 %29
         %31 = OpAccessChain %7 %23 %53
               OpStore %31 %30
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %53 %29
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %54 = OpPhi %6 %9 %12 %52 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %54 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpIAdd %6 %54 %29
         %46 = OpAccessChain %7 %23 %45
         %47 = OpLoad %6 %46
         %49 = OpIAdd %6 %47 %48
         %50 = OpAccessChain %7 %42 %54
               OpStore %50 %49
               OpBranch %38
         %38 = OpLabel
         %52 = OpIAdd %6 %54 %29
               OpStore %34 %52
               OpBranch %35
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();
  ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInOrderOfAppearance();

  opt::LoopFusion fusion(context.get(), loops[0], loops[1]);

  EXPECT_TRUE(fusion.AreCompatible());
  EXPECT_FALSE(fusion.IsLegal());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core

int func() {
  return 10;
}

void main() {
  int[10] a;
  int[10] b;
  // Illegal, function call
  for (int i = 0; i < 10; i++) {
    a[i] = func();
  }
  for (int i = 0; i < 10; i++) {
    b[i] = a[i];
  }
}
*/
TEST_F(FusionIllegalTest, FunctionCall) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "func("
               OpName %14 "i"
               OpName %28 "a"
               OpName %35 "i"
               OpName %43 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeFunction %6
         %10 = OpConstant %6 10
         %13 = OpTypePointer Function %6
         %15 = OpConstant %6 0
         %22 = OpTypeBool
         %24 = OpTypeInt 32 0
         %25 = OpConstant %24 10
         %26 = OpTypeArray %6 %25
         %27 = OpTypePointer Function %26
         %33 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpVariable %13 Function
         %28 = OpVariable %27 Function
         %35 = OpVariable %13 Function
         %43 = OpVariable %27 Function
               OpStore %14 %15
               OpBranch %16
         %16 = OpLabel
         %51 = OpPhi %6 %15 %5 %34 %19
               OpLoopMerge %18 %19 None
               OpBranch %20
         %20 = OpLabel
         %23 = OpSLessThan %22 %51 %10
               OpBranchConditional %23 %17 %18
         %17 = OpLabel
         %30 = OpFunctionCall %6 %8
         %31 = OpAccessChain %13 %28 %51
               OpStore %31 %30
               OpBranch %19
         %19 = OpLabel
         %34 = OpIAdd %6 %51 %33
               OpStore %14 %34
               OpBranch %16
         %18 = OpLabel
               OpStore %35 %15
               OpBranch %36
         %36 = OpLabel
         %52 = OpPhi %6 %15 %18 %50 %39
               OpLoopMerge %38 %39 None
               OpBranch %40
         %40 = OpLabel
         %42 = OpSLessThan %22 %52 %10
               OpBranchConditional %42 %37 %38
         %37 = OpLabel
         %46 = OpAccessChain %13 %28 %52
         %47 = OpLoad %6 %46
         %48 = OpAccessChain %13 %43 %52
               OpStore %48 %47
               OpBranch %39
         %39 = OpLabel
         %50 = OpIAdd %6 %52 %33
               OpStore %35 %50
               OpBranch %36
         %38 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %6 None %7
          %9 = OpLabel
               OpReturnValue %10
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();
  ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
  EXPECT_EQ(ld.NumLoops(), 2u);

  auto loops = ld.GetLoopsInOrderOfAppearance();

  opt::LoopFusion fusion(context.get(), loops[0], loops[1]);

  EXPECT_TRUE(fusion.AreCompatible());
  EXPECT_FALSE(fusion.IsLegal());
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 16
#version 440 core
void main() {
  int[10][10] a;
  int[10][10] b;
  int[10][10] c;
  // Illegal outer.
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      c[i][j] = a[i][j] + 2;
    }
  }
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      b[i][j] = c[i+1][j] + 10;
    }
  }
}

*/
TEST_F(FusionIllegalTest, PositiveDistanceCreatedRAWOuterLoop) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %19 "j"
               OpName %32 "c"
               OpName %35 "a"
               OpName %48 "i"
               OpName %56 "j"
               OpName %64 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %6 %28
         %30 = OpTypeArray %29 %28
         %31 = OpTypePointer Function %30
         %40 = OpConstant %6 2
         %44 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %19 = OpVariable %7 Function
         %32 = OpVariable %31 Function
         %35 = OpVariable %31 Function
         %48 = OpVariable %7 Function
         %56 = OpVariable %7 Function
         %64 = OpVariable %31 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %78 = OpPhi %6 %9 %5 %47 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %78 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %19 %9
               OpBranch %20
         %20 = OpLabel
         %82 = OpPhi %6 %9 %11 %45 %23
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %26 = OpSLessThan %17 %82 %16
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
         %38 = OpAccessChain %7 %35 %78 %82
         %39 = OpLoad %6 %38
         %41 = OpIAdd %6 %39 %40
         %42 = OpAccessChain %7 %32 %78 %82
               OpStore %42 %41
               OpBranch %23
         %23 = OpLabel
         %45 = OpIAdd %6 %82 %44
               OpStore %19 %45
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %47 = OpIAdd %6 %78 %44
               OpStore %8 %47
               OpBranch %10
         %12 = OpLabel
               OpStore %48 %9
               OpBranch %49
         %49 = OpLabel
         %79 = OpPhi %6 %9 %12 %77 %52
               OpLoopMerge %51 %52 None
               OpBranch %53
         %53 = OpLabel
         %55 = OpSLessThan %17 %79 %16
               OpBranchConditional %55 %50 %51
         %50 = OpLabel
               OpStore %56 %9
               OpBranch %57
         %57 = OpLabel
         %80 = OpPhi %6 %9 %50 %75 %60
               OpLoopMerge %59 %60 None
               OpBranch %61
         %61 = OpLabel
         %63 = OpSLessThan %17 %80 %16
               OpBranchConditional %63 %58 %59
         %58 = OpLabel
         %68 = OpIAdd %6 %79 %44
         %70 = OpAccessChain %7 %32 %68 %80
         %71 = OpLoad %6 %70
         %72 = OpIAdd %6 %71 %16
         %73 = OpAccessChain %7 %64 %79 %80
               OpStore %73 %72
               OpBranch %60
         %60 = OpLabel
         %75 = OpIAdd %6 %80 %44
               OpStore %56 %75
               OpBranch %57
         %59 = OpLabel
               OpBranch %52
         %52 = OpLabel
         %77 = OpIAdd %6 %79 %44
               OpStore %48 %77
               OpBranch %49
         %51 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();

  {
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 4u);

    auto loops = ld.GetLoopsInOrderOfAppearance();

    auto loop_0 = loops[0];
    auto loop_1 = loops[1];
    auto loop_2 = loops[2];
    auto loop_3 = loops[3];

    {
      opt::LoopFusion fusion(context.get(), loop_0, loop_1);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      opt::LoopFusion fusion(context.get(), loop_0, loop_2);
      EXPECT_TRUE(fusion.AreCompatible());
      EXPECT_FALSE(fusion.IsLegal());
    }

    {
      opt::LoopFusion fusion(context.get(), loop_1, loop_2);
      EXPECT_FALSE(fusion.AreCompatible());
    }

    {
      opt::LoopFusion fusion(context.get(), loop_2, loop_3);
      EXPECT_FALSE(fusion.AreCompatible());
    }
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 19
#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Illegal, would create a backward loop-carried anti-dependence.
  for (int i = 0; i < 10; i++) {
    c[i] = a[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    a[i+1] = c[i] + 2;
  }
}

*/
TEST_F(FusionIllegalTest, PositiveDistanceCreatedWAR) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "c"
               OpName %25 "a"
               OpName %34 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 10
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %29 = OpConstant %6 1
         %47 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %34 = OpVariable %7 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %52 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %52 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %52
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %28 %29
         %31 = OpAccessChain %7 %23 %52
               OpStore %31 %30
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %52 %29
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %53 = OpPhi %6 %9 %12 %51 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %53 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %43 = OpIAdd %6 %53 %29
         %45 = OpAccessChain %7 %23 %53
         %46 = OpLoad %6 %45
         %48 = OpIAdd %6 %46 %47
         %49 = OpAccessChain %7 %25 %43
               OpStore %49 %48
               OpBranch %38
         %38 = OpLabel
         %51 = OpIAdd %6 %53 %29
               OpStore %34 %51
               OpBranch %35
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();

  {
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInOrderOfAppearance();

    opt::LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_FALSE(fusion.IsLegal());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 21
#version 440 core
void main() {
  int[10] a;
  int[10] b;
  int[10] c;
  // Illegal, would create a backward loop-carried anti-dependence.
  for (int i = 0; i < 10; i++) {
    a[i] = b[i] + 1;
  }
  for (int i = 0; i < 10; i++) {
    a[i+1] = c[i+1] + 2;
  }
}

*/
TEST_F(FusionIllegalTest, PositiveDistanceCreatedWAW) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %25 "b"
               OpName %34 "i"
               OpName %44 "c"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 10
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %29 = OpConstant %6 1
         %49 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %25 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %44 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %54 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %54 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpAccessChain %7 %25 %54
         %28 = OpLoad %6 %27
         %30 = OpIAdd %6 %28 %29
         %31 = OpAccessChain %7 %23 %54
               OpStore %31 %30
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %54 %29
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %55 = OpPhi %6 %9 %12 %53 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %55 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %43 = OpIAdd %6 %55 %29
         %46 = OpIAdd %6 %55 %29
         %47 = OpAccessChain %7 %44 %46
         %48 = OpLoad %6 %47
         %50 = OpIAdd %6 %48 %49
         %51 = OpAccessChain %7 %23 %43
               OpStore %51 %50
               OpBranch %38
         %38 = OpLabel
         %53 = OpIAdd %6 %55 %29
               OpStore %34 %53
               OpBranch %35
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();

  {
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInOrderOfAppearance();

    opt::LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_FALSE(fusion.IsLegal());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 28
#version 440 core
void main() {
  int[10] a;
  int[10] b;

  int sum_0 = 0;

  // Illegal
  for (int i = 0; i < 10; i++) {
    sum_0 += a[i];
  }
  for (int j = 0; j < 10; j++) {
    sum_0 += b[j];
  }
}

*/
TEST_F(FusionIllegalTest, SameReductionVariable) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "sum_0"
               OpName %10 "i"
               OpName %24 "a"
               OpName %33 "j"
               OpName %41 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %18 = OpTypeBool
         %20 = OpTypeInt 32 0
         %21 = OpConstant %20 10
         %22 = OpTypeArray %6 %21
         %23 = OpTypePointer Function %22
         %31 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %24 = OpVariable %23 Function
         %33 = OpVariable %7 Function
         %41 = OpVariable %23 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
         %52 = OpPhi %6 %9 %5 %29 %14
         %49 = OpPhi %6 %9 %5 %32 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %18 %49 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
         %26 = OpAccessChain %7 %24 %49
         %27 = OpLoad %6 %26
         %29 = OpIAdd %6 %52 %27
               OpStore %8 %29
               OpBranch %14
         %14 = OpLabel
         %32 = OpIAdd %6 %49 %31
               OpStore %10 %32
               OpBranch %11
         %13 = OpLabel
               OpStore %33 %9
               OpBranch %34
         %34 = OpLabel
         %51 = OpPhi %6 %52 %13 %46 %37
         %50 = OpPhi %6 %9 %13 %48 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %40 = OpSLessThan %18 %50 %17
               OpBranchConditional %40 %35 %36
         %35 = OpLabel
         %43 = OpAccessChain %7 %41 %50
         %44 = OpLoad %6 %43
         %46 = OpIAdd %6 %51 %44
               OpStore %8 %46
               OpBranch %37
         %37 = OpLabel
         %48 = OpIAdd %6 %50 %31
               OpStore %33 %48
               OpBranch %34
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();

  {
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInOrderOfAppearance();

    opt::LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_FALSE(fusion.IsLegal());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 28
#version 440 core
void main() {
  int[10] a;
  int[10] b;

  int sum_0 = 0;

  // Illegal
  for (int i = 0; i < 10; i++) {
    sum_0 += a[i];
  }
  for (int j = 0; j < 10; j++) {
    sum_0 += b[j];
  }
}

*/
TEST_F(FusionIllegalTest, SameReductionVariableLCSSA) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "sum_0"
               OpName %10 "i"
               OpName %24 "a"
               OpName %33 "j"
               OpName %41 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %18 = OpTypeBool
         %20 = OpTypeInt 32 0
         %21 = OpConstant %20 10
         %22 = OpTypeArray %6 %21
         %23 = OpTypePointer Function %22
         %31 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %24 = OpVariable %23 Function
         %33 = OpVariable %7 Function
         %41 = OpVariable %23 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
         %52 = OpPhi %6 %9 %5 %29 %14
         %49 = OpPhi %6 %9 %5 %32 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %18 %49 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
         %26 = OpAccessChain %7 %24 %49
         %27 = OpLoad %6 %26
         %29 = OpIAdd %6 %52 %27
               OpStore %8 %29
               OpBranch %14
         %14 = OpLabel
         %32 = OpIAdd %6 %49 %31
               OpStore %10 %32
               OpBranch %11
         %13 = OpLabel
               OpStore %33 %9
               OpBranch %34
         %34 = OpLabel
         %51 = OpPhi %6 %52 %13 %46 %37
         %50 = OpPhi %6 %9 %13 %48 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %40 = OpSLessThan %18 %50 %17
               OpBranchConditional %40 %35 %36
         %35 = OpLabel
         %43 = OpAccessChain %7 %41 %50
         %44 = OpLoad %6 %43
         %46 = OpIAdd %6 %51 %44
               OpStore %8 %46
               OpBranch %37
         %37 = OpLabel
         %48 = OpIAdd %6 %50 %31
               OpStore %33 %48
               OpBranch %34
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();

  {
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInOrderOfAppearance();

    opt::LoopUtils utils_0(context.get(), loops[0]);
    utils_0.MakeLoopClosedSSA();

    opt::LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_FALSE(fusion.IsLegal());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 30
#version 440 core
int x;
void main() {
  int[10] a;
  int[10] b;

  // Illegal, x is unknown.
  for (int i = 0; i < 10; i++) {
    a[x] = a[i];
  }
  for (int j = 0; j < 10; j++) {
    a[j] = b[j];
  }
}

*/
TEST_F(FusionIllegalTest, UnknownIndexVariable) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %25 "x"
               OpName %34 "j"
               OpName %43 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 10
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %24 = OpTypePointer Private %6
         %25 = OpVariable %24 Private
         %32 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %34 = OpVariable %7 Function
         %43 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %50 = OpPhi %6 %9 %5 %33 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %50 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpLoad %6 %25
         %28 = OpAccessChain %7 %23 %50
         %29 = OpLoad %6 %28
         %30 = OpAccessChain %7 %23 %26
               OpStore %30 %29
               OpBranch %13
         %13 = OpLabel
         %33 = OpIAdd %6 %50 %32
               OpStore %8 %33
               OpBranch %10
         %12 = OpLabel
               OpStore %34 %9
               OpBranch %35
         %35 = OpLabel
         %51 = OpPhi %6 %9 %12 %49 %38
               OpLoopMerge %37 %38 None
               OpBranch %39
         %39 = OpLabel
         %41 = OpSLessThan %17 %51 %16
               OpBranchConditional %41 %36 %37
         %36 = OpLabel
         %45 = OpAccessChain %7 %43 %51
         %46 = OpLoad %6 %45
         %47 = OpAccessChain %7 %23 %51
               OpStore %47 %46
               OpBranch %38
         %38 = OpLabel
         %49 = OpIAdd %6 %51 %32
               OpStore %34 %49
               OpBranch %35
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();

  {
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInOrderOfAppearance();

    opt::LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_FALSE(fusion.IsLegal());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
void main() {
  int[10] a;
  int[10] b;

  int sum = 0;

  // Illegal, accumulator used for indexing.
  for (int i = 0; i < 10; i++) {
    sum += a[i];
    b[sum] = a[i];
  }
  for (int j = 0; j < 10; j++) {
    b[j] = b[j]+1;
  }
}

*/
TEST_F(FusionIllegalTest, AccumulatorIndexing) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "sum"
               OpName %10 "i"
               OpName %24 "a"
               OpName %30 "b"
               OpName %39 "j"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %18 = OpTypeBool
         %20 = OpTypeInt 32 0
         %21 = OpConstant %20 10
         %22 = OpTypeArray %6 %21
         %23 = OpTypePointer Function %22
         %37 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %24 = OpVariable %23 Function
         %30 = OpVariable %23 Function
         %39 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
         %57 = OpPhi %6 %9 %5 %29 %14
         %55 = OpPhi %6 %9 %5 %38 %14
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %19 = OpSLessThan %18 %55 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
         %26 = OpAccessChain %7 %24 %55
         %27 = OpLoad %6 %26
         %29 = OpIAdd %6 %57 %27
               OpStore %8 %29
         %33 = OpAccessChain %7 %24 %55
         %34 = OpLoad %6 %33
         %35 = OpAccessChain %7 %30 %29
               OpStore %35 %34
               OpBranch %14
         %14 = OpLabel
         %38 = OpIAdd %6 %55 %37
               OpStore %10 %38
               OpBranch %11
         %13 = OpLabel
               OpStore %39 %9
               OpBranch %40
         %40 = OpLabel
         %56 = OpPhi %6 %9 %13 %54 %43
               OpLoopMerge %42 %43 None
               OpBranch %44
         %44 = OpLabel
         %46 = OpSLessThan %18 %56 %17
               OpBranchConditional %46 %41 %42
         %41 = OpLabel
         %49 = OpAccessChain %7 %30 %56
         %50 = OpLoad %6 %49
         %51 = OpIAdd %6 %50 %37
         %52 = OpAccessChain %7 %30 %56
               OpStore %52 %51
               OpBranch %43
         %43 = OpLabel
         %54 = OpIAdd %6 %56 %37
               OpStore %39 %54
               OpBranch %40
         %42 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();

  {
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInOrderOfAppearance();

    opt::LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_FALSE(fusion.IsLegal());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

// 33
#version 440 core
void main() {
  int[10] a;
  int[10] b;

  // Illegal, barrier.
  for (int i = 0; i < 10; i++) {
    a[i] = a[i] * 2;
    memoryBarrier();
  }
  for (int j = 0; j < 10; j++) {
    b[j] = b[j] + 1;
  }
}

*/
TEST_F(FusionIllegalTest, Barrier) {
  std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %23 "a"
               OpName %36 "j"
               OpName %44 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 10
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 10
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %28 = OpConstant %6 2
         %31 = OpConstant %19 1
         %32 = OpConstant %19 3400
         %34 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %23 = OpVariable %22 Function
         %36 = OpVariable %7 Function
         %44 = OpVariable %22 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %53 = OpPhi %6 %9 %5 %35 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %53 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpAccessChain %7 %23 %53
         %27 = OpLoad %6 %26
         %29 = OpIMul %6 %27 %28
         %30 = OpAccessChain %7 %23 %53
               OpStore %30 %29
               OpMemoryBarrier %31 %32
               OpBranch %13
         %13 = OpLabel
         %35 = OpIAdd %6 %53 %34
               OpStore %8 %35
               OpBranch %10
         %12 = OpLabel
               OpStore %36 %9
               OpBranch %37
         %37 = OpLabel
         %54 = OpPhi %6 %9 %12 %52 %40
               OpLoopMerge %39 %40 None
               OpBranch %41
         %41 = OpLabel
         %43 = OpSLessThan %17 %54 %16
               OpBranchConditional %43 %38 %39
         %38 = OpLabel
         %47 = OpAccessChain %7 %44 %54
         %48 = OpLoad %6 %47
         %49 = OpIAdd %6 %48 %34
         %50 = OpAccessChain %7 %44 %54
               OpStore %50 %49
               OpBranch %40
         %40 = OpLabel
         %52 = OpIAdd %6 %54 %34
               OpStore %36 %52
               OpBranch %37
         %39 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  ir::Function& f = *module->begin();

  {
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);
    EXPECT_EQ(ld.NumLoops(), 2u);

    auto loops = ld.GetLoopsInOrderOfAppearance();

    opt::LoopFusion fusion(context.get(), loops[0], loops[1]);
    EXPECT_TRUE(fusion.AreCompatible());
    EXPECT_FALSE(fusion.IsLegal());
  }
}

} // namespace
