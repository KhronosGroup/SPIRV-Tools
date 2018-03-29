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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "../assembly_builder.h"
#include "../function_utils.h"
#include "../pass_fixture.h"
#include "../pass_utils.h"

#include "opt/iterator.h"
#include "opt/loop_dependence.h"
#include "opt/loop_descriptor.h"
#include "opt/pass.h"
#include "opt/tree_iterator.h"

namespace {

using namespace spvtools;
using DependencyAnalysis = ::testing::Test;

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void main(){
  int[10] arr;
  int[10] arr2;
  int a = 2;
  for (int i = 0; i < 10; i++) {
    arr[a] = arr[3];
    arr[a*2] = arr[a+3];
    arr[6] = arr2[6];
    arr[a+5] = arr2[7];
  }
}
*/
TEST(DependencyAnalysis, ZIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %25 "arr"
               OpName %39 "arr2"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %11 = OpConstant %6 0
         %18 = OpConstant %6 10
         %19 = OpTypeBool
         %21 = OpTypeInt 32 0
         %22 = OpConstant %21 10
         %23 = OpTypeArray %6 %22
         %24 = OpTypePointer Function %23
         %27 = OpConstant %6 3
         %38 = OpConstant %6 6
         %44 = OpConstant %6 5
         %46 = OpConstant %6 7
         %51 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %25 = OpVariable %24 Function
         %39 = OpVariable %24 Function
               OpBranch %12
         %12 = OpLabel
         %53 = OpPhi %6 %11 %5 %52 %15
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %20 = OpSLessThan %19 %53 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %28 = OpAccessChain %7 %25 %27
         %29 = OpLoad %6 %28
         %30 = OpAccessChain %7 %25 %9
               OpStore %30 %29
         %32 = OpIMul %6 %9 %9
         %34 = OpIAdd %6 %9 %27
         %35 = OpAccessChain %7 %25 %34
         %36 = OpLoad %6 %35
         %37 = OpAccessChain %7 %25 %32
               OpStore %37 %36
         %40 = OpAccessChain %7 %39 %38
         %41 = OpLoad %6 %40
         %42 = OpAccessChain %7 %25 %38
               OpStore %42 %41
         %45 = OpIAdd %6 %9 %44
         %47 = OpAccessChain %7 %39 %46
         %48 = OpLoad %6 %47
         %49 = OpAccessChain %7 %25 %45
               OpStore %49 %48
               OpBranch %15
         %15 = OpLabel
         %52 = OpIAdd %6 %53 %51
               OpBranch %12
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const ir::Function* f = spvtest::GetFunction(module, 4);
  ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  ir::Loop* loop = &ld.GetLoopByIndex(0);
  std::vector<const ir::Loop*> loops{loop};
  opt::LoopDependenceAnalysis analysis{context.get(), loops};

  const ir::Instruction* store[4];
  int stores_found = 0;
  for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 13)) {
    if (inst.opcode() == SpvOp::SpvOpStore) {
      store[stores_found] = &inst;
      ++stores_found;
    }
  }

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(store[i]);
  }

  // 29 -> 30 tests looking through constants.
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(29),
                                       store[0], &distance_vector));
  }

  // 36 -> 37 tests looking through additions.
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(36),
                                       store[1], &distance_vector));
  }

  // 41 -> 42 tests looking at same index across two different arrays.
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(41),
                                       store[2], &distance_vector));
  }

  // 48 -> 49 tests looking through additions for same index in two different
  // arrays.
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(48),
                                       store[3], &distance_vector));
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
layout(location = 0) in vec4 c;
void main(){
  int[10] arr;
  int[10] arr2;
  int[10] arr3;
  int[10] arr4;
  int[10] arr5;
  int N = int(c.x);
  for (int i = 0; i < N; i++) {
    arr[2*N] = arr[N];
    arr2[2*N+1] = arr2[N];
    arr3[2*N] = arr3[N-1];
    arr4[N] = arr5[N];
  }
}
*/
TEST(DependencyAnalysis, SymbolicZIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %12
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %12 "c"
               OpName %33 "arr"
               OpName %41 "arr2"
               OpName %50 "arr3"
               OpName %58 "arr4"
               OpName %60 "arr5"
               OpDecorate %12 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpTypeFloat 32
         %10 = OpTypeVector %9 4
         %11 = OpTypePointer Input %10
         %12 = OpVariable %11 Input
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 0
         %15 = OpTypePointer Input %9
         %20 = OpConstant %6 0
         %28 = OpTypeBool
         %30 = OpConstant %13 10
         %31 = OpTypeArray %6 %30
         %32 = OpTypePointer Function %31
         %34 = OpConstant %6 2
         %44 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %33 = OpVariable %32 Function
         %41 = OpVariable %32 Function
         %50 = OpVariable %32 Function
         %58 = OpVariable %32 Function
         %60 = OpVariable %32 Function
         %16 = OpAccessChain %15 %12 %14
         %17 = OpLoad %9 %16
         %18 = OpConvertFToS %6 %17
               OpBranch %21
         %21 = OpLabel
         %67 = OpPhi %6 %20 %5 %66 %24
               OpLoopMerge %23 %24 None
               OpBranch %25
         %25 = OpLabel
         %29 = OpSLessThan %28 %67 %18
               OpBranchConditional %29 %22 %23
         %22 = OpLabel
         %36 = OpIMul %6 %34 %18
         %38 = OpAccessChain %7 %33 %18
         %39 = OpLoad %6 %38
         %40 = OpAccessChain %7 %33 %36
               OpStore %40 %39
         %43 = OpIMul %6 %34 %18
         %45 = OpIAdd %6 %43 %44
         %47 = OpAccessChain %7 %41 %18
         %48 = OpLoad %6 %47
         %49 = OpAccessChain %7 %41 %45
               OpStore %49 %48
         %52 = OpIMul %6 %34 %18
         %54 = OpISub %6 %18 %44
         %55 = OpAccessChain %7 %50 %54
         %56 = OpLoad %6 %55
         %57 = OpAccessChain %7 %50 %52
               OpStore %57 %56
         %62 = OpAccessChain %7 %60 %18
         %63 = OpLoad %6 %62
         %64 = OpAccessChain %7 %58 %18
               OpStore %64 %63
               OpBranch %24
         %24 = OpLabel
         %66 = OpIAdd %6 %67 %44
               OpBranch %21
         %23 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const ir::Function* f = spvtest::GetFunction(module, 4);
  ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  ir::Loop* loop = &ld.GetLoopByIndex(0);
  std::vector<const ir::Loop*> loops{loop};
  opt::LoopDependenceAnalysis analysis{context.get(), loops};

  const ir::Instruction* store[4];
  int stores_found = 0;
  for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 22)) {
    if (inst.opcode() == SpvOp::SpvOpStore) {
      store[stores_found] = &inst;
      ++stores_found;
    }
  }

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(store[i]);
  }

  // independent due to loop bounds (won't enter if N <= 0).
  // 39 -> 40 tests looking through symbols and multiplicaiton.
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(39),
                                       store[0], &distance_vector));
  }

  // 48 -> 49 tests looking through symbols and multiplication + addition.
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(48),
                                       store[1], &distance_vector));
  }

  // 56 -> 57 tests looking through symbols and arithmetic on load and store.
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(56),
                                       store[2], &distance_vector));
  }

  // independent as different arrays
  // 63 -> 64 tests looking through symbols and load/store from/to different
  // arrays.
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(63),
                                       store[3], &distance_vector));
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a(){
  int[10] arr;
  int[11] arr2;
  int[20] arr3;
  int[20] arr4;
  int a = 2;
  for (int i = 0; i < 10; i++) {
    arr[i] = arr[i];
    arr2[i] = arr2[i+1];
    arr3[i] = arr3[i-1];
    arr4[2*i] = arr4[i];
  }
}
void b(){
  int[10] arr;
  int[11] arr2;
  int[20] arr3;
  int[20] arr4;
  int a = 2;
  for (int i = 10; i > 0; i--) {
    arr[i] = arr[i];
    arr2[i] = arr2[i+1];
    arr3[i] = arr3[i-1];
    arr4[2*i] = arr4[i];
  }
}

void main() {
  a();
  b();
}
*/
TEST(DependencyAnalysis, SIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %12 "a"
               OpName %14 "i"
               OpName %29 "arr"
               OpName %38 "arr2"
               OpName %49 "arr3"
               OpName %56 "arr4"
               OpName %65 "a"
               OpName %66 "i"
               OpName %74 "arr"
               OpName %80 "arr2"
               OpName %87 "arr3"
               OpName %94 "arr4"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %13 = OpConstant %10 2
         %15 = OpConstant %10 0
         %22 = OpConstant %10 10
         %23 = OpTypeBool
         %25 = OpTypeInt 32 0
         %26 = OpConstant %25 10
         %27 = OpTypeArray %10 %26
         %28 = OpTypePointer Function %27
         %35 = OpConstant %25 11
         %36 = OpTypeArray %10 %35
         %37 = OpTypePointer Function %36
         %41 = OpConstant %10 1
         %46 = OpConstant %25 20
         %47 = OpTypeArray %10 %46
         %48 = OpTypePointer Function %47
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %103 = OpFunctionCall %2 %6
        %104 = OpFunctionCall %2 %8
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %12 = OpVariable %11 Function
         %14 = OpVariable %11 Function
         %29 = OpVariable %28 Function
         %38 = OpVariable %37 Function
         %49 = OpVariable %48 Function
         %56 = OpVariable %48 Function
               OpStore %12 %13
               OpStore %14 %15
               OpBranch %16
         %16 = OpLabel
        %105 = OpPhi %10 %15 %7 %64 %19
               OpLoopMerge %18 %19 None
               OpBranch %20
         %20 = OpLabel
         %24 = OpSLessThan %23 %105 %22
               OpBranchConditional %24 %17 %18
         %17 = OpLabel
         %32 = OpAccessChain %11 %29 %105
         %33 = OpLoad %10 %32
         %34 = OpAccessChain %11 %29 %105
               OpStore %34 %33
         %42 = OpIAdd %10 %105 %41
         %43 = OpAccessChain %11 %38 %42
         %44 = OpLoad %10 %43
         %45 = OpAccessChain %11 %38 %105
               OpStore %45 %44
         %52 = OpISub %10 %105 %41
         %53 = OpAccessChain %11 %49 %52
         %54 = OpLoad %10 %53
         %55 = OpAccessChain %11 %49 %105
               OpStore %55 %54
         %58 = OpIMul %10 %13 %105
         %60 = OpAccessChain %11 %56 %105
         %61 = OpLoad %10 %60
         %62 = OpAccessChain %11 %56 %58
               OpStore %62 %61
               OpBranch %19
         %19 = OpLabel
         %64 = OpIAdd %10 %105 %41
               OpStore %14 %64
               OpBranch %16
         %18 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %65 = OpVariable %11 Function
         %66 = OpVariable %11 Function
         %74 = OpVariable %28 Function
         %80 = OpVariable %37 Function
         %87 = OpVariable %48 Function
         %94 = OpVariable %48 Function
               OpStore %65 %13
               OpStore %66 %22
               OpBranch %67
         %67 = OpLabel
        %106 = OpPhi %10 %22 %9 %102 %70
               OpLoopMerge %69 %70 None
               OpBranch %71
         %71 = OpLabel
         %73 = OpSGreaterThan %23 %106 %15
               OpBranchConditional %73 %68 %69
         %68 = OpLabel
         %77 = OpAccessChain %11 %74 %106
         %78 = OpLoad %10 %77
         %79 = OpAccessChain %11 %74 %106
               OpStore %79 %78
         %83 = OpIAdd %10 %106 %41
         %84 = OpAccessChain %11 %80 %83
         %85 = OpLoad %10 %84
         %86 = OpAccessChain %11 %80 %106
               OpStore %86 %85
         %90 = OpISub %10 %106 %41
         %91 = OpAccessChain %11 %87 %90
         %92 = OpLoad %10 %91
         %93 = OpAccessChain %11 %87 %106
               OpStore %93 %92
         %96 = OpIMul %10 %13 %106
         %98 = OpAccessChain %11 %94 %106
         %99 = OpLoad %10 %98
        %100 = OpAccessChain %11 %94 %96
               OpStore %100 %99
               OpBranch %70
         %70 = OpLabel
        %102 = OpISub %10 %106 %41
               OpStore %66 %102
               OpBranch %67
         %69 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  // For the loop in function a.
  {
    const ir::Function* f = spvtest::GetFunction(module, 6);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store[4];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 17)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // = dependence
    // 33 -> 34 tests looking at SIV in same array.
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(33), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                opt::DistanceEntry::Directions::EQ);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
    }

    // > -1 dependence
    // 44 -> 45 tests looking at SIV in same array with addition.
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(44), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                opt::DistanceEntry::Directions::GT);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, -1);
    }

    // < 1 dependence
    // 54 -> 55 tests looking at SIV in same array with subtraction.
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(54), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                opt::DistanceEntry::Directions::LT);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 1);
    }

    // <=> dependence
    // 61 -> 62 tests looking at SIV in same array with multiplication.
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(61), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::UNKNOWN);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                opt::DistanceEntry::Directions::ALL);
    }
  }
  // For the loop in function b.
  {
    const ir::Function* f = spvtest::GetFunction(module, 8);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store[4];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 68)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // = dependence
    // 78 -> 79 tests looking at SIV in same array.
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(78), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                opt::DistanceEntry::Directions::EQ);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
    }

    // < 1 dependence
    // 85 -> 86 tests looking at SIV in same array with addition.
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(85), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                opt::DistanceEntry::Directions::LT);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 1);
    }

    // > -1 dependence
    // 92 -> 93 tests looking at SIV in same array with subtraction.
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(92), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                opt::DistanceEntry::Directions::GT);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, -1);
    }

    // <=> dependence
    // 99 -> 100 tests looking at SIV in same array with multiplication.
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(99), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::UNKNOWN);
      EXPECT_EQ(distance_vector.GetEntries()[0].direction,
                opt::DistanceEntry::Directions::ALL);
    }
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
layout(location = 0) in vec4 c;
void a() {
  int[13] arr;
  int[15] arr2;
  int[18] arr3;
  int[18] arr4;
  int N = int(c.x);
  int C = 2;
  int a = 2;
  for (int i = 0; i < N; i++) { // Bounds are N - 1
    arr[i+2*N] = arr[i+N]; // |distance| = N
    arr2[i+N] = arr2[i+2*N] + C; // |distance| = N
    arr3[2*i+2*N+1] = arr3[2*i+N+1]; // |distance| = N
    arr4[a*i+N+1] = arr4[a*i+2*N+1]; // |distance| = N
  }
}
void b() {
  int[13] arr;
  int[15] arr2;
  int[18] arr3;
  int[18] arr4;
  int N = int(c.x);
  int C = 2;
  int a = 2;
  for (int i = N; i > 0; i--) { // Bounds are N - 1
    arr[i+2*N] = arr[i+N]; // |distance| = N
    arr2[i+N] = arr2[i+2*N] + C; // |distance| = N
    arr3[2*i+2*N+1] = arr3[2*i+N+1]; // |distance| = N
    arr4[a*i+N+1] = arr4[a*i+2*N+1]; // |distance| = N
  }
}
void main(){
  a();
  b();
}*/
TEST(DependencyAnalysis, SymbolicSIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %16
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %12 "N"
               OpName %16 "c"
               OpName %23 "C"
               OpName %25 "a"
               OpName %26 "i"
               OpName %40 "arr"
               OpName %54 "arr2"
               OpName %70 "arr3"
               OpName %86 "arr4"
               OpName %105 "N"
               OpName %109 "C"
               OpName %110 "a"
               OpName %111 "i"
               OpName %120 "arr"
               OpName %131 "arr2"
               OpName %144 "arr3"
               OpName %159 "arr4"
               OpDecorate %16 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %13 = OpTypeFloat 32
         %14 = OpTypeVector %13 4
         %15 = OpTypePointer Input %14
         %16 = OpVariable %15 Input
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 0
         %19 = OpTypePointer Input %13
         %24 = OpConstant %10 2
         %27 = OpConstant %10 0
         %35 = OpTypeBool
         %37 = OpConstant %17 13
         %38 = OpTypeArray %10 %37
         %39 = OpTypePointer Function %38
         %51 = OpConstant %17 15
         %52 = OpTypeArray %10 %51
         %53 = OpTypePointer Function %52
         %67 = OpConstant %17 18
         %68 = OpTypeArray %10 %67
         %69 = OpTypePointer Function %68
         %76 = OpConstant %10 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %178 = OpFunctionCall %2 %6
        %179 = OpFunctionCall %2 %8
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %12 = OpVariable %11 Function
         %23 = OpVariable %11 Function
         %25 = OpVariable %11 Function
         %26 = OpVariable %11 Function
         %40 = OpVariable %39 Function
         %54 = OpVariable %53 Function
         %70 = OpVariable %69 Function
         %86 = OpVariable %69 Function
         %20 = OpAccessChain %19 %16 %18
         %21 = OpLoad %13 %20
         %22 = OpConvertFToS %10 %21
               OpStore %12 %22
               OpStore %23 %24
               OpStore %25 %24
               OpStore %26 %27
               OpBranch %28
         %28 = OpLabel
        %180 = OpPhi %10 %27 %7 %104 %31
               OpLoopMerge %30 %31 None
               OpBranch %32
         %32 = OpLabel
         %36 = OpSLessThan %35 %180 %22
               OpBranchConditional %36 %29 %30
         %29 = OpLabel
         %43 = OpIMul %10 %24 %22
         %44 = OpIAdd %10 %180 %43
         %47 = OpIAdd %10 %180 %22
         %48 = OpAccessChain %11 %40 %47
         %49 = OpLoad %10 %48
         %50 = OpAccessChain %11 %40 %44
               OpStore %50 %49
         %57 = OpIAdd %10 %180 %22
         %60 = OpIMul %10 %24 %22
         %61 = OpIAdd %10 %180 %60
         %62 = OpAccessChain %11 %54 %61
         %63 = OpLoad %10 %62
         %65 = OpIAdd %10 %63 %24
         %66 = OpAccessChain %11 %54 %57
               OpStore %66 %65
         %72 = OpIMul %10 %24 %180
         %74 = OpIMul %10 %24 %22
         %75 = OpIAdd %10 %72 %74
         %77 = OpIAdd %10 %75 %76
         %79 = OpIMul %10 %24 %180
         %81 = OpIAdd %10 %79 %22
         %82 = OpIAdd %10 %81 %76
         %83 = OpAccessChain %11 %70 %82
         %84 = OpLoad %10 %83
         %85 = OpAccessChain %11 %70 %77
               OpStore %85 %84
         %89 = OpIMul %10 %24 %180
         %91 = OpIAdd %10 %89 %22
         %92 = OpIAdd %10 %91 %76
         %95 = OpIMul %10 %24 %180
         %97 = OpIMul %10 %24 %22
         %98 = OpIAdd %10 %95 %97
         %99 = OpIAdd %10 %98 %76
        %100 = OpAccessChain %11 %86 %99
        %101 = OpLoad %10 %100
        %102 = OpAccessChain %11 %86 %92
               OpStore %102 %101
               OpBranch %31
         %31 = OpLabel
        %104 = OpIAdd %10 %180 %76
               OpStore %26 %104
               OpBranch %28
         %30 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
        %105 = OpVariable %11 Function
        %109 = OpVariable %11 Function
        %110 = OpVariable %11 Function
        %111 = OpVariable %11 Function
        %120 = OpVariable %39 Function
        %131 = OpVariable %53 Function
        %144 = OpVariable %69 Function
        %159 = OpVariable %69 Function
        %106 = OpAccessChain %19 %16 %18
        %107 = OpLoad %13 %106
        %108 = OpConvertFToS %10 %107
               OpStore %105 %108
               OpStore %109 %24
               OpStore %110 %24
               OpStore %111 %108
               OpBranch %113
        %113 = OpLabel
        %181 = OpPhi %10 %108 %9 %177 %116
               OpLoopMerge %115 %116 None
               OpBranch %117
        %117 = OpLabel
        %119 = OpSGreaterThan %35 %181 %27
               OpBranchConditional %119 %114 %115
        %114 = OpLabel
        %123 = OpIMul %10 %24 %108
        %124 = OpIAdd %10 %181 %123
        %127 = OpIAdd %10 %181 %108
        %128 = OpAccessChain %11 %120 %127
        %129 = OpLoad %10 %128
        %130 = OpAccessChain %11 %120 %124
               OpStore %130 %129
        %134 = OpIAdd %10 %181 %108
        %137 = OpIMul %10 %24 %108
        %138 = OpIAdd %10 %181 %137
        %139 = OpAccessChain %11 %131 %138
        %140 = OpLoad %10 %139
        %142 = OpIAdd %10 %140 %24
        %143 = OpAccessChain %11 %131 %134
               OpStore %143 %142
        %146 = OpIMul %10 %24 %181
        %148 = OpIMul %10 %24 %108
        %149 = OpIAdd %10 %146 %148
        %150 = OpIAdd %10 %149 %76
        %152 = OpIMul %10 %24 %181
        %154 = OpIAdd %10 %152 %108
        %155 = OpIAdd %10 %154 %76
        %156 = OpAccessChain %11 %144 %155
        %157 = OpLoad %10 %156
        %158 = OpAccessChain %11 %144 %150
               OpStore %158 %157
        %162 = OpIMul %10 %24 %181
        %164 = OpIAdd %10 %162 %108
        %165 = OpIAdd %10 %164 %76
        %168 = OpIMul %10 %24 %181
        %170 = OpIMul %10 %24 %108
        %171 = OpIAdd %10 %168 %170
        %172 = OpIAdd %10 %171 %76
        %173 = OpAccessChain %11 %159 %172
        %174 = OpLoad %10 %173
        %175 = OpAccessChain %11 %159 %165
               OpStore %175 %174
               OpBranch %116
        %116 = OpLabel
        %177 = OpISub %10 %181 %76
               OpStore %111 %177
               OpBranch %113
        %115 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  // For the loop in function a.
  {
    const ir::Function* f = spvtest::GetFunction(module, 6);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store[4];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 29)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // independent due to loop bounds (won't enter when N <= 0)
    // 49 -> 50 tests looking through SIV and symbols with multiplication
    {
      opt::DistanceVector distance_vector{loops.size()};
      // Independent but not yet supported.
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(49), store[0], &distance_vector));
    }

    // 63 -> 66 tests looking through SIV and symbols with multiplication and +
    // C
    {
      opt::DistanceVector distance_vector{loops.size()};
      // Independent.
      EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(63),
                                         store[1], &distance_vector));
    }

    // 84 -> 85 tests looking through arithmetic on SIV and symbols
    {
      opt::DistanceVector distance_vector{loops.size()};
      // Independent but not yet supported.
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(84), store[2], &distance_vector));
    }

    // 101 -> 102 tests looking through symbol arithmetic on SIV and symbols
    {
      opt::DistanceVector distance_vector{loops.size()};
      // Independent.
      EXPECT_TRUE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(101), store[3], &distance_vector));
    }
  }
  // For the loop in function b.
  {
    const ir::Function* f = spvtest::GetFunction(module, 8);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store[4];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 114)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // independent due to loop bounds (won't enter when N <= 0).
    // 129 -> 130 tests looking through SIV and symbols with multiplication.
    {
      opt::DistanceVector distance_vector{loops.size()};
      // Independent but not yet supported.
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(129), store[0], &distance_vector));
    }

    // 140 -> 143 tests looking through SIV and symbols with multiplication and
    // + C.
    {
      opt::DistanceVector distance_vector{loops.size()};
      // Independent.
      EXPECT_TRUE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(140), store[1], &distance_vector));
    }

    // 157 -> 158 tests looking through arithmetic on SIV and symbols.
    {
      opt::DistanceVector distance_vector{loops.size()};
      // Independent but not yet supported.
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(157), store[2], &distance_vector));
    }

    // 174 -> 175 tests looking through symbol arithmetic on SIV and symbols.
    {
      opt::DistanceVector distance_vector{loops.size()};
      // Independent.
      EXPECT_TRUE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(174), store[3], &distance_vector));
    }
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a() {
  int[6] arr;
  int N = 5;
  for (int i = 1; i < N; i++) {
    arr[i] = arr[N-i];
  }
}
void b() {
  int[6] arr;
  int N = 5;
  for (int i = 1; i < N; i++) {
    arr[N-i] = arr[i];
  }
}
void c() {
  int[11] arr;
  int N = 10;
  for (int i = 1; i < N; i++) {
    arr[i] = arr[N-i+1];
  }
}
void d() {
  int[11] arr;
  int N = 10;
  for (int i = 1; i < N; i++) {
    arr[N-i+1] = arr[i];
  }
}
void e() {
  int[6] arr;
  int N = 5;
  for (int i = N; i > 0; i--) {
    arr[i] = arr[N-i];
  }
}
void f() {
  int[6] arr;
  int N = 5;
  for (int i = N; i > 0; i--) {
    arr[N-i] = arr[i];
  }
}
void g() {
  int[11] arr;
  int N = 10;
  for (int i = N; i > 0; i--) {
    arr[i] = arr[N-i+1];
  }
}
void h() {
  int[11] arr;
  int N = 10;
  for (int i = N; i > 0; i--) {
    arr[N-i+1] = arr[i];
  }
}
void main(){
  a();
  b();
  c();
  d();
  e();
  f();
  g();
  h();
}
*/
TEST(DependencyAnalysis, Crossing) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %10 "c("
               OpName %12 "d("
               OpName %14 "e("
               OpName %16 "f("
               OpName %18 "g("
               OpName %20 "h("
               OpName %24 "N"
               OpName %26 "i"
               OpName %41 "arr"
               OpName %51 "N"
               OpName %52 "i"
               OpName %61 "arr"
               OpName %71 "N"
               OpName %73 "i"
               OpName %85 "arr"
               OpName %96 "N"
               OpName %97 "i"
               OpName %106 "arr"
               OpName %117 "N"
               OpName %118 "i"
               OpName %128 "arr"
               OpName %138 "N"
               OpName %139 "i"
               OpName %148 "arr"
               OpName %158 "N"
               OpName %159 "i"
               OpName %168 "arr"
               OpName %179 "N"
               OpName %180 "i"
               OpName %189 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %22 = OpTypeInt 32 1
         %23 = OpTypePointer Function %22
         %25 = OpConstant %22 5
         %27 = OpConstant %22 1
         %35 = OpTypeBool
         %37 = OpTypeInt 32 0
         %38 = OpConstant %37 6
         %39 = OpTypeArray %22 %38
         %40 = OpTypePointer Function %39
         %72 = OpConstant %22 10
         %82 = OpConstant %37 11
         %83 = OpTypeArray %22 %82
         %84 = OpTypePointer Function %83
        %126 = OpConstant %22 0
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %200 = OpFunctionCall %2 %6
        %201 = OpFunctionCall %2 %8
        %202 = OpFunctionCall %2 %10
        %203 = OpFunctionCall %2 %12
        %204 = OpFunctionCall %2 %14
        %205 = OpFunctionCall %2 %16
        %206 = OpFunctionCall %2 %18
        %207 = OpFunctionCall %2 %20
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %24 = OpVariable %23 Function
         %26 = OpVariable %23 Function
         %41 = OpVariable %40 Function
               OpStore %24 %25
               OpStore %26 %27
               OpBranch %28
         %28 = OpLabel
        %208 = OpPhi %22 %27 %7 %50 %31
               OpLoopMerge %30 %31 None
               OpBranch %32
         %32 = OpLabel
         %36 = OpSLessThan %35 %208 %25
               OpBranchConditional %36 %29 %30
         %29 = OpLabel
         %45 = OpISub %22 %25 %208
         %46 = OpAccessChain %23 %41 %45
         %47 = OpLoad %22 %46
         %48 = OpAccessChain %23 %41 %208
               OpStore %48 %47
               OpBranch %31
         %31 = OpLabel
         %50 = OpIAdd %22 %208 %27
               OpStore %26 %50
               OpBranch %28
         %30 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %51 = OpVariable %23 Function
         %52 = OpVariable %23 Function
         %61 = OpVariable %40 Function
               OpStore %51 %25
               OpStore %52 %27
               OpBranch %53
         %53 = OpLabel
        %209 = OpPhi %22 %27 %9 %70 %56
               OpLoopMerge %55 %56 None
               OpBranch %57
         %57 = OpLabel
         %60 = OpSLessThan %35 %209 %25
               OpBranchConditional %60 %54 %55
         %54 = OpLabel
         %64 = OpISub %22 %25 %209
         %66 = OpAccessChain %23 %61 %209
         %67 = OpLoad %22 %66
         %68 = OpAccessChain %23 %61 %64
               OpStore %68 %67
               OpBranch %56
         %56 = OpLabel
         %70 = OpIAdd %22 %209 %27
               OpStore %52 %70
               OpBranch %53
         %55 = OpLabel
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %3
         %11 = OpLabel
         %71 = OpVariable %23 Function
         %73 = OpVariable %23 Function
         %85 = OpVariable %84 Function
               OpStore %71 %72
               OpStore %73 %27
               OpBranch %74
         %74 = OpLabel
        %210 = OpPhi %22 %27 %11 %95 %77
               OpLoopMerge %76 %77 None
               OpBranch %78
         %78 = OpLabel
         %81 = OpSLessThan %35 %210 %72
               OpBranchConditional %81 %75 %76
         %75 = OpLabel
         %89 = OpISub %22 %72 %210
         %90 = OpIAdd %22 %89 %27
         %91 = OpAccessChain %23 %85 %90
         %92 = OpLoad %22 %91
         %93 = OpAccessChain %23 %85 %210
               OpStore %93 %92
               OpBranch %77
         %77 = OpLabel
         %95 = OpIAdd %22 %210 %27
               OpStore %73 %95
               OpBranch %74
         %76 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
         %96 = OpVariable %23 Function
         %97 = OpVariable %23 Function
        %106 = OpVariable %84 Function
               OpStore %96 %72
               OpStore %97 %27
               OpBranch %98
         %98 = OpLabel
        %211 = OpPhi %22 %27 %13 %116 %101
               OpLoopMerge %100 %101 None
               OpBranch %102
        %102 = OpLabel
        %105 = OpSLessThan %35 %211 %72
               OpBranchConditional %105 %99 %100
         %99 = OpLabel
        %109 = OpISub %22 %72 %211
        %110 = OpIAdd %22 %109 %27
        %112 = OpAccessChain %23 %106 %211
        %113 = OpLoad %22 %112
        %114 = OpAccessChain %23 %106 %110
               OpStore %114 %113
               OpBranch %101
        %101 = OpLabel
        %116 = OpIAdd %22 %211 %27
               OpStore %97 %116
               OpBranch %98
        %100 = OpLabel
               OpReturn
               OpFunctionEnd
         %14 = OpFunction %2 None %3
         %15 = OpLabel
        %117 = OpVariable %23 Function
        %118 = OpVariable %23 Function
        %128 = OpVariable %40 Function
               OpStore %117 %25
               OpStore %118 %25
               OpBranch %120
        %120 = OpLabel
        %212 = OpPhi %22 %25 %15 %137 %123
               OpLoopMerge %122 %123 None
               OpBranch %124
        %124 = OpLabel
        %127 = OpSGreaterThan %35 %212 %126
               OpBranchConditional %127 %121 %122
        %121 = OpLabel
        %132 = OpISub %22 %25 %212
        %133 = OpAccessChain %23 %128 %132
        %134 = OpLoad %22 %133
        %135 = OpAccessChain %23 %128 %212
               OpStore %135 %134
               OpBranch %123
        %123 = OpLabel
        %137 = OpISub %22 %212 %27
               OpStore %118 %137
               OpBranch %120
        %122 = OpLabel
               OpReturn
               OpFunctionEnd
         %16 = OpFunction %2 None %3
         %17 = OpLabel
        %138 = OpVariable %23 Function
        %139 = OpVariable %23 Function
        %148 = OpVariable %40 Function
               OpStore %138 %25
               OpStore %139 %25
               OpBranch %141
        %141 = OpLabel
        %213 = OpPhi %22 %25 %17 %157 %144
               OpLoopMerge %143 %144 None
               OpBranch %145
        %145 = OpLabel
        %147 = OpSGreaterThan %35 %213 %126
               OpBranchConditional %147 %142 %143
        %142 = OpLabel
        %151 = OpISub %22 %25 %213
        %153 = OpAccessChain %23 %148 %213
        %154 = OpLoad %22 %153
        %155 = OpAccessChain %23 %148 %151
               OpStore %155 %154
               OpBranch %144
        %144 = OpLabel
        %157 = OpISub %22 %213 %27
               OpStore %139 %157
               OpBranch %141
        %143 = OpLabel
               OpReturn
               OpFunctionEnd
         %18 = OpFunction %2 None %3
         %19 = OpLabel
        %158 = OpVariable %23 Function
        %159 = OpVariable %23 Function
        %168 = OpVariable %84 Function
               OpStore %158 %72
               OpStore %159 %72
               OpBranch %161
        %161 = OpLabel
        %214 = OpPhi %22 %72 %19 %178 %164
               OpLoopMerge %163 %164 None
               OpBranch %165
        %165 = OpLabel
        %167 = OpSGreaterThan %35 %214 %126
               OpBranchConditional %167 %162 %163
        %162 = OpLabel
        %172 = OpISub %22 %72 %214
        %173 = OpIAdd %22 %172 %27
        %174 = OpAccessChain %23 %168 %173
        %175 = OpLoad %22 %174
        %176 = OpAccessChain %23 %168 %214
               OpStore %176 %175
               OpBranch %164
        %164 = OpLabel
        %178 = OpISub %22 %214 %27
               OpStore %159 %178
               OpBranch %161
        %163 = OpLabel
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %3
         %21 = OpLabel
        %179 = OpVariable %23 Function
        %180 = OpVariable %23 Function
        %189 = OpVariable %84 Function
               OpStore %179 %72
               OpStore %180 %72
               OpBranch %182
        %182 = OpLabel
        %215 = OpPhi %22 %72 %21 %199 %185
               OpLoopMerge %184 %185 None
               OpBranch %186
        %186 = OpLabel
        %188 = OpSGreaterThan %35 %215 %126
               OpBranchConditional %188 %183 %184
        %183 = OpLabel
        %192 = OpISub %22 %72 %215
        %193 = OpIAdd %22 %192 %27
        %195 = OpAccessChain %23 %189 %215
        %196 = OpLoad %22 %195
        %197 = OpAccessChain %23 %189 %193
               OpStore %197 %196
               OpBranch %185
        %185 = OpLabel
        %199 = OpISub %22 %215 %27
               OpStore %180 %199
               OpBranch %182
        %184 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  // First two tests can be split into two loops.
  // Tests even crossing subscripts from low to high indexes.
  // 47 -> 48
  {
    const ir::Function* f = spvtest::GetFunction(module, 6);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 29)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(47),
                                        store, &distance_vector));
  }

  // Tests even crossing subscripts from high to low indexes.
  // 67 -> 68
  {
    const ir::Function* f = spvtest::GetFunction(module, 8);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 54)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(67),
                                        store, &distance_vector));
  }

  // Next two tests can have an end peeled, then be split.
  // Tests uneven crossing subscripts from low to high indexes.
  // 92 -> 93
  {
    const ir::Function* f = spvtest::GetFunction(module, 10);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 75)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(92),
                                        store, &distance_vector));
  }

  // Tests uneven crossing subscripts from high to low indexes.
  // 113 -> 114
  {
    const ir::Function* f = spvtest::GetFunction(module, 12);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 99)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(113),
                                        store, &distance_vector));
  }

  // First two tests can be split into two loops.
  // Tests even crossing subscripts from low to high indexes.
  // 134 -> 135
  {
    const ir::Function* f = spvtest::GetFunction(module, 14);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 121)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(134),
                                        store, &distance_vector));
  }

  // Tests even crossing subscripts from high to low indexes.
  // 154 -> 155
  {
    const ir::Function* f = spvtest::GetFunction(module, 16);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 142)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(154),
                                        store, &distance_vector));
  }

  // Next two tests can have an end peeled, then be split.
  // Tests uneven crossing subscripts from low to high indexes.
  // 175 -> 176
  {
    const ir::Function* f = spvtest::GetFunction(module, 18);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 162)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(175),
                                        store, &distance_vector));
  }

  // Tests uneven crossing subscripts from high to low indexes.
  // 196 -> 197
  {
    const ir::Function* f = spvtest::GetFunction(module, 20);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 183)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store = &inst;
      }
    }
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(196),
                                        store, &distance_vector));
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a() {
  int[10] arr;
  for (int i = 0; i < 10; i++) {
    arr[0] = arr[i]; // peel first
    arr[i] = arr[0]; // peel first
    arr[9] = arr[i]; // peel last
    arr[i] = arr[9]; // peel last
  }
}
void b() {
  int[11] arr;
  for (int i = 0; i <= 10; i++) {
    arr[0] = arr[i]; // peel first
    arr[i] = arr[0]; // peel first
    arr[10] = arr[i]; // peel last
    arr[i] = arr[10]; // peel last

  }
}
void c() {
  int[11] arr;
  for (int i = 10; i > 0; i--) {
    arr[10] = arr[i]; // peel first
    arr[i] = arr[10]; // peel first
    arr[1] = arr[i]; // peel last
    arr[i] = arr[1]; // peel last

  }
}
void d() {
  int[11] arr;
  for (int i = 10; i >= 0; i--) {
    arr[10] = arr[i]; // peel first
    arr[i] = arr[10]; // peel first
    arr[0] = arr[i]; // peel last
    arr[i] = arr[0]; // peel last

  }
}
void main(){
  a();
  b();
  c();
  d();
}
*/
TEST(DependencyAnalysis, WeakZeroSIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %10 "c("
               OpName %12 "d("
               OpName %16 "i"
               OpName %31 "arr"
               OpName %52 "i"
               OpName %63 "arr"
               OpName %82 "i"
               OpName %90 "arr"
               OpName %109 "i"
               OpName %117 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %17 = OpConstant %14 0
         %24 = OpConstant %14 10
         %25 = OpTypeBool
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 10
         %29 = OpTypeArray %14 %28
         %30 = OpTypePointer Function %29
         %40 = OpConstant %14 9
         %50 = OpConstant %14 1
         %60 = OpConstant %27 11
         %61 = OpTypeArray %14 %60
         %62 = OpTypePointer Function %61
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %136 = OpFunctionCall %2 %6
        %137 = OpFunctionCall %2 %8
        %138 = OpFunctionCall %2 %10
        %139 = OpFunctionCall %2 %12
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %16 = OpVariable %15 Function
         %31 = OpVariable %30 Function
               OpStore %16 %17
               OpBranch %18
         %18 = OpLabel
        %140 = OpPhi %14 %17 %7 %51 %21
               OpLoopMerge %20 %21 None
               OpBranch %22
         %22 = OpLabel
         %26 = OpSLessThan %25 %140 %24
               OpBranchConditional %26 %19 %20
         %19 = OpLabel
         %33 = OpAccessChain %15 %31 %140
         %34 = OpLoad %14 %33
         %35 = OpAccessChain %15 %31 %17
               OpStore %35 %34
         %37 = OpAccessChain %15 %31 %17
         %38 = OpLoad %14 %37
         %39 = OpAccessChain %15 %31 %140
               OpStore %39 %38
         %42 = OpAccessChain %15 %31 %140
         %43 = OpLoad %14 %42
         %44 = OpAccessChain %15 %31 %40
               OpStore %44 %43
         %46 = OpAccessChain %15 %31 %40
         %47 = OpLoad %14 %46
         %48 = OpAccessChain %15 %31 %140
               OpStore %48 %47
               OpBranch %21
         %21 = OpLabel
         %51 = OpIAdd %14 %140 %50
               OpStore %16 %51
               OpBranch %18
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %52 = OpVariable %15 Function
         %63 = OpVariable %62 Function
               OpStore %52 %17
               OpBranch %53
         %53 = OpLabel
        %141 = OpPhi %14 %17 %9 %81 %56
               OpLoopMerge %55 %56 None
               OpBranch %57
         %57 = OpLabel
         %59 = OpSLessThanEqual %25 %141 %24
               OpBranchConditional %59 %54 %55
         %54 = OpLabel
         %65 = OpAccessChain %15 %63 %141
         %66 = OpLoad %14 %65
         %67 = OpAccessChain %15 %63 %17
               OpStore %67 %66
         %69 = OpAccessChain %15 %63 %17
         %70 = OpLoad %14 %69
         %71 = OpAccessChain %15 %63 %141
               OpStore %71 %70
         %73 = OpAccessChain %15 %63 %141
         %74 = OpLoad %14 %73
         %75 = OpAccessChain %15 %63 %24
               OpStore %75 %74
         %77 = OpAccessChain %15 %63 %24
         %78 = OpLoad %14 %77
         %79 = OpAccessChain %15 %63 %141
               OpStore %79 %78
               OpBranch %56
         %56 = OpLabel
         %81 = OpIAdd %14 %141 %50
               OpStore %52 %81
               OpBranch %53
         %55 = OpLabel
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %3
         %11 = OpLabel
         %82 = OpVariable %15 Function
         %90 = OpVariable %62 Function
               OpStore %82 %24
               OpBranch %83
         %83 = OpLabel
        %142 = OpPhi %14 %24 %11 %108 %86
               OpLoopMerge %85 %86 None
               OpBranch %87
         %87 = OpLabel
         %89 = OpSGreaterThan %25 %142 %17
               OpBranchConditional %89 %84 %85
         %84 = OpLabel
         %92 = OpAccessChain %15 %90 %142
         %93 = OpLoad %14 %92
         %94 = OpAccessChain %15 %90 %24
               OpStore %94 %93
         %96 = OpAccessChain %15 %90 %24
         %97 = OpLoad %14 %96
         %98 = OpAccessChain %15 %90 %142
               OpStore %98 %97
        %100 = OpAccessChain %15 %90 %142
        %101 = OpLoad %14 %100
        %102 = OpAccessChain %15 %90 %50
               OpStore %102 %101
        %104 = OpAccessChain %15 %90 %50
        %105 = OpLoad %14 %104
        %106 = OpAccessChain %15 %90 %142
               OpStore %106 %105
               OpBranch %86
         %86 = OpLabel
        %108 = OpISub %14 %142 %50
               OpStore %82 %108
               OpBranch %83
         %85 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
        %109 = OpVariable %15 Function
        %117 = OpVariable %62 Function
               OpStore %109 %24
               OpBranch %110
        %110 = OpLabel
        %143 = OpPhi %14 %24 %13 %135 %113
               OpLoopMerge %112 %113 None
               OpBranch %114
        %114 = OpLabel
        %116 = OpSGreaterThanEqual %25 %143 %17
               OpBranchConditional %116 %111 %112
        %111 = OpLabel
        %119 = OpAccessChain %15 %117 %143
        %120 = OpLoad %14 %119
        %121 = OpAccessChain %15 %117 %24
               OpStore %121 %120
        %123 = OpAccessChain %15 %117 %24
        %124 = OpLoad %14 %123
        %125 = OpAccessChain %15 %117 %143
               OpStore %125 %124
        %127 = OpAccessChain %15 %117 %143
        %128 = OpLoad %14 %127
        %129 = OpAccessChain %15 %117 %17
               OpStore %129 %128
        %131 = OpAccessChain %15 %117 %17
        %132 = OpLoad %14 %131
        %133 = OpAccessChain %15 %117 %143
               OpStore %133 %132
               OpBranch %113
        %113 = OpLabel
        %135 = OpISub %14 %143 %50
               OpStore %109 %135
               OpBranch %110
        %112 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  // For the loop in function a
  {
    const ir::Function* f = spvtest::GetFunction(module, 6);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store[4];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 19)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 34 -> 35
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(34), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 38 -> 39
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(38), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 43 -> 44
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(43), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 47 -> 48
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(47), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }
  }
  // For the loop in function b
  {
    const ir::Function* f = spvtest::GetFunction(module, 8);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store[4];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 54)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 66 -> 67
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(66), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 70 -> 71
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(70), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 74 -> 75
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(74), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 78 -> 79
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(78), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }
  }
  // For the loop in function c
  {
    const ir::Function* f = spvtest::GetFunction(module, 10);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};
    const ir::Instruction* store[4];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 84)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 93 -> 94
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(93), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 97 -> 98
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(97), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 101 -> 102
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(101), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 105 -> 106
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(105), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }
  }
  // For the loop in function d
  {
    const ir::Function* f = spvtest::GetFunction(module, 12);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    ir::Loop* loop = &ld.GetLoopByIndex(0);
    std::vector<const ir::Loop*> loops{loop};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store[4];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 111)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 120 -> 121
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(120), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 124 -> 125
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(124), store[1], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_first);
    }

    // Tests identifying peel first with weak zero with destination as zero
    // index.
    // 128 -> 129
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(128), store[2], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }

    // Tests identifying peel first with weak zero with source as zero index.
    // 132 -> 133
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(132), store[3], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::PEEL);
      EXPECT_TRUE(distance_vector.GetEntries()[0].peel_last);
    }
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void main(){
  int[10][10] arr;
  for (int i = 0; i < 10; i++) {
    arr[i][i] = arr[i][i];
    arr[0][i] = arr[1][i];
    arr[1][i] = arr[0][i];
    arr[i][0] = arr[i][1];
    arr[i][1] = arr[i][0];
    arr[0][1] = arr[1][0];
  }
}
*/
TEST(DependencyAnalysis, MultipleSubscriptZIVSIV) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %8 "i"
               OpName %24 "arr"
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
         %22 = OpTypeArray %21 %20
         %23 = OpTypePointer Function %22
         %33 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %24 = OpVariable %23 Function
               OpStore %8 %9
               OpBranch %10
         %10 = OpLabel
         %58 = OpPhi %6 %9 %5 %57 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %58 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %29 = OpAccessChain %7 %24 %58 %58
         %30 = OpLoad %6 %29
         %31 = OpAccessChain %7 %24 %58 %58
               OpStore %31 %30
         %35 = OpAccessChain %7 %24 %33 %58
         %36 = OpLoad %6 %35
         %37 = OpAccessChain %7 %24 %9 %58
               OpStore %37 %36
         %40 = OpAccessChain %7 %24 %9 %58
         %41 = OpLoad %6 %40
         %42 = OpAccessChain %7 %24 %33 %58
               OpStore %42 %41
         %45 = OpAccessChain %7 %24 %58 %33
         %46 = OpLoad %6 %45
         %47 = OpAccessChain %7 %24 %58 %9
               OpStore %47 %46
         %50 = OpAccessChain %7 %24 %58 %9
         %51 = OpLoad %6 %50
         %52 = OpAccessChain %7 %24 %58 %33
               OpStore %52 %51
         %53 = OpAccessChain %7 %24 %33 %9
         %54 = OpLoad %6 %53
         %55 = OpAccessChain %7 %24 %9 %33
               OpStore %55 %54
               OpBranch %13
         %13 = OpLabel
         %57 = OpIAdd %6 %58 %33
               OpStore %8 %57
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const ir::Function* f = spvtest::GetFunction(module, 4);
  ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

  ir::Loop* loop = &ld.GetLoopByIndex(0);
  std::vector<const ir::Loop*> loops{loop};
  opt::LoopDependenceAnalysis analysis{context.get(), loops};

  const ir::Instruction* store[6];
  int stores_found = 0;
  for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 11)) {
    if (inst.opcode() == SpvOp::SpvOpStore) {
      store[stores_found] = &inst;
      ++stores_found;
    }
  }

  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(store[i]);
  }

  // 30 -> 31
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_FALSE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(30),
                                        store[0], &distance_vector));
    EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
              opt::DistanceEntry::DependenceInformation::DISTANCE);
    EXPECT_EQ(distance_vector.GetEntries()[0].direction,
              opt::DistanceEntry::Directions::EQ);
    EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
  }

  // 36 -> 37
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(36),
                                       store[1], &distance_vector));
  }

  // 41 -> 42
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(41),
                                       store[2], &distance_vector));
  }

  // 46 -> 47
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(46),
                                       store[3], &distance_vector));
    EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
              opt::DistanceEntry::DependenceInformation::DISTANCE);
    EXPECT_EQ(distance_vector.GetEntries()[0].direction,
              opt::DistanceEntry::Directions::EQ);
    EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
  }

  // 51 -> 52
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(51),
                                       store[4], &distance_vector));
    EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
              opt::DistanceEntry::DependenceInformation::DISTANCE);
    EXPECT_EQ(distance_vector.GetEntries()[0].direction,
              opt::DistanceEntry::Directions::EQ);
    EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
  }

  // 54 -> 55
  {
    opt::DistanceVector distance_vector{loops.size()};
    EXPECT_TRUE(analysis.GetDependence(context->get_def_use_mgr()->GetDef(54),
                                       store[5], &distance_vector));
  }
}

/*
  Generated from the following GLSL fragment shader
  with --eliminate-local-multi-store
#version 440 core
void a(){
  int[10] arr;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      arr[j] = arr[j];
    }
  }
}
void b(){
  int[10] arr;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      arr[i] = arr[i];
    }
  }
}
void main() {
  a();
  b();
}
*/
TEST(DependencyAnalysis, IrrelevantSubscripts) {
  const std::string text = R"(               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 440
               OpName %4 "main"
               OpName %6 "a("
               OpName %8 "b("
               OpName %12 "i"
               OpName %23 "j"
               OpName %35 "arr"
               OpName %46 "i"
               OpName %54 "j"
               OpName %62 "arr"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %13 = OpConstant %10 0
         %20 = OpConstant %10 10
         %21 = OpTypeBool
         %31 = OpTypeInt 32 0
         %32 = OpConstant %31 10
         %33 = OpTypeArray %10 %32
         %34 = OpTypePointer Function %33
         %42 = OpConstant %10 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %72 = OpFunctionCall %2 %6
         %73 = OpFunctionCall %2 %8
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %12 = OpVariable %11 Function
         %23 = OpVariable %11 Function
         %35 = OpVariable %34 Function
               OpStore %12 %13
               OpBranch %14
         %14 = OpLabel
         %74 = OpPhi %10 %13 %7 %45 %17
               OpLoopMerge %16 %17 None
               OpBranch %18
         %18 = OpLabel
         %22 = OpSLessThan %21 %74 %20
               OpBranchConditional %22 %15 %16
         %15 = OpLabel
               OpStore %23 %13
               OpBranch %24
         %24 = OpLabel
         %75 = OpPhi %10 %13 %15 %43 %27
               OpLoopMerge %26 %27 None
               OpBranch %28
         %28 = OpLabel
         %30 = OpSLessThan %21 %75 %20
               OpBranchConditional %30 %25 %26
         %25 = OpLabel
         %38 = OpAccessChain %11 %35 %75
         %39 = OpLoad %10 %38
         %40 = OpAccessChain %11 %35 %75
               OpStore %40 %39
               OpBranch %27
         %27 = OpLabel
         %43 = OpIAdd %10 %75 %42
               OpStore %23 %43
               OpBranch %24
         %26 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %45 = OpIAdd %10 %74 %42
               OpStore %12 %45
               OpBranch %14
         %16 = OpLabel
               OpReturn
               OpFunctionEnd
          %8 = OpFunction %2 None %3
          %9 = OpLabel
         %46 = OpVariable %11 Function
         %54 = OpVariable %11 Function
         %62 = OpVariable %34 Function
               OpStore %46 %13
               OpBranch %47
         %47 = OpLabel
         %77 = OpPhi %10 %13 %9 %71 %50
               OpLoopMerge %49 %50 None
               OpBranch %51
         %51 = OpLabel
         %53 = OpSLessThan %21 %77 %20
               OpBranchConditional %53 %48 %49
         %48 = OpLabel
               OpStore %54 %13
               OpBranch %55
         %55 = OpLabel
         %78 = OpPhi %10 %13 %48 %69 %58
               OpLoopMerge %57 %58 None
               OpBranch %59
         %59 = OpLabel
         %61 = OpSLessThan %21 %78 %20
               OpBranchConditional %61 %56 %57
         %56 = OpLabel
         %65 = OpAccessChain %11 %62 %77
         %66 = OpLoad %10 %65
         %67 = OpAccessChain %11 %62 %77
               OpStore %67 %66
               OpBranch %58
         %58 = OpLabel
         %69 = OpIAdd %10 %78 %42
               OpStore %54 %69
               OpBranch %55
         %57 = OpLabel
               OpBranch %50
         %50 = OpLabel
         %71 = OpIAdd %10 %77 %42
               OpStore %46 %71
               OpBranch %47
         %49 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  // For the loop in function a
  {
    const ir::Function* f = spvtest::GetFunction(module, 6);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    std::vector<const ir::Loop*> loops{&ld.GetLoopByIndex(1),
                                       &ld.GetLoopByIndex(0)};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store[1];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 25)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 1; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // 39 -> 40
    {
      opt::DistanceVector distance_vector{loops.size()};
      analysis.SetDebugStream(std::cout);
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(39), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::IRRELEVANT);
      EXPECT_EQ(distance_vector.GetEntries()[1].dependence_information,
                opt::DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[1].distance, 0);
    }
  }

  // For the loop in function b
  {
    const ir::Function* f = spvtest::GetFunction(module, 8);
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(f);

    std::vector<const ir::Loop*> loops{&ld.GetLoopByIndex(1),
                                       &ld.GetLoopByIndex(0)};
    opt::LoopDependenceAnalysis analysis{context.get(), loops};

    const ir::Instruction* store[1];
    int stores_found = 0;
    for (const ir::Instruction& inst : *spvtest::GetBasicBlock(f, 56)) {
      if (inst.opcode() == SpvOp::SpvOpStore) {
        store[stores_found] = &inst;
        ++stores_found;
      }
    }

    for (int i = 0; i < 1; ++i) {
      EXPECT_TRUE(store[i]);
    }

    // 66 -> 67
    {
      opt::DistanceVector distance_vector{loops.size()};
      EXPECT_FALSE(analysis.GetDependence(
          context->get_def_use_mgr()->GetDef(66), store[0], &distance_vector));
      EXPECT_EQ(distance_vector.GetEntries()[0].dependence_information,
                opt::DistanceEntry::DependenceInformation::DISTANCE);
      EXPECT_EQ(distance_vector.GetEntries()[0].distance, 0);
      EXPECT_EQ(distance_vector.GetEntries()[1].dependence_information,
                opt::DistanceEntry::DependenceInformation::IRRELEVANT);
    }
  }
}

}  // namespace
