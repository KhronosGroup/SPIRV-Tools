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

#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "../assembly_builder.h"
#include "../function_utils.h"
#include "../pass_fixture.h"
#include "../pass_utils.h"
#include "opt/loop_unroller.h"
#include "opt/loop_utils.h"
#include "opt/pass.h"

namespace {

using namespace spvtools;
using ::testing::UnorderedElementsAre;

using PassClassTest = PassTest<::testing::Test>;

/*
Generated from the following GLSL
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  float x[4];
  for (int i = 0; i < 4; ++i) {
    x[i] = 1.0f;
  }
}
*/
TEST_F(PassClassTest, SimpleFullyUnrollTest) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(
            OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %2 "main" %3
            OpExecutionMode %2 OriginUpperLeft
            OpSource GLSL 330
            OpName %2 "main"
            OpName %5 "x"
            OpName %3 "c"
            OpDecorate %3 Location 0
            %6 = OpTypeVoid
            %7 = OpTypeFunction %6
            %8 = OpTypeInt 32 1
            %9 = OpTypePointer Function %8
            %10 = OpConstant %8 0
            %11 = OpConstant %8 4
            %12 = OpTypeBool
            %13 = OpTypeFloat 32
            %14 = OpTypeInt 32 0
            %15 = OpConstant %14 4
            %16 = OpTypeArray %13 %15
            %17 = OpTypePointer Function %16
            %18 = OpConstant %13 1
            %19 = OpTypePointer Function %13
            %20 = OpConstant %8 1
            %21 = OpTypeVector %13 4
            %22 = OpTypePointer Output %21
            %3 = OpVariable %22 Output
            %2 = OpFunction %6 None %7
            %23 = OpLabel
            %5 = OpVariable %17 Function
            OpBranch %24
            %24 = OpLabel
            %35 = OpPhi %8 %10 %23 %34 %26
            OpLoopMerge %25 %26 Unroll
            OpBranch %27
            %27 = OpLabel
            %29 = OpSLessThan %12 %35 %11
            OpBranchConditional %29 %30 %25
            %30 = OpLabel
            %32 = OpAccessChain %19 %5 %35
            OpStore %32 %18
            OpBranch %26
            %26 = OpLabel
            %34 = OpIAdd %8 %35 %20
            OpBranch %24
            %25 = OpLabel
            OpReturn
            OpFunctionEnd
  )";

const std::string output =
R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 330
OpName %2 "main"
OpName %4 "x"
OpName %3 "c"
OpDecorate %3 Location 0
%5 = OpTypeVoid
%6 = OpTypeFunction %5
%7 = OpTypeInt 32 1
%8 = OpTypePointer Function %7
%9 = OpConstant %7 0
%10 = OpConstant %7 4
%11 = OpTypeBool
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 4
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpConstant %12 1
%18 = OpTypePointer Function %12
%19 = OpConstant %7 1
%20 = OpTypeVector %12 4
%21 = OpTypePointer Output %20
%3 = OpVariable %21 Output
%2 = OpFunction %5 None %6
%22 = OpLabel
%4 = OpVariable %16 Function
OpBranch %23
%23 = OpLabel
OpBranch %28
%28 = OpLabel
%29 = OpSLessThan %11 %9 %10
OpBranch %30
%30 = OpLabel
%31 = OpAccessChain %18 %4 %9
OpStore %31 %17
OpBranch %26
%26 = OpLabel
%25 = OpIAdd %7 %9 %19
OpBranch %32
%32 = OpLabel
OpBranch %34
%34 = OpLabel
%35 = OpSLessThan %11 %25 %10
OpBranch %36
%36 = OpLabel
%37 = OpAccessChain %18 %4 %25
OpStore %37 %17
OpBranch %38
%38 = OpLabel
%39 = OpIAdd %7 %25 %19
OpBranch %40
%40 = OpLabel
OpBranch %42
%42 = OpLabel
%43 = OpSLessThan %11 %39 %10
OpBranch %44
%44 = OpLabel
%45 = OpAccessChain %18 %4 %39
OpStore %45 %17
OpBranch %46
%46 = OpLabel
%47 = OpIAdd %7 %39 %19
OpBranch %48
%48 = OpLabel
OpBranch %50
%50 = OpLabel
%51 = OpSLessThan %11 %47 %10
OpBranch %52
%52 = OpLabel
%53 = OpAccessChain %18 %4 %47
OpStore %53 %17
OpBranch %54
%54 = OpLabel
%55 = OpIAdd %7 %47 %19
OpBranch %27
%27 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, output, false);
}

template <int factor>
class PartialUnrollerTestPass : public opt::Pass {
 public:
  PartialUnrollerTestPass() : Pass() {}

  const char* name() const override { return "Loop unroller"; }

  Status Process(ir::IRContext* context) override {
    for (ir::Function& f : *context->module()) {
      ir::LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(&f);
      for (auto& loop : loop_descriptor) {
        opt::LoopUtils loop_utils{context, &loop};
        loop_utils.PartiallyUnroll(factor);
      }
    }

    return Pass::Status::SuccessWithChange;
  }
};

/*
Generated from the following GLSL
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  float x[10];
  for (int i = 0; i < 10; ++i) {
    x[i] = 1.0f;
  }
}
*/
TEST_F(PassClassTest, SimplePartialUnroll) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(
            OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %2 "main" %3
            OpExecutionMode %2 OriginUpperLeft
            OpSource GLSL 330
            OpName %2 "main"
            OpName %5 "x"
            OpName %3 "c"
            OpDecorate %3 Location 0
            %6 = OpTypeVoid
            %7 = OpTypeFunction %6
            %8 = OpTypeInt 32 1
            %9 = OpTypePointer Function %8
            %10 = OpConstant %8 0
            %11 = OpConstant %8 10
            %12 = OpTypeBool
            %13 = OpTypeFloat 32
            %14 = OpTypeInt 32 0
            %15 = OpConstant %14 10
            %16 = OpTypeArray %13 %15
            %17 = OpTypePointer Function %16
            %18 = OpConstant %13 1
            %19 = OpTypePointer Function %13
            %20 = OpConstant %8 1
            %21 = OpTypeVector %13 4
            %22 = OpTypePointer Output %21
            %3 = OpVariable %22 Output
            %2 = OpFunction %6 None %7
            %23 = OpLabel
            %5 = OpVariable %17 Function
            OpBranch %24
            %24 = OpLabel
            %35 = OpPhi %8 %10 %23 %34 %26
            OpLoopMerge %25 %26 Unroll
            OpBranch %27
            %27 = OpLabel
            %29 = OpSLessThan %12 %35 %11
            OpBranchConditional %29 %30 %25
            %30 = OpLabel
            %32 = OpAccessChain %19 %5 %35
            OpStore %32 %18
            OpBranch %26
            %26 = OpLabel
            %34 = OpIAdd %8 %35 %20
            OpBranch %24
            %25 = OpLabel
            OpReturn
            OpFunctionEnd
  )";

  const std::string output = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 330
OpName %2 "main"
OpName %4 "x"
OpName %3 "c"
OpDecorate %3 Location 0
%5 = OpTypeVoid
%6 = OpTypeFunction %5
%7 = OpTypeInt 32 1
%8 = OpTypePointer Function %7
%9 = OpConstant %7 0
%10 = OpConstant %7 10
%11 = OpTypeBool
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpConstant %12 1
%18 = OpTypePointer Function %12
%19 = OpConstant %7 1
%20 = OpTypeVector %12 4
%21 = OpTypePointer Output %20
%3 = OpVariable %21 Output
%2 = OpFunction %5 None %6
%22 = OpLabel
%4 = OpVariable %16 Function
OpBranch %23
%23 = OpLabel
%24 = OpPhi %7 %9 %22 %39 %38
OpLoopMerge %27 %38 Unroll
OpBranch %28
%28 = OpLabel
%29 = OpSLessThan %11 %24 %10
OpBranchConditional %29 %30 %27
%30 = OpLabel
%31 = OpAccessChain %18 %4 %24
OpStore %31 %17
OpBranch %26
%26 = OpLabel
%25 = OpIAdd %7 %24 %19
OpBranch %32
%32 = OpLabel
OpBranch %34
%34 = OpLabel
%35 = OpSLessThan %11 %25 %10
OpBranch %36
%36 = OpLabel
%37 = OpAccessChain %18 %4 %25
OpStore %37 %17
OpBranch %38
%38 = OpLabel
%39 = OpIAdd %7 %25 %19
OpBranch %23
%27 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, output, false);
}

/*
Generated from the following GLSL
#version 330 core
layout(location = 0) out vec4 c;
void main() {
  float x[10];
  for (int i = 0; i < 10; ++i) {
    x[i] = 1.0f;
  }
}
*/
TEST_F(PassClassTest, SimpleUnevenPartialUnroll) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(
            OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %2 "main" %3
            OpExecutionMode %2 OriginUpperLeft
            OpSource GLSL 330
            OpName %2 "main"
            OpName %5 "x"
            OpName %3 "c"
            OpDecorate %3 Location 0
            %6 = OpTypeVoid
            %7 = OpTypeFunction %6
            %8 = OpTypeInt 32 1
            %9 = OpTypePointer Function %8
            %10 = OpConstant %8 0
            %11 = OpConstant %8 10
            %12 = OpTypeBool
            %13 = OpTypeFloat 32
            %14 = OpTypeInt 32 0
            %15 = OpConstant %14 10
            %16 = OpTypeArray %13 %15
            %17 = OpTypePointer Function %16
            %18 = OpConstant %13 1
            %19 = OpTypePointer Function %13
            %20 = OpConstant %8 1
            %21 = OpTypeVector %13 4
            %22 = OpTypePointer Output %21
            %3 = OpVariable %22 Output
            %2 = OpFunction %6 None %7
            %23 = OpLabel
            %5 = OpVariable %17 Function
            OpBranch %24
            %24 = OpLabel
            %35 = OpPhi %8 %10 %23 %34 %26
            OpLoopMerge %25 %26 Unroll
            OpBranch %27
            %27 = OpLabel
            %29 = OpSLessThan %12 %35 %11
            OpBranchConditional %29 %30 %25
            %30 = OpLabel
            %32 = OpAccessChain %19 %5 %35
            OpStore %32 %18
            OpBranch %26
            %26 = OpLabel
            %34 = OpIAdd %8 %35 %20
            OpBranch %24
            %25 = OpLabel
            OpReturn
            OpFunctionEnd
  )";

const std::string output =
R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 330
OpName %2 "main"
OpName %4 "x"
OpName %3 "c"
OpDecorate %3 Location 0
%5 = OpTypeVoid
%6 = OpTypeFunction %5
%7 = OpTypeInt 32 1
%8 = OpTypePointer Function %7
%9 = OpConstant %7 0
%10 = OpConstant %7 10
%11 = OpTypeBool
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpConstant %12 1
%18 = OpTypePointer Function %12
%19 = OpConstant %7 1
%20 = OpTypeVector %12 4
%21 = OpTypePointer Output %20
%3 = OpVariable %21 Output
%58 = OpConstant %13 1
%2 = OpFunction %5 None %6
%22 = OpLabel
%4 = OpVariable %16 Function
OpBranch %23
%23 = OpLabel
%24 = OpPhi %7 %9 %22 %25 %26
OpLoopMerge %32 %26 Unroll
OpBranch %28
%28 = OpLabel
%29 = OpSLessThan %11 %24 %58
OpBranchConditional %29 %30 %32
%30 = OpLabel
%31 = OpAccessChain %18 %4 %24
OpStore %31 %17
OpBranch %26
%26 = OpLabel
%25 = OpIAdd %7 %24 %19
OpBranch %23
%32 = OpLabel
OpBranch %33
%33 = OpLabel
%34 = OpPhi %7 %58 %32 %57 %56
OpLoopMerge %41 %56 Unroll
OpBranch %35
%35 = OpLabel
%36 = OpSLessThan %11 %34 %10
OpBranchConditional %36 %37 %41
%37 = OpLabel
%38 = OpAccessChain %18 %4 %34
OpStore %38 %17
OpBranch %39
%39 = OpLabel
%40 = OpIAdd %7 %34 %19
OpBranch %42
%42 = OpLabel
OpBranch %44
%44 = OpLabel
%45 = OpSLessThan %11 %40 %10
OpBranch %46
%46 = OpLabel
%47 = OpAccessChain %18 %4 %40
OpStore %47 %17
OpBranch %48
%48 = OpLabel
%49 = OpIAdd %7 %40 %19
OpBranch %50
%50 = OpLabel
OpBranch %52
%52 = OpLabel
%53 = OpSLessThan %11 %49 %10
OpBranch %54
%54 = OpLabel
%55 = OpAccessChain %18 %4 %49
OpStore %55 %17
OpBranch %56
%56 = OpLabel
%57 = OpIAdd %7 %49 %19
OpBranch %33
%41 = OpLabel
OpReturn
%27 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  // By unrolling by a factor that doesn't divide evenly into the number of loop
  // iterations we perfom an additional transform when partially unrolling to
  // account for the remainder.
  SinglePassRunAndCheck<PartialUnrollerTestPass<3>>(text, output, false);
}

/* Generated from
#version 410 core
layout(location=0) flat in int upper_bound;
void main() {
    float x[10];
    for (int i = 2; i < 8; i+=2) {
        x[i] = i;
    }
}
*/
TEST_F(PassClassTest, SimpleLoopIterationsCheck) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %5 "x"
OpName %3 "upper_bound"
OpDecorate %3 Flat
OpDecorate %3 Location 0
%6 = OpTypeVoid
%7 = OpTypeFunction %6
%8 = OpTypeInt 32 1
%9 = OpTypePointer Function %8
%10 = OpConstant %8 2
%11 = OpConstant %8 8
%12 = OpTypeBool
%13 = OpTypeFloat 32
%14 = OpTypeInt 32 0
%15 = OpConstant %14 10
%16 = OpTypeArray %13 %15
%17 = OpTypePointer Function %16
%18 = OpTypePointer Function %13
%19 = OpTypePointer Input %8
%3 = OpVariable %19 Input
%2 = OpFunction %6 None %7
%20 = OpLabel
%5 = OpVariable %17 Function
OpBranch %21
%21 = OpLabel
%34 = OpPhi %8 %10 %20 %33 %23
OpLoopMerge %22 %23 Unroll
OpBranch %24
%24 = OpLabel
%26 = OpSLessThan %12 %34 %11
OpBranchConditional %26 %27 %22
%27 = OpLabel
%30 = OpConvertSToF %13 %34
%31 = OpAccessChain %18 %5 %34
OpStore %31 %30
OpBranch %23
%23 = OpLabel
%33 = OpIAdd %8 %34 %10
OpBranch %21
%22 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  ir::Function* f = spvtest::GetFunction(module, 2);

  ir::LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
  EXPECT_EQ(loop_descriptor.NumLoops(), 1u);

  ir::Loop& loop = loop_descriptor.GetLoopByIndex(0);

  EXPECT_TRUE(loop.HasUnrollLoopControl());

  ir::BasicBlock* condition = loop.FindConditionBlock();
  EXPECT_EQ(condition->id(), 24u);

  ir::Instruction* induction = loop.FindInductionVariable(condition);
  EXPECT_EQ(induction->result_id(), 34u);

  opt::LoopUtils loop_utils{context.get(), &loop};
  EXPECT_TRUE(loop_utils.CanPerformUnroll());

  size_t iterations = 0;
  EXPECT_TRUE(loop.FindNumberOfIterations(induction, &*condition->ctail(),
                                          &iterations));
  EXPECT_EQ(iterations, 3u);
}

/* Generated from
#version 410 core
void main() {
    float x[10];
    for (int i = -1; i < 6; i+=3) {
        x[i] = i;
    }
}
*/
TEST_F(PassClassTest, SimpleLoopIterationsCheckSignedInit) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %5 "x"
OpName %3 "upper_bound"
OpDecorate %3 Flat
OpDecorate %3 Location 0
%6 = OpTypeVoid
%7 = OpTypeFunction %6
%8 = OpTypeInt 32 1
%9 = OpTypePointer Function %8
%10 = OpConstant %8 -1
%11 = OpConstant %8 6
%12 = OpTypeBool
%13 = OpTypeFloat 32
%14 = OpTypeInt 32 0
%15 = OpConstant %14 10
%16 = OpTypeArray %13 %15
%17 = OpTypePointer Function %16
%18 = OpTypePointer Function %13
%19 = OpConstant %8 3
%20 = OpTypePointer Input %8
%3 = OpVariable %20 Input
%2 = OpFunction %6 None %7
%21 = OpLabel
%5 = OpVariable %17 Function
OpBranch %22
%22 = OpLabel
%35 = OpPhi %8 %10 %21 %34 %24
OpLoopMerge %23 %24 None
OpBranch %25
%25 = OpLabel
%27 = OpSLessThan %12 %35 %11
OpBranchConditional %27 %28 %23
%28 = OpLabel
%31 = OpConvertSToF %13 %35
%32 = OpAccessChain %18 %5 %35
OpStore %32 %31
OpBranch %24
%24 = OpLabel
%34 = OpIAdd %8 %35 %19
OpBranch %22
%23 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  ir::Function* f = spvtest::GetFunction(module, 2);

  ir::LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);

  EXPECT_EQ(loop_descriptor.NumLoops(), 1u);

  ir::Loop& loop = loop_descriptor.GetLoopByIndex(0);

  EXPECT_FALSE(loop.HasUnrollLoopControl());

  ir::BasicBlock* condition = loop.FindConditionBlock();
  EXPECT_EQ(condition->id(), 25u);

  ir::Instruction* induction = loop.FindInductionVariable(condition);
  EXPECT_EQ(induction->result_id(), 35u);

  opt::LoopUtils loop_utils{context.get(), &loop};
  EXPECT_TRUE(loop_utils.CanPerformUnroll());

  size_t iterations = 0;
  EXPECT_TRUE(loop.FindNumberOfIterations(induction, &*condition->ctail(),
                                          &iterations));
  EXPECT_EQ(iterations, 3u);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[6];
    for (uint i = 0; i < 2; i++) {
      for (int x = 0; x < 3; ++x) {
        out_array[x + i*3] = i;
      }
    }
}
*/
TEST_F(PassClassTest, UnrollNestedLoops) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %35 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 0
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 2
         %17 = OpTypeBool
         %19 = OpTypeInt 32 1
         %20 = OpTypePointer Function %19
         %22 = OpConstant %19 0
         %29 = OpConstant %19 3
         %31 = OpTypeFloat 32
         %32 = OpConstant %6 6
         %33 = OpTypeArray %31 %32
         %34 = OpTypePointer Function %33
         %39 = OpConstant %6 3
         %44 = OpTypePointer Function %31
         %47 = OpConstant %19 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %35 = OpVariable %34 Function
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %50 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpULessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %54 = OpPhi %19 %22 %11 %48 %26
               OpLoopMerge %25 %26 Unroll
               OpBranch %27
         %27 = OpLabel
         %30 = OpSLessThan %17 %54 %29
               OpBranchConditional %30 %24 %25
         %24 = OpLabel
         %37 = OpBitcast %6 %54
         %40 = OpIMul %6 %51 %39
         %41 = OpIAdd %6 %37 %40
         %43 = OpConvertUToF %31 %51
         %45 = OpAccessChain %44 %35 %41
               OpStore %45 %43
               OpBranch %26
         %26 = OpLabel
         %48 = OpIAdd %19 %54 %47
               OpBranch %23
         %25 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %50 = OpIAdd %6 %51 %47
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

const std::string output =
R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 0
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 2
%10 = OpTypeBool
%11 = OpTypeInt 32 1
%12 = OpTypePointer Function %11
%13 = OpConstant %11 0
%14 = OpConstant %11 3
%15 = OpTypeFloat 32
%16 = OpConstant %6 6
%17 = OpTypeArray %15 %16
%18 = OpTypePointer Function %17
%19 = OpConstant %6 3
%20 = OpTypePointer Function %15
%21 = OpConstant %11 1
%2 = OpFunction %4 None %5
%22 = OpLabel
%3 = OpVariable %18 Function
OpBranch %23
%23 = OpLabel
OpBranch %28
%28 = OpLabel
%29 = OpULessThan %10 %8 %9
OpBranch %30
%30 = OpLabel
OpBranch %31
%31 = OpLabel
OpBranch %36
%36 = OpLabel
%37 = OpSLessThan %10 %13 %14
OpBranch %38
%38 = OpLabel
%39 = OpBitcast %6 %13
%40 = OpIMul %6 %8 %19
%41 = OpIAdd %6 %39 %40
%42 = OpConvertUToF %15 %8
%43 = OpAccessChain %20 %3 %41
OpStore %43 %42
OpBranch %34
%34 = OpLabel
%33 = OpIAdd %11 %13 %21
OpBranch %44
%44 = OpLabel
OpBranch %46
%46 = OpLabel
%47 = OpSLessThan %10 %33 %14
OpBranch %48
%48 = OpLabel
%49 = OpBitcast %6 %33
%50 = OpIMul %6 %8 %19
%51 = OpIAdd %6 %49 %50
%52 = OpConvertUToF %15 %8
%53 = OpAccessChain %20 %3 %51
OpStore %53 %52
OpBranch %54
%54 = OpLabel
%55 = OpIAdd %11 %33 %21
OpBranch %56
%56 = OpLabel
OpBranch %58
%58 = OpLabel
%59 = OpSLessThan %10 %55 %14
OpBranch %60
%60 = OpLabel
%61 = OpBitcast %6 %55
%62 = OpIMul %6 %8 %19
%63 = OpIAdd %6 %61 %62
%64 = OpConvertUToF %15 %8
%65 = OpAccessChain %20 %3 %63
OpStore %65 %64
OpBranch %66
%66 = OpLabel
%67 = OpIAdd %11 %55 %21
OpBranch %35
%35 = OpLabel
OpBranch %26
%26 = OpLabel
%25 = OpIAdd %6 %8 %21
OpBranch %68
%68 = OpLabel
OpBranch %70
%70 = OpLabel
%71 = OpULessThan %10 %25 %9
OpBranch %72
%72 = OpLabel
OpBranch %73
%73 = OpLabel
OpBranch %74
%74 = OpLabel
%75 = OpSLessThan %10 %13 %14
OpBranch %76
%76 = OpLabel
%77 = OpBitcast %6 %13
%78 = OpIMul %6 %25 %19
%79 = OpIAdd %6 %77 %78
%80 = OpConvertUToF %15 %25
%81 = OpAccessChain %20 %3 %79
OpStore %81 %80
OpBranch %82
%82 = OpLabel
%83 = OpIAdd %11 %13 %21
OpBranch %84
%84 = OpLabel
OpBranch %85
%85 = OpLabel
%86 = OpSLessThan %10 %83 %14
OpBranch %87
%87 = OpLabel
%88 = OpBitcast %6 %83
%89 = OpIMul %6 %25 %19
%90 = OpIAdd %6 %88 %89
%91 = OpConvertUToF %15 %25
%92 = OpAccessChain %20 %3 %90
OpStore %92 %91
OpBranch %93
%93 = OpLabel
%94 = OpIAdd %11 %83 %21
OpBranch %95
%95 = OpLabel
OpBranch %96
%96 = OpLabel
%97 = OpSLessThan %10 %94 %14
OpBranch %98
%98 = OpLabel
%99 = OpBitcast %6 %94
%100 = OpIMul %6 %25 %19
%101 = OpIAdd %6 %99 %100
%102 = OpConvertUToF %15 %25
%103 = OpAccessChain %20 %3 %101
OpStore %103 %102
OpBranch %104
%104 = OpLabel
%105 = OpIAdd %11 %94 %21
OpBranch %106
%106 = OpLabel
OpBranch %107
%107 = OpLabel
%108 = OpIAdd %6 %25 %21
OpBranch %27
%27 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;
  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, output, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[2];
    for (int i = -3; i < -1; i++) {
      out_array[3 + i] = i;
    }
}
*/
TEST_F(PassClassTest, NegativeConditionAndInit) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %23 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 -3
         %16 = OpConstant %6 -1
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 2
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %25 = OpConstant %6 3
         %30 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %23 = OpVariable %22 Function
               OpBranch %10
         %10 = OpLabel
         %32 = OpPhi %6 %9 %5 %31 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %32 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpIAdd %6 %32 %25
         %28 = OpAccessChain %7 %23 %26
               OpStore %28 %32
               OpBranch %13
         %13 = OpLabel
         %31 = OpIAdd %6 %32 %30
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";

const std::string expected = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 -3
%9 = OpConstant %6 -1
%10 = OpTypeBool
%11 = OpTypeInt 32 0
%12 = OpConstant %11 2
%13 = OpTypeArray %6 %12
%14 = OpTypePointer Function %13
%15 = OpConstant %6 3
%16 = OpConstant %6 1
%2 = OpFunction %4 None %5
%17 = OpLabel
%3 = OpVariable %14 Function
OpBranch %18
%18 = OpLabel
OpBranch %23
%23 = OpLabel
%24 = OpSLessThan %10 %8 %9
OpBranch %25
%25 = OpLabel
%26 = OpIAdd %6 %8 %15
%27 = OpAccessChain %7 %3 %26
OpStore %27 %8
OpBranch %21
%21 = OpLabel
%20 = OpIAdd %6 %8 %16
OpBranch %28
%28 = OpLabel
OpBranch %30
%30 = OpLabel
%31 = OpSLessThan %10 %20 %9
OpBranch %32
%32 = OpLabel
%33 = OpIAdd %6 %20 %15
%34 = OpAccessChain %7 %3 %33
OpStore %34 %20
OpBranch %35
%35 = OpLabel
%36 = OpIAdd %6 %20 %16
OpBranch %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  // SinglePassRunAndCheck<opt::LoopUnroller>(text, expected, false);

  ir::Function* f = spvtest::GetFunction(module, 4);

  ir::LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
  EXPECT_EQ(loop_descriptor.NumLoops(), 1u);

  ir::Loop& loop = loop_descriptor.GetLoopByIndex(0);

  EXPECT_TRUE(loop.HasUnrollLoopControl());

  ir::BasicBlock* condition = loop.FindConditionBlock();
  EXPECT_EQ(condition->id(), 14u);

  ir::Instruction* induction = loop.FindInductionVariable(condition);
  EXPECT_EQ(induction->result_id(), 32u);

  opt::LoopUtils loop_utils{context.get(), &loop};
  EXPECT_TRUE(loop_utils.CanPerformUnroll());

  size_t iterations = 0;
  EXPECT_TRUE(loop.FindNumberOfIterations(induction, &*condition->ctail(),
                                          &iterations));
  EXPECT_EQ(iterations, 2u);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, expected, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[9];
    for (int i = -10; i < -1; i++) {
      out_array[i] = i;
    }
}
*/
TEST_F(PassClassTest, NegativeConditionAndInitResidualUnroll) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %23 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 -10
         %16 = OpConstant %6 -1
         %17 = OpTypeBool
         %19 = OpTypeInt 32 0
         %20 = OpConstant %19 9
         %21 = OpTypeArray %6 %20
         %22 = OpTypePointer Function %21
         %25 = OpConstant %6 10
         %30 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %23 = OpVariable %22 Function
               OpBranch %10
         %10 = OpLabel
         %32 = OpPhi %6 %9 %5 %31 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %17 %32 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %26 = OpIAdd %6 %32 %25
         %28 = OpAccessChain %7 %23 %26
               OpStore %28 %32
               OpBranch %13
         %13 = OpLabel
         %31 = OpIAdd %6 %32 %30
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";

const std::string expected = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 -10
%9 = OpConstant %6 -1
%10 = OpTypeBool
%11 = OpTypeInt 32 0
%12 = OpConstant %11 9
%13 = OpTypeArray %6 %12
%14 = OpTypePointer Function %13
%15 = OpConstant %6 10
%16 = OpConstant %6 1
%48 = OpConstant %6 -9
%2 = OpFunction %4 None %5
%17 = OpLabel
%3 = OpVariable %14 Function
OpBranch %18
%18 = OpLabel
%19 = OpPhi %6 %8 %17 %20 %21
OpLoopMerge %28 %21 Unroll
OpBranch %23
%23 = OpLabel
%24 = OpSLessThan %10 %19 %48
OpBranchConditional %24 %25 %28
%25 = OpLabel
%26 = OpIAdd %6 %19 %15
%27 = OpAccessChain %7 %3 %26
OpStore %27 %19
OpBranch %21
%21 = OpLabel
%20 = OpIAdd %6 %19 %16
OpBranch %18
%28 = OpLabel
OpBranch %29
%29 = OpLabel
%30 = OpPhi %6 %48 %28 %47 %46
OpLoopMerge %38 %46 Unroll
OpBranch %31
%31 = OpLabel
%32 = OpSLessThan %10 %30 %9
OpBranchConditional %32 %33 %38
%33 = OpLabel
%34 = OpIAdd %6 %30 %15
%35 = OpAccessChain %7 %3 %34
OpStore %35 %30
OpBranch %36
%36 = OpLabel
%37 = OpIAdd %6 %30 %16
OpBranch %39
%39 = OpLabel
OpBranch %41
%41 = OpLabel
%42 = OpSLessThan %10 %37 %9
OpBranch %43
%43 = OpLabel
%44 = OpIAdd %6 %37 %15
%45 = OpAccessChain %7 %3 %44
OpStore %45 %37
OpBranch %46
%46 = OpLabel
%47 = OpIAdd %6 %37 %16
OpBranch %29
%38 = OpLabel
OpReturn
%22 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  ir::Function* f = spvtest::GetFunction(module, 4);

  ir::LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
  EXPECT_EQ(loop_descriptor.NumLoops(), 1u);

  ir::Loop& loop = loop_descriptor.GetLoopByIndex(0);

  EXPECT_TRUE(loop.HasUnrollLoopControl());

  ir::BasicBlock* condition = loop.FindConditionBlock();
  EXPECT_EQ(condition->id(), 14u);

  ir::Instruction* induction = loop.FindInductionVariable(condition);
  EXPECT_EQ(induction->result_id(), 32u);

  opt::LoopUtils loop_utils{context.get(), &loop};
  EXPECT_TRUE(loop_utils.CanPerformUnroll());

  size_t iterations = 0;
  EXPECT_TRUE(loop.FindNumberOfIterations(induction, &*condition->ctail(),
                                          &iterations));
  EXPECT_EQ(iterations, 9u);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, expected, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[10];
    for (uint i = 0; i < 2; i++) {
      for (int x = 0; x < 5; ++x) {
        out_array[x + i*5] = i;
      }
    }
}
*/
TEST_F(PassClassTest, UnrollNestedLoopsValidateDescriptor) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %35 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 0
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %16 = OpConstant %6 2
         %17 = OpTypeBool
         %19 = OpTypeInt 32 1
         %20 = OpTypePointer Function %19
         %22 = OpConstant %19 0
         %29 = OpConstant %19 5
         %31 = OpTypeFloat 32
         %32 = OpConstant %6 10
         %33 = OpTypeArray %31 %32
         %34 = OpTypePointer Function %33
         %39 = OpConstant %6 5
         %44 = OpTypePointer Function %31
         %47 = OpConstant %19 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %35 = OpVariable %34 Function
               OpBranch %10
         %10 = OpLabel
         %51 = OpPhi %6 %9 %5 %50 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpULessThan %17 %51 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %23
         %23 = OpLabel
         %54 = OpPhi %19 %22 %11 %48 %26
               OpLoopMerge %25 %26 Unroll
               OpBranch %27
         %27 = OpLabel
         %30 = OpSLessThan %17 %54 %29
               OpBranchConditional %30 %24 %25
         %24 = OpLabel
         %37 = OpBitcast %6 %54
         %40 = OpIMul %6 %51 %39
         %41 = OpIAdd %6 %37 %40
         %43 = OpConvertUToF %31 %51
         %45 = OpAccessChain %44 %35 %41
               OpStore %45 %43
               OpBranch %26
         %26 = OpLabel
         %48 = OpIAdd %19 %54 %47
               OpBranch %23
         %25 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %50 = OpIAdd %6 %51 %47
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

  // clang-format on

  {  // Test fully unroll
    std::unique_ptr<ir::IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ir::Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                               << text << std::endl;
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

    ir::Function* f = spvtest::GetFunction(module, 4);
    ir::LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
    EXPECT_EQ(loop_descriptor.NumLoops(), 2u);

    ir::Loop& outer_loop = loop_descriptor.GetLoopByIndex(1);

    EXPECT_TRUE(outer_loop.HasUnrollLoopControl());

    ir::Loop& inner_loop = loop_descriptor.GetLoopByIndex(0);

    EXPECT_TRUE(inner_loop.HasUnrollLoopControl());

    EXPECT_EQ(outer_loop.GetBlocks().size(), 9u);

    EXPECT_EQ(inner_loop.GetBlocks().size(), 4u);
    EXPECT_EQ(outer_loop.NumImmediateChildren(), 1u);
    EXPECT_EQ(inner_loop.NumImmediateChildren(), 0u);

    {
      opt::LoopUtils loop_utils{context.get(), &inner_loop};
      loop_utils.FullyUnroll();
      loop_utils.Finalize();
    }

    EXPECT_EQ(loop_descriptor.NumLoops(), 1u);
    EXPECT_EQ(outer_loop.GetBlocks().size(), 25u);
    EXPECT_EQ(outer_loop.NumImmediateChildren(), 0u);
    {
      opt::LoopUtils loop_utils{context.get(), &outer_loop};
      loop_utils.FullyUnroll();
      loop_utils.Finalize();
    }
    EXPECT_EQ(loop_descriptor.NumLoops(), 0u);
  }

  {  // Test partially unroll
    std::unique_ptr<ir::IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ir::Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                               << text << std::endl;
    SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

    ir::Function* f = spvtest::GetFunction(module, 4);
    ir::LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(f);
    EXPECT_EQ(loop_descriptor.NumLoops(), 2u);

    ir::Loop& outer_loop = loop_descriptor.GetLoopByIndex(1);

    EXPECT_TRUE(outer_loop.HasUnrollLoopControl());

    ir::Loop& inner_loop = loop_descriptor.GetLoopByIndex(0);

    EXPECT_TRUE(inner_loop.HasUnrollLoopControl());

    EXPECT_EQ(outer_loop.GetBlocks().size(), 9u);

    EXPECT_EQ(inner_loop.GetBlocks().size(), 4u);

    EXPECT_EQ(outer_loop.NumImmediateChildren(), 1u);
    EXPECT_EQ(inner_loop.NumImmediateChildren(), 0u);

    opt::LoopUtils loop_utils{context.get(), &inner_loop};
    loop_utils.PartiallyUnroll(2);
    loop_utils.Finalize();

    // The number of loops should actually grow.
    EXPECT_EQ(loop_descriptor.NumLoops(), 3u);
    EXPECT_EQ(outer_loop.GetBlocks().size(), 19u);
    EXPECT_EQ(outer_loop.NumImmediateChildren(), 2u);
  }
}

/*
Generated from the following GLSL
#version 440 core
void main(){
  float x[10];
  int i = 1;
  i = 0;
  for (; i < 10; i++) {
    x[i] = i;
  }
}
*/
TEST_F(PassClassTest, UnrollWithInductionOutsideHeader) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 440
OpName %main "main"
OpName %x "x"
%void = OpTypeVoid
%3 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_1 = OpConstant %int 1
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%_ptr_Function__arr_float_uint_10 = OpTypePointer Function %_arr_float_uint_10
%_ptr_Function_float = OpTypePointer Function %float
%main = OpFunction %void None %3
%5 = OpLabel
%x = OpVariable %_ptr_Function__arr_float_uint_10 Function
OpBranch %11
%11 = OpLabel
%33 = OpPhi %int %int_0 %5 %32 %14
OpLoopMerge %13 %14 None
OpBranch %15
%15 = OpLabel
%19 = OpSLessThan %bool %33 %int_10
OpBranchConditional %19 %12 %13
%12 = OpLabel
%28 = OpConvertSToF %float %33
%30 = OpAccessChain %_ptr_Function_float %x %33
OpStore %30 %28
OpBranch %14
%14 = OpLabel
%32 = OpIAdd %int %33 %int_1
OpBranch %11
%13 = OpLabel
OpReturn
OpFunctionEnd
)";

const std::string expected = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 440
OpName %main "main"
OpName %x "x"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_1 = OpConstant %int 1
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%uint_10 = OpConstant %uint 10
%_arr_float_uint_10 = OpTypeArray %float %uint_10
%_ptr_Function__arr_float_uint_10 = OpTypePointer Function %_arr_float_uint_10
%_ptr_Function_float = OpTypePointer Function %float
%main = OpFunction %void None %5
%18 = OpLabel
%x = OpVariable %_ptr_Function__arr_float_uint_10 Function
OpBranch %19
%19 = OpLabel
%20 = OpPhi %int %int_0 %18 %37 %36
OpLoopMerge %23 %36 None
OpBranch %24
%24 = OpLabel
%25 = OpSLessThan %bool %20 %int_10
OpBranchConditional %25 %26 %23
%26 = OpLabel
%27 = OpConvertSToF %float %20
%28 = OpAccessChain %_ptr_Function_float %x %20
OpStore %28 %27
OpBranch %22
%22 = OpLabel
%21 = OpIAdd %int %20 %int_1
OpBranch %29
%29 = OpLabel
OpBranch %31
%31 = OpLabel
%32 = OpSLessThan %bool %21 %int_10
OpBranch %33
%33 = OpLabel
%34 = OpConvertSToF %float %21
%35 = OpAccessChain %_ptr_Function_float %x %21
OpStore %35 %34
OpBranch %36
%36 = OpLabel
%37 = OpIAdd %int %21 %int_1
OpBranch %19
%23 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on

  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, expected, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
  float out_array[3];
  for (int i = 3; i > 0; --i) {
    out_array[i] = i;
  }
}
*/
TEST_F(PassClassTest, FullyUnrollNegativeStepLoopTest) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %24 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 3
         %16 = OpConstant %6 0
         %17 = OpTypeBool
         %19 = OpTypeFloat 32
         %20 = OpTypeInt 32 0
         %21 = OpConstant %20 3
         %22 = OpTypeArray %19 %21
         %23 = OpTypePointer Function %22
         %28 = OpTypePointer Function %19
         %31 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %24 = OpVariable %23 Function
               OpBranch %10
         %10 = OpLabel
         %33 = OpPhi %6 %9 %5 %32 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpSGreaterThan %17 %33 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpConvertSToF %19 %33
         %29 = OpAccessChain %28 %24 %33
               OpStore %29 %27
               OpBranch %13
         %13 = OpLabel
         %32 = OpISub %6 %33 %31
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

const std::string output =
R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 3
%9 = OpConstant %6 0
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 3
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 1
%2 = OpFunction %4 None %5
%18 = OpLabel
%3 = OpVariable %15 Function
OpBranch %19
%19 = OpLabel
OpBranch %24
%24 = OpLabel
%25 = OpSGreaterThan %10 %8 %9
OpBranch %26
%26 = OpLabel
%27 = OpConvertSToF %11 %8
%28 = OpAccessChain %16 %3 %8
OpStore %28 %27
OpBranch %22
%22 = OpLabel
%21 = OpISub %6 %8 %17
OpBranch %29
%29 = OpLabel
OpBranch %31
%31 = OpLabel
%32 = OpSGreaterThan %10 %21 %9
OpBranch %33
%33 = OpLabel
%34 = OpConvertSToF %11 %21
%35 = OpAccessChain %16 %3 %21
OpStore %35 %34
OpBranch %36
%36 = OpLabel
%37 = OpISub %6 %21 %17
OpBranch %38
%38 = OpLabel
OpBranch %40
%40 = OpLabel
%41 = OpSGreaterThan %10 %37 %9
OpBranch %42
%42 = OpLabel
%43 = OpConvertSToF %11 %37
%44 = OpAccessChain %16 %3 %37
OpStore %44 %43
OpBranch %45
%45 = OpLabel
%46 = OpISub %6 %37 %17
OpBranch %23
%23 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, output, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
  float out_array[3];
  for (int i = 9; i > 0; i-=3) {
    out_array[i] = i;
  }
}
*/
TEST_F(PassClassTest, FullyUnrollNegativeNonOneStepLoop) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 410
               OpName %4 "main"
               OpName %24 "out_array"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 9
         %16 = OpConstant %6 0
         %17 = OpTypeBool
         %19 = OpTypeFloat 32
         %20 = OpTypeInt 32 0
         %21 = OpConstant %20 3
         %22 = OpTypeArray %19 %21
         %23 = OpTypePointer Function %22
         %28 = OpTypePointer Function %19
         %30 = OpConstant %6 3
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %24 = OpVariable %23 Function
               OpBranch %10
         %10 = OpLabel
         %33 = OpPhi %6 %9 %5 %32 %13
               OpLoopMerge %12 %13 Unroll
               OpBranch %14
         %14 = OpLabel
         %18 = OpSGreaterThan %17 %33 %16
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %27 = OpConvertSToF %19 %33
         %29 = OpAccessChain %28 %24 %33
               OpStore %29 %27
               OpBranch %13
         %13 = OpLabel
         %32 = OpISub %6 %33 %30
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
    )";

const std::string output =
R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 9
%9 = OpConstant %6 0
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 3
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 3
%2 = OpFunction %4 None %5
%18 = OpLabel
%3 = OpVariable %15 Function
OpBranch %19
%19 = OpLabel
OpBranch %24
%24 = OpLabel
%25 = OpSGreaterThan %10 %8 %9
OpBranch %26
%26 = OpLabel
%27 = OpConvertSToF %11 %8
%28 = OpAccessChain %16 %3 %8
OpStore %28 %27
OpBranch %22
%22 = OpLabel
%21 = OpISub %6 %8 %17
OpBranch %29
%29 = OpLabel
OpBranch %31
%31 = OpLabel
%32 = OpSGreaterThan %10 %21 %9
OpBranch %33
%33 = OpLabel
%34 = OpConvertSToF %11 %21
%35 = OpAccessChain %16 %3 %21
OpStore %35 %34
OpBranch %36
%36 = OpLabel
%37 = OpISub %6 %21 %17
OpBranch %38
%38 = OpLabel
OpBranch %40
%40 = OpLabel
%41 = OpSGreaterThan %10 %37 %9
OpBranch %42
%42 = OpLabel
%43 = OpConvertSToF %11 %37
%44 = OpAccessChain %16 %3 %37
OpStore %44 %43
OpBranch %45
%45 = OpLabel
%46 = OpISub %6 %37 %17
OpBranch %23
%23 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, output, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
  float out_array[3];
  for (int i = 0; i < 7; i+=3) {
    out_array[i] = i;
  }
}
*/
TEST_F(PassClassTest, FullyUnrollNonDivisibleStepLoop) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %4 "main"
OpExecutionMode %4 OriginUpperLeft
OpSource GLSL 410
OpName %4 "main"
OpName %24 "out_array"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%9 = OpConstant %6 0
%16 = OpConstant %6 7
%17 = OpTypeBool
%19 = OpTypeFloat 32
%20 = OpTypeInt 32 0
%21 = OpConstant %20 3
%22 = OpTypeArray %19 %21
%23 = OpTypePointer Function %22
%28 = OpTypePointer Function %19
%30 = OpConstant %6 3
%4 = OpFunction %2 None %3
%5 = OpLabel
%24 = OpVariable %23 Function
OpBranch %10
%10 = OpLabel
%33 = OpPhi %6 %9 %5 %32 %13
OpLoopMerge %12 %13 Unroll
OpBranch %14
%14 = OpLabel
%18 = OpSLessThan %17 %33 %16
OpBranchConditional %18 %11 %12
%11 = OpLabel
%27 = OpConvertSToF %19 %33
%29 = OpAccessChain %28 %24 %33
OpStore %29 %27
OpBranch %13
%13 = OpLabel
%32 = OpIAdd %6 %33 %30
OpBranch %10
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

const std::string output =
R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 7
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 3
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 3
%2 = OpFunction %4 None %5
%18 = OpLabel
%3 = OpVariable %15 Function
OpBranch %19
%19 = OpLabel
OpBranch %24
%24 = OpLabel
%25 = OpSLessThan %10 %8 %9
OpBranch %26
%26 = OpLabel
%27 = OpConvertSToF %11 %8
%28 = OpAccessChain %16 %3 %8
OpStore %28 %27
OpBranch %22
%22 = OpLabel
%21 = OpIAdd %6 %8 %17
OpBranch %29
%29 = OpLabel
OpBranch %31
%31 = OpLabel
%32 = OpSLessThan %10 %21 %9
OpBranch %33
%33 = OpLabel
%34 = OpConvertSToF %11 %21
%35 = OpAccessChain %16 %3 %21
OpStore %35 %34
OpBranch %36
%36 = OpLabel
%37 = OpIAdd %6 %21 %17
OpBranch %38
%38 = OpLabel
OpBranch %40
%40 = OpLabel
%41 = OpSLessThan %10 %37 %9
OpBranch %42
%42 = OpLabel
%43 = OpConvertSToF %11 %37
%44 = OpAccessChain %16 %3 %37
OpStore %44 %43
OpBranch %45
%45 = OpLabel
%46 = OpIAdd %6 %37 %17
OpBranch %23
%23 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, output, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
  float out_array[4];
  for (int i = 11; i > 0; i-=3) {
    out_array[i] = i;
  }
}
*/
TEST_F(PassClassTest, FullyUnrollNegativeNonDivisibleStepLoop) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %4 "main"
OpExecutionMode %4 OriginUpperLeft
OpSource GLSL 410
OpName %4 "main"
OpName %24 "out_array"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%9 = OpConstant %6 11
%16 = OpConstant %6 0
%17 = OpTypeBool
%19 = OpTypeFloat 32
%20 = OpTypeInt 32 0
%21 = OpConstant %20 4
%22 = OpTypeArray %19 %21
%23 = OpTypePointer Function %22
%28 = OpTypePointer Function %19
%30 = OpConstant %6 3
%4 = OpFunction %2 None %3
%5 = OpLabel
%24 = OpVariable %23 Function
OpBranch %10
%10 = OpLabel
%33 = OpPhi %6 %9 %5 %32 %13
OpLoopMerge %12 %13 Unroll
OpBranch %14
%14 = OpLabel
%18 = OpSGreaterThan %17 %33 %16
OpBranchConditional %18 %11 %12
%11 = OpLabel
%27 = OpConvertSToF %19 %33
%29 = OpAccessChain %28 %24 %33
OpStore %29 %27
OpBranch %13
%13 = OpLabel
%32 = OpISub %6 %33 %30
OpBranch %10
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

const std::string output =
R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "out_array"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 11
%9 = OpConstant %6 0
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 4
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 3
%2 = OpFunction %4 None %5
%18 = OpLabel
%3 = OpVariable %15 Function
OpBranch %19
%19 = OpLabel
OpBranch %24
%24 = OpLabel
%25 = OpSGreaterThan %10 %8 %9
OpBranch %26
%26 = OpLabel
%27 = OpConvertSToF %11 %8
%28 = OpAccessChain %16 %3 %8
OpStore %28 %27
OpBranch %22
%22 = OpLabel
%21 = OpISub %6 %8 %17
OpBranch %29
%29 = OpLabel
OpBranch %31
%31 = OpLabel
%32 = OpSGreaterThan %10 %21 %9
OpBranch %33
%33 = OpLabel
%34 = OpConvertSToF %11 %21
%35 = OpAccessChain %16 %3 %21
OpStore %35 %34
OpBranch %36
%36 = OpLabel
%37 = OpISub %6 %21 %17
OpBranch %38
%38 = OpLabel
OpBranch %40
%40 = OpLabel
%41 = OpSGreaterThan %10 %37 %9
OpBranch %42
%42 = OpLabel
%43 = OpConvertSToF %11 %37
%44 = OpAccessChain %16 %3 %37
OpStore %44 %43
OpBranch %45
%45 = OpLabel
%46 = OpISub %6 %37 %17
OpBranch %47
%47 = OpLabel
OpBranch %49
%49 = OpLabel
%50 = OpSGreaterThan %10 %46 %9
OpBranch %51
%51 = OpLabel
%52 = OpConvertSToF %11 %46
%53 = OpAccessChain %16 %3 %46
OpStore %53 %52
OpBranch %54
%54 = OpLabel
%55 = OpISub %6 %46 %17
OpBranch %23
%23 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for ushader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, output, false);
}

}  // namespace
