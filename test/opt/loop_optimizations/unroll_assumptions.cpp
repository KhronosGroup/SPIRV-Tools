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

template <int factor>
class PartialUnrollerTestPass : public opt::Pass {
 public:
  PartialUnrollerTestPass() : Pass() {}

  const char* name() const override { return "Loop unroller"; }

  Status Process(ir::IRContext* context) override {
    bool changed = false;
    for (ir::Function& f : *context->module()) {
      ir::LoopDescriptor& loop_descriptor = *context->GetLoopDescriptor(&f);
      for (auto& loop : loop_descriptor) {
        opt::LoopUtils loop_utils{context, &loop};
        if (loop_utils.PartiallyUnroll(factor)) {
          changed = true;
        }
      }
    }

    if (changed) return Pass::Status::SuccessWithChange;
    return Pass::Status::SuccessWithoutChange;
  }
};

using PassClassTest = PassTest<::testing::Test>;

/*
Generated from the following GLSL
#version 410 core
layout(location = 0) flat in int in_upper_bound;
void main() {
  for (int i = ; i < in_upper_bound; ++i) {
    x[i] = 1.0f;
  }
}
*/
TEST_F(PassClassTest, CheckUpperBound) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main" %3
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 410
OpName %2 "main"
OpName %3 "in_upper_bound"
OpName %4 "x"
OpDecorate %3 Flat
OpDecorate %3 Location 0
%5 = OpTypeVoid
%6 = OpTypeFunction %5
%7 = OpTypeInt 32 1
%8 = OpTypePointer Function %7
%9 = OpConstant %7 0
%10 = OpTypePointer Input %7
%3 = OpVariable %10 Input
%11 = OpTypeBool
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpConstant %12 1
%18 = OpTypePointer Function %12
%19 = OpConstant %7 1
%2 = OpFunction %5 None %6
%20 = OpLabel
%4 = OpVariable %16 Function
OpBranch %21
%21 = OpLabel
%22 = OpPhi %7 %9 %20 %23 %24
OpLoopMerge %25 %24 None
OpBranch %26
%26 = OpLabel
%27 = OpLoad %7 %3
%28 = OpSLessThan %11 %22 %27
OpBranchConditional %28 %29 %25
%29 = OpLabel
%30 = OpAccessChain %18 %4 %22
OpStore %30 %17
OpBranch %24
%24 = OpLabel
%23 = OpIAdd %7 %22 %19
OpBranch %21
%25 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<opt::LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[10];
    int i = 0;
    for (int i = 0; i < 10; ++i) {
        out_array[i] = i;
    }
    out_array[9] = i*10;
}
*/
TEST_F(PassClassTest, InductionUsedOutsideOfLoop) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
  const std::string text = R"(OpCapability Shader
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
%9 = OpConstant %6 10
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 10
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 1
%18 = OpConstant %6 9
%2 = OpFunction %4 None %5
%19 = OpLabel
%3 = OpVariable %15 Function
OpBranch %20
%20 = OpLabel
%21 = OpPhi %6 %8 %19 %22 %23
OpLoopMerge %24 %23 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %10 %21 %9
OpBranchConditional %26 %27 %24
%27 = OpLabel
%28 = OpConvertSToF %11 %21
%29 = OpAccessChain %16 %3 %21
OpStore %29 %28
OpBranch %23
%23 = OpLabel
%22 = OpIAdd %6 %21 %17
OpBranch %20
%24 = OpLabel
%30 = OpIMul %6 %21 %9
%31 = OpConvertSToF %11 %30
%32 = OpAccessChain %16 %3 %18
OpStore %32 %31
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);

  // Make sure the pass doesn't run
  SinglePassRunAndCheck<opt::LoopUnroller>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<1>>(text, text, false);
  SinglePassRunAndCheck<PartialUnrollerTestPass<2>>(text, text, false);
}

/*
Generated from the following GLSL
#version 410 core
void main() {
    float out_array[10];
    for (uint i = 0; i < 2; i++) {
      for (float x = 0; x < 5; ++x) {
        out_array[x + i*5] = i;
      }
    }
}
*/
TEST_F(PassClassTest, UnrollNestedLoopsInvalid) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
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
%14 = OpConstant %11 5
%15 = OpTypeFloat 32
%16 = OpConstant %6 10
%17 = OpTypeArray %15 %16
%18 = OpTypePointer Function %17
%19 = OpConstant %6 5
%20 = OpTypePointer Function %15
%21 = OpConstant %11 1
%22 = OpUndef %11
%2 = OpFunction %4 None %5
%23 = OpLabel
%3 = OpVariable %18 Function
OpBranch %24
%24 = OpLabel
%25 = OpPhi %6 %8 %23 %26 %27
%28 = OpPhi %11 %22 %23 %29 %27
OpLoopMerge %30 %27 None
OpBranch %31
%31 = OpLabel
%32 = OpULessThan %10 %25 %9
OpBranchConditional %32 %33 %30
%33 = OpLabel
OpBranch %34
%34 = OpLabel
%29 = OpPhi %11 %13 %33 %35 %36
OpLoopMerge %37 %36 None
OpBranch %38
%38 = OpLabel
%39 = OpSLessThan %10 %29 %14
OpBranchConditional %39 %40 %37
%40 = OpLabel
%41 = OpBitcast %6 %29
%42 = OpIMul %6 %25 %19
%43 = OpIAdd %6 %41 %42
%44 = OpConvertUToF %15 %25
%45 = OpAccessChain %20 %3 %43
OpStore %45 %44
OpBranch %36
%36 = OpLabel
%35 = OpIAdd %11 %29 %21
OpBranch %34
%37 = OpLabel
OpBranch %27
%27 = OpLabel
%26 = OpIAdd %6 %25 %21
OpBranch %24
%30 = OpLabel
OpReturn
OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, text, false);
}


/*
Generated from the following GLSL
#version 440 core
void main(){
  float x[10];
  int ind = 0;
  for (int i = 0; i < 10; i++) {
    ind = i;
    x[i] = i;
  }
}
*/
TEST_F(PassClassTest, MultiplePhiInHeader) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 440
OpName %2 "main"
OpName %3 "x"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 10
%10 = OpTypeBool
%11 = OpTypeFloat 32
%12 = OpTypeInt 32 0
%13 = OpConstant %12 10
%14 = OpTypeArray %11 %13
%15 = OpTypePointer Function %14
%16 = OpTypePointer Function %11
%17 = OpConstant %6 1
%2 = OpFunction %4 None %5
%18 = OpLabel
%3 = OpVariable %15 Function
OpBranch %19
%19 = OpLabel
%20 = OpPhi %6 %8 %18 %21 %22
%21 = OpPhi %6 %8 %18 %23 %22
OpLoopMerge %24 %22 None
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %10 %21 %9
OpBranchConditional %26 %27 %24
%27 = OpLabel
%28 = OpConvertSToF %11 %21
%29 = OpAccessChain %16 %3 %21
OpStore %29 %28
OpBranch %22
%22 = OpLabel
%23 = OpIAdd %6 %21 %17
OpBranch %19
%24 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main(){
  float x[10];
  for (int i = 0; i < 10; i++) {
    if (i == 5) {
      break;
    }
    x[i] = i;
  }
}
*/
TEST_F(PassClassTest, BreakInBody) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 440
OpName %2 "main"
OpName %3 "x"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 10
%10 = OpTypeBool
%11 = OpConstant %6 5
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpTypePointer Function %12
%18 = OpConstant %6 1
%2 = OpFunction %4 None %5
%19 = OpLabel
%3 = OpVariable %16 Function
OpBranch %20
%20 = OpLabel
%21 = OpPhi %6 %8 %19 %22 %23
OpLoopMerge %24 %23 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %10 %21 %9
OpBranchConditional %26 %27 %24
%27 = OpLabel
%28 = OpIEqual %10 %21 %11
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
OpBranch %24
%29 = OpLabel
%31 = OpConvertSToF %12 %21
%32 = OpAccessChain %17 %3 %21
OpStore %32 %31
OpBranch %23
%23 = OpLabel
%22 = OpIAdd %6 %21 %18
OpBranch %20
%24 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main(){
  float x[10];
  for (int i = 0; i < 10; i++) {
    if (i == 5) {
      continue;
    }
    x[i] = i;
  }
}
*/
TEST_F(PassClassTest, ContinueInBody) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 440
OpName %2 "main"
OpName %3 "x"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 10
%10 = OpTypeBool
%11 = OpConstant %6 5
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpTypePointer Function %12
%18 = OpConstant %6 1
%2 = OpFunction %4 None %5
%19 = OpLabel
%3 = OpVariable %16 Function
OpBranch %20
%20 = OpLabel
%21 = OpPhi %6 %8 %19 %22 %23
OpLoopMerge %24 %23 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %10 %21 %9
OpBranchConditional %26 %27 %24
%27 = OpLabel
%28 = OpIEqual %10 %21 %11
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
OpBranch %23
%29 = OpLabel
%31 = OpConvertSToF %12 %21
%32 = OpAccessChain %17 %3 %21
OpStore %32 %31
OpBranch %23
%23 = OpLabel
%22 = OpIAdd %6 %21 %18
OpBranch %20
%24 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;

  opt::LoopUnroller loop_unroller;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, text, false);
}

/*
Generated from the following GLSL
#version 440 core
void main(){
  float x[10];
  for (int i = 0; i < 10; i++) {
    if (i == 5) {
      return;
    }
    x[i] = i;
  }
}
*/
TEST_F(PassClassTest, ReturnInBody) {
  // clang-format off
  // With opt::LocalMultiStoreElimPass
const std::string text = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
OpSource GLSL 440
OpName %2 "main"
OpName %3 "x"
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 1
%7 = OpTypePointer Function %6
%8 = OpConstant %6 0
%9 = OpConstant %6 10
%10 = OpTypeBool
%11 = OpConstant %6 5
%12 = OpTypeFloat 32
%13 = OpTypeInt 32 0
%14 = OpConstant %13 10
%15 = OpTypeArray %12 %14
%16 = OpTypePointer Function %15
%17 = OpTypePointer Function %12
%18 = OpConstant %6 1
%2 = OpFunction %4 None %5
%19 = OpLabel
%3 = OpVariable %16 Function
OpBranch %20
%20 = OpLabel
%21 = OpPhi %6 %8 %19 %22 %23
OpLoopMerge %24 %23 Unroll
OpBranch %25
%25 = OpLabel
%26 = OpSLessThan %10 %21 %9
OpBranchConditional %26 %27 %24
%27 = OpLabel
%28 = OpIEqual %10 %21 %11
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
OpReturn
%29 = OpLabel
%31 = OpConvertSToF %12 %21
%32 = OpAccessChain %17 %3 %21
OpStore %32 %31
OpBranch %23
%23 = OpLabel
%22 = OpIAdd %6 %21 %18
OpBranch %20
%24 = OpLabel
OpReturn
OpFunctionEnd
)";
  // clang-format on
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::LoopUnroller>(text, text, false);
}

}  // namespace
