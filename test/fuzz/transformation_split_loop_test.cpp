// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_split_loop.h"
#include "test/fuzz/fuzz_test_util.h"

// to be removed
#include "source/fuzz/fuzzer_pass_split_loops.h"
#include "source/fuzz/pseudo_random_generator.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationSplitLoopTest, BasicScenarios) {
  // This is a simple transformation and this test handles the main cases.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "s"
               OpName %10 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
         %60 = OpTypeInt 32 0
          %7 = OpTypePointer Function %6
         %61 = OpTypePointer Function %60
          %9 = OpConstant %60 0
         %17 = OpConstant %60 10
         %18 = OpTypeBool
         %55 = OpConstantTrue %18
         %56 = OpConstantFalse %18
         %53 = OpTypePointer Function %18
         %24 = OpConstant %60 3
         %30 = OpConstant %60 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %61 Function
         %10 = OpVariable %61 Function
         %50 = OpVariable %61 Function
         %54 = OpVariable %53 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %16 = OpLoad %60 %10
         %19 = OpSLessThan %18 %16 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
         %20 = OpLoad %60 %10
         %21 = OpLoad %60 %8
         %22 = OpIAdd %60 %21 %20
               OpStore %8 %22
         %23 = OpLoad %60 %10
         %25 = OpIEqual %18 %23 %24
               OpSelectionMerge %27 None
               OpBranchConditional %25 %26 %27
         %26 = OpLabel
               OpBranch %13
         %27 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %29 = OpLoad %60 %10
         %31 = OpIAdd %60 %29 %30
               OpStore %10 %31
               OpBranch %11
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto transformation =
      TransformationSplitLoop(11, 50, 54, 24, 101, 102, 103, 104, 105, 106, 107,
                              {{11, 201},
                               {15, 202},
                               {12, 203},
                               {13, 204},
                               {26, 205},
                               {27, 206},
                               {14, 207}},
                              {{16, 301},
                               {19, 302},
                               {20, 303},
                               {21, 304},
                               {22, 305},
                               {23, 306},
                               {25, 307},
                               {29, 308},
                               {31, 309}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));
}

TEST(TransformationSplitLoopTest, TestShaderFirstLoop) {
  // This is a simple transformation and this test handles the main cases.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 320
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
        %110 = OpTypeInt 32 0
        %111 = OpConstant %110 0
        %112 = OpConstant %110 1
        %113 = OpTypePointer Function %110
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 20
         %11 = OpConstant %6 0
         %18 = OpConstant %6 100
         %19 = OpTypeBool
        %103 = OpConstantTrue %19
        %104 = OpConstantFalse %19
        %101 = OpTypePointer Function %19
         %30 = OpConstant %6 300
         %39 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %41 = OpVariable %7 Function
        %100 = OpVariable %113 Function
        %102 = OpVariable %101 Function
               OpStore %8 %9
               OpStore %10 %11
               OpBranch %12
         %12 = OpLabel
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %17 = OpLoad %6 %10
         %20 = OpSLessThan %19 %17 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %21 = OpLoad %6 %10
         %22 = OpLoad %6 %8
         %23 = OpSGreaterThan %19 %21 %22
               OpSelectionMerge %25 None
               OpBranchConditional %23 %24 %25
         %24 = OpLabel
               OpBranch %14
         %25 = OpLabel
         %27 = OpLoad %6 %8
         %28 = OpLoad %6 %10
         %29 = OpIAdd %6 %27 %28
         %31 = OpSGreaterThan %19 %29 %30
               OpSelectionMerge %33 None
               OpBranchConditional %31 %32 %33
         %32 = OpLabel
               OpBranch %14
         %33 = OpLabel
         %35 = OpLoad %6 %10
         %36 = OpLoad %6 %8
         %37 = OpIAdd %6 %36 %35
               OpStore %8 %37
               OpBranch %15
         %15 = OpLabel
         %38 = OpLoad %6 %10
         %40 = OpIAdd %6 %38 %39
               OpStore %10 %40
               OpBranch %12
         %14 = OpLabel
               OpStore %41 %11
               OpBranch %42
         %42 = OpLabel
         %47 = OpLoad %6 %41
         %48 = OpSLessThan %19 %47 %18
               OpLoopMerge %44 %45 None
               OpBranchConditional %48 %43 %44
         %43 = OpLabel
               OpBranch %45
         %45 = OpLabel
         %49 = OpLoad %6 %41
         %50 = OpIAdd %6 %49 %39
               OpStore %41 %50
               OpBranch %42
         %44 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // First loop.

  auto transformation = TransformationSplitLoop(12, 100, 102, 9, 201, 202, 203,
                                                204, 205, 206, 207,
                                                {{12, 301},
                                                 {16, 302},
                                                 {13, 303},
                                                 {24, 304},
                                                 {25, 305},
                                                 {32, 306},
                                                 {33, 307},
                                                 {15, 308},
                                                 {14, 309}},
                                                {{17, 401},
                                                 {20, 402},
                                                 {21, 403},
                                                 {22, 404},
                                                 {23, 405},
                                                 {27, 406},
                                                 {28, 407},
                                                 {29, 408},
                                                 {31, 409},
                                                 {35, 410},
                                                 {36, 411},
                                                 {37, 412},
                                                 {38, 413},
                                                 {40, 414}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));
}

TEST(TransformationSplitLoopTest, TestShaderSecondLoop) {
  // This is a simple transformation and this test handles the main cases.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 320
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
        %110 = OpTypeInt 32 0
        %111 = OpConstant %110 0
        %112 = OpConstant %110 1
        %113 = OpTypePointer Function %110
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 20
         %11 = OpConstant %6 0
         %18 = OpConstant %6 100
         %19 = OpTypeBool
        %103 = OpConstantTrue %19
        %104 = OpConstantFalse %19
        %101 = OpTypePointer Function %19
         %30 = OpConstant %6 300
         %39 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %41 = OpVariable %7 Function
        %100 = OpVariable %113 Function
        %102 = OpVariable %101 Function
               OpStore %8 %9
               OpStore %10 %11
               OpBranch %12
         %12 = OpLabel
               OpLoopMerge %14 %15 None
               OpBranch %16
         %16 = OpLabel
         %17 = OpLoad %6 %10
         %20 = OpSLessThan %19 %17 %18
               OpBranchConditional %20 %13 %14
         %13 = OpLabel
         %21 = OpLoad %6 %10
         %22 = OpLoad %6 %8
         %23 = OpSGreaterThan %19 %21 %22
               OpSelectionMerge %25 None
               OpBranchConditional %23 %24 %25
         %24 = OpLabel
               OpBranch %14
         %25 = OpLabel
         %27 = OpLoad %6 %8
         %28 = OpLoad %6 %10
         %29 = OpIAdd %6 %27 %28
         %31 = OpSGreaterThan %19 %29 %30
               OpSelectionMerge %33 None
               OpBranchConditional %31 %32 %33
         %32 = OpLabel
               OpBranch %14
         %33 = OpLabel
         %35 = OpLoad %6 %10
         %36 = OpLoad %6 %8
         %37 = OpIAdd %6 %36 %35
               OpStore %8 %37
               OpBranch %15
         %15 = OpLabel
         %38 = OpLoad %6 %10
         %40 = OpIAdd %6 %38 %39
               OpStore %10 %40
               OpBranch %12
         %14 = OpLabel
               OpStore %41 %11
               OpBranch %42
         %42 = OpLabel
         %47 = OpLoad %6 %41
         %48 = OpSLessThan %19 %47 %18
               OpLoopMerge %44 %45 None
               OpBranchConditional %48 %43 %44
         %43 = OpLabel
               OpBranch %45
         %45 = OpLabel
         %49 = OpLoad %6 %41
         %50 = OpIAdd %6 %49 %39
               OpStore %41 %50
               OpBranch %42
         %44 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Second loop.

  auto transformation =
      TransformationSplitLoop(42, 100, 102, 9, 201, 202, 203, 204, 205, 206,
                              207, {{42, 301}, {43, 302}, {44, 303}, {45, 304}},
                              {{47, 401}, {48, 402}, {49, 403}, {50, 404}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));

  transformation.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));
}

TEST(TransformationSplitLoopTest, FuzzerPassBasicTest) {
  // This is a simple transformation and this test handles the main cases.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "s"
               OpName %10 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
         %60 = OpTypeInt 32 0
          %7 = OpTypePointer Function %6
         %61 = OpTypePointer Function %60
          %9 = OpConstant %60 0
         %17 = OpConstant %60 10
         %18 = OpTypeBool
         %55 = OpConstantTrue %18
         %56 = OpConstantFalse %18
         %53 = OpTypePointer Function %18
         %24 = OpConstant %60 3
         %30 = OpConstant %60 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %61 Function
         %10 = OpVariable %61 Function
         %50 = OpVariable %61 Function
         %54 = OpVariable %53 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %16 = OpLoad %60 %10
         %19 = OpSLessThan %18 %16 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
         %20 = OpLoad %60 %10
         %21 = OpLoad %60 %8
         %22 = OpIAdd %60 %21 %20
               OpStore %8 %22
         %23 = OpLoad %60 %10
         %25 = OpIEqual %18 %23 %24
               OpSelectionMerge %27 None
               OpBranchConditional %25 %26 %27
         %26 = OpLabel
               OpBranch %13
         %27 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %29 = OpLoad %60 %10
         %31 = OpIAdd %60 %29 %30
               OpStore %10 %31
               OpBranch %11
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto prng = MakeUnique<PseudoRandomGenerator>(0);

  FuzzerContext fuzzer_context(prng.get(), 100);
  protobufs::TransformationSequence transformation_sequence;

  for (int i = 0; i < 20; i++) {
    FuzzerPassSplitLoops fuzzer_pass(context.get(), &transformation_context,
                                     &fuzzer_context, &transformation_sequence);

    fuzzer_pass.Apply();
  }

  std::vector<uint32_t> actual_binary;
  context.get()->module()->ToBinary(&actual_binary, false);
  SpirvTools t(env);
  std::string actual_disassembled;
  t.Disassemble(actual_binary, &actual_disassembled, kFuzzDisassembleOption);
  std::cout << actual_disassembled << std::endl;

  ASSERT_TRUE(IsValid(env, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
