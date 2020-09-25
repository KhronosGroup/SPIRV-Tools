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

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationSplitLoopTest, BasicScenario) {
  // In this test scenario we have a simple loop.
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

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation = TransformationSplitLoop(11, 120, 121, 24, 101, 102, 103,
                                                104, 105, 106, 107, {{}},
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
  // In this test scenario we process the first loop from the test shader.
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

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation = TransformationSplitLoop(12, 120, 121, 9, 201, 202, 203,
                                                204, 205, 206, 207, {{}},
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
  // In this test scenario we process the second loop from the test shader.
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

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation = TransformationSplitLoop(
      42, 120, 121, 9, 201, 202, 203, 204, 205, 206, 207, {{}},
      {{42, 301}, {43, 302}, {44, 303}, {45, 304}},
      {{47, 401}, {48, 402}, {49, 403}, {50, 404}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));

  transformation.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));
}

TEST(TransformationSplitLoopTest, HeaderIsContinueTargetTest) {
  // This test handles a case where the header of the loop is also the continue
  // target of the loop.
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
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
         %18 = OpTypeBool
         %58 = OpConstantTrue %18
         %59 = OpConstantFalse %18
         %52 = OpTypePointer Function %18
         %24 = OpConstant %6 5
         %30 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %13 %11 None
               OpBranchConditional %59 %11 %13
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation =
      TransformationSplitLoop(11, 120, 121, 55, 101, 102, 103, 104, 105, 106,
                              107, {}, {{11, 201}, {13, 202}}, {{}});

  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));
}

TEST(TransformationSplitLoopTest, BranchConditionalContinueTargetTest) {
  // This test handles a case where there is a OpBranchConditional instruction
  // in the continue target.
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
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
         %18 = OpTypeBool
         %58 = OpConstantTrue %18
         %59 = OpConstantFalse %18
         %52 = OpTypePointer Function %18
         %24 = OpConstant %6 5
         %30 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %13 %12 None
               OpBranch %12
         %12 = OpLabel
               OpBranchConditional %58 %13 %11
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation = TransformationSplitLoop(
      11, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {108},
      {{11, 201}, {12, 202}, {13, 203}}, {{}});

  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));
}

TEST(TransformationSplitLoopTest, NotApplicableScenarios) {
  // This test handles some cases where the transformation is not applicable.
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
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
         %18 = OpTypeBool
         %58 = OpConstantTrue %18
         %59 = OpConstantFalse %18
         %52 = OpTypePointer Function %18
         %24 = OpConstant %6 5
         %30 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %13 %14 None
               OpBranchConditional %58 %13 %14
         %14 = OpLabel
         %70 = OpCopyObject %18 %58
               OpBranch %11
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  // Bad: |load_counter_fresh_id| is not fresh.
  auto transformation_bad_1 = TransformationSplitLoop(
      11, 120, 121, 55, 55, 102, 103, 104, 105, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: |increment_counter_fresh_id| is not fresh.
  auto transformation_bad_2 = TransformationSplitLoop(
      11, 120, 121, 55, 101, 55, 103, 104, 105, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: |condition_counter_fresh_id| is not fresh.
  auto transformation_bad_3 = TransformationSplitLoop(
      11, 120, 121, 55, 101, 102, 55, 104, 105, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));

  // Bad: |new_body_entry_block_fresh_id| is not fresh.
  auto transformation_bad_4 = TransformationSplitLoop(
      11, 120, 121, 55, 101, 102, 103, 55, 105, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(
      transformation_bad_4.IsApplicable(context.get(), transformation_context));

  // Bad: |conditional_block_fresh_id| is not fresh.
  auto transformation_bad_5 = TransformationSplitLoop(
      11, 120, 121, 55, 101, 102, 103, 104, 55, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(
      transformation_bad_5.IsApplicable(context.get(), transformation_context));

  // Bad: |load_run_second_id| is not fresh.
  auto transformation_bad_6 = TransformationSplitLoop(
      11, 120, 121, 55, 101, 102, 103, 104, 105, 55, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(
      transformation_bad_6.IsApplicable(context.get(), transformation_context));

  // Bad: |selection_merge_fresh_block_fresh_id| is not fresh.
  auto transformation_bad_7 = TransformationSplitLoop(
      11, 120, 121, 55, 101, 102, 103, 104, 105, 106, 55, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(
      transformation_bad_7.IsApplicable(context.get(), transformation_context));

  // Bad: |constant_limit_id| does not refer to an existing instruction.
  auto transformation_bad_8 = TransformationSplitLoop(
      11, 120, 121, 90, 101, 102, 103, 104, 105, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(
      transformation_bad_8.IsApplicable(context.get(), transformation_context));

  // Bad: |constant_limit_id| does not refer to an integer value.
  auto transformation_bad_9 = TransformationSplitLoop(
      11, 120, 121, 58, 101, 102, 103, 104, 105, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(
      transformation_bad_9.IsApplicable(context.get(), transformation_context));
  // Bad: |loop_header_id| does not refer to a loop header.
  auto transformation_bad_10 = TransformationSplitLoop(
      14, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(transformation_bad_10.IsApplicable(context.get(),
                                                  transformation_context));
  // Bad: There is no entry for block with id 13 in
  // |original_label_to_duplicate_label|.
  auto transformation_bad_11 =
      TransformationSplitLoop(11, 120, 121, 55, 101, 102, 103, 104, 105, 106,
                              107, {108}, {{11, 201}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(transformation_bad_11.IsApplicable(context.get(),
                                                  transformation_context));

  // Bad: Value of id 13 in |original_label_to_duplicate_label| is not a fresh
  // id.
  auto transformation_bad_12 = TransformationSplitLoop(
      11, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {108},
      {{11, 201}, {13, 55}, {14, 203}}, {{70, 301}});
  ASSERT_FALSE(transformation_bad_12.IsApplicable(context.get(),
                                                  transformation_context));

  // Bad: There is no entry for instruction with id 70 in
  // |original_id_to_duplicate_id|.
  auto transformation_bad_13 = TransformationSplitLoop(
      11, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{}});
  ASSERT_FALSE(transformation_bad_13.IsApplicable(context.get(),
                                                  transformation_context));

  // Bad: Value of id 70 in |original_id_to_duplicate_id| is not a fresh id.
  auto transformation_bad_14 = TransformationSplitLoop(
      11, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {108},
      {{11, 201}, {13, 202}, {14, 203}}, {{70, 55}});
  ASSERT_FALSE(transformation_bad_14.IsApplicable(context.get(),
                                                  transformation_context));
}

TEST(TransformationSplitLoopTest, ResolvingOpPhiMergeBlock) {
  // This test handles a case where there is a OpPhi instruction referring to
  // the merge block. Its id must be replaced by the id of the
  // |selection_merge_block|.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "a("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %13 = OpTypeBool
         %52 = OpTypePointer Function %13
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
         %14 = OpConstantTrue %13
         %59 = OpConstantFalse %13
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpBranch %8
          %8 = OpLabel
               OpLoopMerge %10 %11 None
               OpBranchConditional %14 %11 %10
         %11 = OpLabel
               OpBranch %8
         %10 = OpLabel
         %70 = OpCopyObject %13 %14
               OpBranch %20
         %20 = OpLabel
         %71 = OpPhi %13 %70 %10
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation_good_1 = TransformationSplitLoop(
      8, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {{}},
      {{8, 201}, {11, 202}, {10, 203}}, {{70, 301}});

  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "a("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %13 = OpTypeBool
         %52 = OpTypePointer Function %13
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
         %14 = OpConstantTrue %13
         %59 = OpConstantFalse %13
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
        %120 = OpVariable %51 Function %56
        %121 = OpVariable %52 Function %14
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpBranch %8
          %8 = OpLabel
        %101 = OpLoad %50 %120
        %102 = OpIAdd %50 %101 %57
               OpStore %120 %102
        %103 = OpULessThan %13 %102 %55
               OpLoopMerge %10 %11 None
               OpBranchConditional %103 %104 %10
        %104 = OpLabel
               OpBranchConditional %14 %11 %10
         %11 = OpLabel
               OpBranch %8
         %10 = OpLabel
         %70 = OpCopyObject %13 %14
               OpBranch %105
        %105 = OpLabel
        %106 = OpLoad %13 %121
               OpSelectionMerge %107 None
               OpBranchConditional %106 %201 %107
        %201 = OpLabel
               OpLoopMerge %203 %202 None
               OpBranchConditional %14 %202 %203
        %202 = OpLabel
               OpBranch %201
        %203 = OpLabel
        %301 = OpCopyObject %13 %14
               OpBranch %107
        %107 = OpLabel
               OpBranch %20
         %20 = OpLabel
         %71 = OpPhi %13 %70 %107
               OpReturn
               OpFunctionEnd
        )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationSplitLoopTest, ResolvingOpPhiHeaderBlock) {
  // This test handles a case where there is a OpPhi instruction referring to
  // the predecessor of the header block. Its id must be replaced by the id of
  // the |conditional_block|.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "a("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %13 = OpTypeBool
         %52 = OpTypePointer Function %13
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
         %14 = OpConstantTrue %13
         %59 = OpConstantFalse %13
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpBranch %70
         %70 = OpLabel
         %71 = OpCopyObject %13 %14
               OpBranch %8
          %8 = OpLabel
         %72 = OpPhi %13 %71 %70 %71 %11
               OpLoopMerge %10 %11 None
               OpBranchConditional %14 %11 %10
         %11 = OpLabel
               OpBranch %8
         %10 = OpLabel
               OpBranch %20
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation_good_1 = TransformationSplitLoop(
      8, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {{}},
      {{8, 201}, {11, 202}, {10, 203}}, {{72, 301}});

  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "a("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %13 = OpTypeBool
         %52 = OpTypePointer Function %13
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
         %14 = OpConstantTrue %13
         %59 = OpConstantFalse %13
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
        %120 = OpVariable %51 Function %56
        %121 = OpVariable %52 Function %14
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpBranch %70
         %70 = OpLabel
         %71 = OpCopyObject %13 %14
               OpBranch %8
          %8 = OpLabel
         %72 = OpPhi %13 %71 %70 %71 %11
        %101 = OpLoad %50 %120
        %102 = OpIAdd %50 %101 %57
               OpStore %120 %102
        %103 = OpULessThan %13 %102 %55
               OpLoopMerge %10 %11 None
               OpBranchConditional %103 %104 %10
        %104 = OpLabel
               OpBranchConditional %14 %11 %10
         %11 = OpLabel
               OpBranch %8
         %10 = OpLabel
               OpBranch %105
        %105 = OpLabel
        %106 = OpLoad %13 %121
               OpSelectionMerge %107 None
               OpBranchConditional %106 %201 %107
        %201 = OpLabel
        %301 = OpPhi %13 %71 %105 %71 %202
               OpLoopMerge %203 %202 None
               OpBranchConditional %14 %202 %203
        %202 = OpLabel
               OpBranch %201
        %203 = OpLabel
               OpBranch %107
        %107 = OpLabel
               OpBranch %20
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
        )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationSplitLoopTest, ResolvingOpPhiBodyBlock) {
  // This test handles a case where there is an OpPhi instruction referring to
  // the header in the body. In the first loop this id must be replaced by the
  // id of |new_body_entry_block|.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "a("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %13 = OpTypeBool
         %52 = OpTypePointer Function %13
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
         %14 = OpConstantTrue %13
         %59 = OpConstantFalse %13
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpBranch %8
          %8 = OpLabel
         %70 = OpCopyObject %13 %14
               OpLoopMerge %10 %11 None
               OpBranchConditional %14 %30 %10
         %30 = OpLabel
         %71 = OpPhi %13 %70 %8
               OpBranch %11
         %11 = OpLabel
               OpBranch %8
         %10 = OpLabel
               OpBranch %20
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation_good_1 = TransformationSplitLoop(
      8, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {{}},
      {{8, 201}, {11, 202}, {10, 203}, {30, 204}}, {{70, 301}, {71, 302}});

  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));
  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "a("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %13 = OpTypeBool
         %52 = OpTypePointer Function %13
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
         %14 = OpConstantTrue %13
         %59 = OpConstantFalse %13
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
        %120 = OpVariable %51 Function %56
        %121 = OpVariable %52 Function %14
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpBranch %8
          %8 = OpLabel
        %101 = OpLoad %50 %120
        %102 = OpIAdd %50 %101 %57
               OpStore %120 %102
        %103 = OpULessThan %13 %102 %55
               OpLoopMerge %10 %11 None
               OpBranchConditional %103 %104 %10
        %104 = OpLabel
         %70 = OpCopyObject %13 %14
               OpBranchConditional %14 %30 %10
         %30 = OpLabel
         %71 = OpPhi %13 %70 %104
               OpBranch %11
         %11 = OpLabel
               OpBranch %8
         %10 = OpLabel
               OpBranch %105
        %105 = OpLabel
        %106 = OpLoad %13 %121
               OpSelectionMerge %107 None
               OpBranchConditional %106 %201 %107
        %201 = OpLabel
        %301 = OpCopyObject %13 %14
               OpLoopMerge %203 %202 None
               OpBranchConditional %14 %204 %203
        %204 = OpLabel
        %302 = OpPhi %13 %301 %201
               OpBranch %202
        %202 = OpLabel
               OpBranch %201
        %203 = OpLabel
               OpBranch %107
        %107 = OpLabel
               OpBranch %20
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
        )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationSplitLoopTest, OpPhiInMergeBlockNotApplicable) {
  // This test handles a case where there is an OpPhi instruction in the
  // the merge block. We currently exclude this case and the transformation is
  // not applicable.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "a("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %13 = OpTypeBool
         %52 = OpTypePointer Function %13
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
         %14 = OpConstantTrue %13
         %59 = OpConstantFalse %13
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpBranch %8
          %8 = OpLabel
         %70 = OpCopyObject %13 %14
               OpLoopMerge %10 %11 None
               OpBranchConditional %14 %11 %10
         %11 = OpLabel
               OpBranch %8
         %10 = OpLabel
         %71 = OpPhi %13 %70 %8
               OpBranch %20
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation_bad = TransformationSplitLoop(
      8, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {{}},
      {{8, 201}, {11, 202}, {10, 203}}, {{70, 301}, {71, 302}});

  ASSERT_FALSE(
      transformation_bad.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationSplitLoopTest, LogicalNotFreshIdsNotApplicable) {
  // This test handles cases where the ids provided as |logical_not_fresh_ids|
  // are not fresh or their number is insufficient.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "a("
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %13 = OpTypeBool
         %52 = OpTypePointer Function %13
         %50 = OpTypeInt 32 0
         %51 = OpTypePointer Function %50
         %14 = OpConstantTrue %13
         %59 = OpConstantFalse %13
         %55 = OpConstant %50 4
         %56 = OpConstant %50 0
         %57 = OpConstant %50 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %53 = OpVariable %51 Function
         %54 = OpVariable %52 Function
               OpBranch %70
         %70 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpLoopMerge %10 %11 None
               OpBranchConditional %14 %10 %11
         %11 = OpLabel
               OpBranch %8
         %10 = OpLabel
               OpBranch %20
         %20 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  // Bad: Insufficient number of fresh ids as |logical_not_fresh_ids|.
  auto transformation_bad_1 =
      TransformationSplitLoop(8, 120, 121, 55, 101, 102, 103, 104, 105, 106,
                              107, {}, {{8, 201}, {11, 202}, {10, 203}}, {{}});

  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: The id in |logical_not_fresh_ids| is not fresh.
  auto transformation_bad_2 = TransformationSplitLoop(
      8, 120, 121, 55, 101, 102, 103, 104, 105, 106, 107, {55},
      {{8, 201}, {11, 202}, {10, 203}}, {{}});
  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationSplitLoopTest, NestedLoopSplitOuter) {
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
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
        %900 = OpTypeInt 32 0
        %901 = OpConstant %900 0
        %902 = OpConstant %900 1
        %903 = OpConstant %900 3
        %904 = OpTypePointer Function %900
         %18 = OpTypeBool
        %905 = OpConstantTrue %18
        %906 = OpConstantFalse %18
        %100 = OpTypePointer Function %18
        %101 = OpConstant %6 5
         %29 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %20 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %16 = OpLoad %6 %10
         %19 = OpSLessThan %18 %16 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
               OpStore %20 %9
               OpBranch %21
         %21 = OpLabel
               OpLoopMerge %23 %24 None
               OpBranch %25
         %25 = OpLabel
         %26 = OpLoad %6 %20
         %27 = OpSLessThan %18 %26 %17
               OpBranchConditional %27 %22 %23
         %22 = OpLabel
         %28 = OpLoad %6 %8
         %30 = OpIAdd %6 %28 %29
               OpStore %8 %30
               OpBranch %24
         %24 = OpLabel
         %31 = OpLoad %6 %20
         %32 = OpIAdd %6 %31 %29
               OpStore %20 %32
               OpBranch %21
         %23 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %33 = OpLoad %6 %10
         %34 = OpIAdd %6 %33 %29
               OpStore %10 %34
               OpBranch %11
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
      )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation =
      TransformationSplitLoop(/*loop_header_id*/ 11,
                              /*variable_counter_id*/ 1000,
                              /*variable_run_second_id*/ 1001,
                              /*constant_limit_id*/ 903,
                              /*load_counter_fresh_id*/ 1002,
                              /*increment_counter_fresh_id*/ 1003,
                              /*condition_counter_fresh_id*/ 1004,
                              /*new_body_entry_block_fresh_id*/ 1005,
                              /*conditional_block_fresh_id*/ 1006,
                              /*load_run_second_fresh_id*/ 1007,
                              /*selection_merge_block_fresh_id*/ 1008,
                              /*logical_not_fresh_ids*/ {},
                              /*original_label_to_duplicate_label*/
                              {{11, 2000},
                               {15, 2001},
                               {12, 2002},
                               {21, 2003},
                               {25, 2004},
                               {22, 2005},
                               {24, 2006},
                               {23, 2007},
                               {14, 2008},
                               {13, 2009}},
                              /*original_id_to_duplicate_id*/
                              {{16, 3000},
                               {19, 3001},
                               {26, 3002},
                               {27, 3003},
                               {28, 3004},
                               {30, 3005},
                               {31, 3006},
                               {32, 3007},
                               {33, 3008},
                               {34, 3009}});

  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));
}
TEST(TransformationSplitLoopTest, ConstantsNotPresentNotApplicable) {
  // In this test scenario we don't provide the required constant
  // OpConstantFalse. Therefore, the transformation is not applicable.
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

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  auto transformation = TransformationSplitLoop(11, 120, 121, 24, 101, 102, 103,
                                                104, 105, 106, 107, {{}},
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
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
