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

#include "fuzz_test_util.h"

#include "source/fuzz/transformation_add_dead_break.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddDeadBreakTest, BreaksOutOfSimpleIf) {
  // For a simple if-then-else, checks that some dead break scenarios are
  // possible, and sanity-checks that some illegal scenarios are indeed not
  // allowed.

  // The SPIR-V for this test is adapted from the following GLSL, by separating
  // some assignments into their own basic blocks, and adding constants for true
  // and false:
  //
  // void main() {
  //   int x;
  //   int y;
  //   x = 1;
  //   if (x < y) {
  //     x = 2;
  //     x = 3;
  //   } else {
  //     y = 2;
  //     y = 3;
  //   }
  //   x = y;
  // }

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
               OpName %11 "y"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %13 = OpTypeBool
         %17 = OpConstant %6 2
         %18 = OpConstant %6 3
         %25 = OpConstantTrue %13
         %26 = OpConstantFalse %13
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %11 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %12 = OpLoad %6 %11
         %14 = OpSLessThan %13 %10 %12
               OpSelectionMerge %16 None
               OpBranchConditional %14 %15 %19
         %15 = OpLabel
               OpStore %8 %17
               OpBranch %21
         %21 = OpLabel
               OpStore %8 %18
               OpBranch %22
         %22 = OpLabel
               OpBranch %16
         %19 = OpLabel
               OpStore %11 %17
               OpBranch %23
         %23 = OpLabel
               OpStore %11 %18
               OpBranch %24
         %24 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %20 = OpLoad %6 %11
               OpStore %8 %20
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  const uint32_t merge_block = 16;
  const uint32_t true_constant = 25;
  const uint32_t false_constant = 26;

  // These are all possibilities.
  ASSERT_TRUE(TransformationAddDeadBreak(15, merge_block, true_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(15, merge_block, false_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(21, merge_block, true_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(21, merge_block, false_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(22, merge_block, true_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(22, merge_block, false_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(19, merge_block, true_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(19, merge_block, false_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(23, merge_block, true_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(23, merge_block, false_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(24, merge_block, true_constant)
                  .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(24, merge_block, false_constant)
                  .IsApplicable(context.get()));

  // Inapplicable: 100 is not a block id.
  ASSERT_FALSE(TransformationAddDeadBreak(100, merge_block, true_constant)
                   .IsApplicable(context.get()));
  ASSERT_TRUE(TransformationAddDeadBreak(15, 100, true_constant)
                  .IsApplicable(context.get()));

  // Inapplicable: 2 is not the id of a boolean constant.
  ASSERT_FALSE(TransformationAddDeadBreak(15, merge_block, 2)
                   .IsApplicable(context.get()));

  // Inapplicable: 24 is not a merge block.
  ASSERT_FALSE(TransformationAddDeadBreak(15, 24, true_constant)
                   .IsApplicable(context.get()));

  // These are the transformations we will apply.
  auto transformation1 =
      TransformationAddDeadBreak(15, merge_block, true_constant);
  auto transformation2 =
      TransformationAddDeadBreak(21, merge_block, false_constant);
  auto transformation3 =
      TransformationAddDeadBreak(22, merge_block, true_constant);
  auto transformation4 =
      TransformationAddDeadBreak(19, merge_block, false_constant);
  auto transformation5 =
      TransformationAddDeadBreak(23, merge_block, true_constant);
  auto transformation6 =
      TransformationAddDeadBreak(24, merge_block, false_constant);

  ASSERT_TRUE(transformation1.IsApplicable(context.get()));
  transformation1.Apply(context.get());
  CheckValid(env, context.get());

  ASSERT_TRUE(transformation2.IsApplicable(context.get()));
  transformation2.Apply(context.get());
  CheckValid(env, context.get());

  ASSERT_TRUE(transformation3.IsApplicable(context.get()));
  transformation3.Apply(context.get());
  CheckValid(env, context.get());

  ASSERT_TRUE(transformation4.IsApplicable(context.get()));
  transformation4.Apply(context.get());
  CheckValid(env, context.get());

  ASSERT_TRUE(transformation5.IsApplicable(context.get()));
  transformation5.Apply(context.get());
  CheckValid(env, context.get());

  ASSERT_TRUE(transformation6.IsApplicable(context.get()));
  transformation6.Apply(context.get());
  CheckValid(env, context.get());

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "x"
               OpName %11 "y"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %13 = OpTypeBool
         %17 = OpConstant %6 2
         %18 = OpConstant %6 3
         %25 = OpConstantTrue %13
         %26 = OpConstantFalse %13
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %11 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %12 = OpLoad %6 %11
         %14 = OpSLessThan %13 %10 %12
               OpSelectionMerge %16 None
               OpBranchConditional %14 %15 %19
         %15 = OpLabel
               OpStore %8 %17
               OpBranchConditional %25 %21 %16
         %21 = OpLabel
               OpStore %8 %18
               OpBranchConditional %26 %16 %22
         %22 = OpLabel
               OpBranchConditional %25 %16 %16
         %19 = OpLabel
               OpStore %11 %17
               OpBranchConditional %26 %16 %23
         %23 = OpLabel
               OpStore %11 %18
               OpBranchConditional %25 %24 %16
         %24 = OpLabel
               OpBranchConditional %26 %16 %16
         %16 = OpLabel
         %20 = OpLoad %6 %11
               OpStore %8 %20
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_transformation, context.get());
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
