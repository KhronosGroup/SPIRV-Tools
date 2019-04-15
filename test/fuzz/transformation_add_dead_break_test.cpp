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

TEST(TransformationAddDeadBreakTest, BreakOutOfNestedIfs) {
  // Checks some allowed and disallowed scenarios for nests of ifs.

  // The SPIR-V for this test is adapted from the following GLSL:
  //
  // void main() {
  //   int x;
  //   int y;
  //   x = 1;
  //   if (x < y) {
  //     x = 2;
  //     x = 3;
  //     if (x == y) {
  //       y = 3;
  //     }
  //   } else {
  //     y = 2;
  //     y = 3;
  //   }
  //   if (x == y) {
  //     x = 2;
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
         %31 = OpConstantTrue %13
         %32 = OpConstantFalse %13
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %11 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %12 = OpLoad %6 %11
         %14 = OpSLessThan %13 %10 %12
               OpSelectionMerge %16 None
               OpBranchConditional %14 %15 %24
         %15 = OpLabel
               OpStore %8 %17
               OpBranch %33
         %33 = OpLabel
               OpStore %8 %18
         %19 = OpLoad %6 %8
               OpBranch %34
         %34 = OpLabel
         %20 = OpLoad %6 %11
         %21 = OpIEqual %13 %19 %20
               OpSelectionMerge %23 None
               OpBranchConditional %21 %22 %23
         %22 = OpLabel
               OpStore %11 %18
               OpBranch %35
         %35 = OpLabel
               OpBranch %23
         %23 = OpLabel
               OpBranch %16
         %24 = OpLabel
               OpStore %11 %17
               OpBranch %36
         %36 = OpLabel
               OpStore %11 %18
               OpBranch %16
         %16 = OpLabel
         %25 = OpLoad %6 %8
               OpBranch %37
         %37 = OpLabel
         %26 = OpLoad %6 %11
         %27 = OpIEqual %13 %25 %26
               OpSelectionMerge %29 None
               OpBranchConditional %27 %28 %29
         %28 = OpLabel
               OpStore %8 %17
               OpBranch %38
         %38 = OpLabel
               OpBranch %29
         %29 = OpLabel
         %30 = OpLoad %6 %11
               OpStore %8 %30
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  // The booleans
  const uint32_t true_constant = 31;
  const uint32_t false_constant = 32;

  // The merge blocks
  const uint32_t merge_inner = 23;
  const uint32_t merge_outer = 16;
  const uint32_t merge_after = 29;

  // The headers
  const uint32_t header_inner = 34;
  const uint32_t header_outer = 5;
  const uint32_t header_after = 37;

  // The non-merge-nor-header blocks in each construct
  const uint32_t inner_block_1 = 22;
  const uint32_t inner_block_2 = 35;
  const uint32_t outer_block_1 = 15;
  const uint32_t outer_block_2 = 33;
  const uint32_t outer_block_3 = 24;
  const uint32_t outer_block_4 = 36;
  const uint32_t after_block_1 = 28;
  const uint32_t after_block_2 = 38;

  // Fine to break from a construct to its merge
  ASSERT_TRUE(
      TransformationAddDeadBreak(inner_block_1, merge_inner, true_constant)
          .IsApplicable(context.get()));
  ASSERT_TRUE(
      TransformationAddDeadBreak(inner_block_2, merge_inner, false_constant)
          .IsApplicable(context.get()));
  ASSERT_TRUE(
      TransformationAddDeadBreak(outer_block_1, merge_outer, true_constant)
          .IsApplicable(context.get()));
  ASSERT_TRUE(
      TransformationAddDeadBreak(outer_block_2, merge_outer, false_constant)
          .IsApplicable(context.get()));
  ASSERT_TRUE(
      TransformationAddDeadBreak(outer_block_3, merge_outer, true_constant)
          .IsApplicable(context.get()));
  ASSERT_TRUE(
      TransformationAddDeadBreak(outer_block_4, merge_outer, false_constant)
          .IsApplicable(context.get()));
  ASSERT_TRUE(
      TransformationAddDeadBreak(after_block_1, merge_after, true_constant)
          .IsApplicable(context.get()));
  ASSERT_TRUE(
      TransformationAddDeadBreak(after_block_2, merge_after, false_constant)
          .IsApplicable(context.get()));

  // Not OK to break to the wrong merge (whether enclosing or not)
  ASSERT_FALSE(
      TransformationAddDeadBreak(inner_block_1, merge_outer, true_constant)
          .IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddDeadBreak(inner_block_2, merge_after, false_constant)
          .IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddDeadBreak(outer_block_1, merge_inner, true_constant)
          .IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddDeadBreak(outer_block_2, merge_after, false_constant)
          .IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddDeadBreak(after_block_1, merge_inner, true_constant)
          .IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddDeadBreak(after_block_2, merge_outer, false_constant)
          .IsApplicable(context.get()));

  // Not OK to break from header (as it does not branch unconditionally)
  ASSERT_FALSE(
      TransformationAddDeadBreak(header_inner, merge_inner, true_constant)
          .IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddDeadBreak(header_outer, merge_outer, false_constant)
          .IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddDeadBreak(header_after, merge_after, true_constant)
          .IsApplicable(context.get()));

  // Not OK to break to non-merge
  ASSERT_FALSE(
      TransformationAddDeadBreak(inner_block_1, inner_block_2, true_constant)
          .IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddDeadBreak(outer_block_2, after_block_1, false_constant)
          .IsApplicable(context.get()));
  ASSERT_FALSE(
      TransformationAddDeadBreak(outer_block_1, header_after, true_constant)
          .IsApplicable(context.get()));

  auto transformation1 =
      TransformationAddDeadBreak(inner_block_1, merge_inner, true_constant);
  auto transformation2 =
      TransformationAddDeadBreak(inner_block_2, merge_inner, false_constant);
  auto transformation3 =
      TransformationAddDeadBreak(outer_block_1, merge_outer, true_constant);
  auto transformation4 =
      TransformationAddDeadBreak(outer_block_2, merge_outer, false_constant);
  auto transformation5 =
      TransformationAddDeadBreak(outer_block_3, merge_outer, true_constant);
  auto transformation6 =
      TransformationAddDeadBreak(outer_block_4, merge_outer, false_constant);
  auto transformation7 =
      TransformationAddDeadBreak(after_block_1, merge_after, true_constant);
  auto transformation8 =
      TransformationAddDeadBreak(after_block_2, merge_after, false_constant);

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

  ASSERT_TRUE(transformation7.IsApplicable(context.get()));
  transformation7.Apply(context.get());
  CheckValid(env, context.get());

  ASSERT_TRUE(transformation8.IsApplicable(context.get()));
  transformation8.Apply(context.get());
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
         %31 = OpConstantTrue %13
         %32 = OpConstantFalse %13
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %11 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %12 = OpLoad %6 %11
         %14 = OpSLessThan %13 %10 %12
               OpSelectionMerge %16 None
               OpBranchConditional %14 %15 %24
         %15 = OpLabel
               OpStore %8 %17
               OpBranchConditional %31 %33 %16
         %33 = OpLabel
               OpStore %8 %18
         %19 = OpLoad %6 %8
               OpBranchConditional %32 %16 %34
         %34 = OpLabel
         %20 = OpLoad %6 %11
         %21 = OpIEqual %13 %19 %20
               OpSelectionMerge %23 None
               OpBranchConditional %21 %22 %23
         %22 = OpLabel
               OpStore %11 %18
               OpBranchConditional %31 %35 %23
         %35 = OpLabel
               OpBranchConditional %32 %23 %23
         %23 = OpLabel
               OpBranch %16
         %24 = OpLabel
               OpStore %11 %17
               OpBranchConditional %31 %36 %16
         %36 = OpLabel
               OpStore %11 %18
               OpBranch Conditional %32 %16 %16
         %16 = OpLabel
         %25 = OpLoad %6 %8
               OpBranch %37
         %37 = OpLabel
         %26 = OpLoad %6 %11
         %27 = OpIEqual %13 %25 %26
               OpSelectionMerge %29 None
               OpBranchConditional %27 %28 %29
         %28 = OpLabel
               OpStore %8 %17
               OpBranchConditional %31 %38 %29
         %38 = OpLabel
               OpBranchConditional %32 %29 %29
         %29 = OpLabel
         %30 = OpLoad %6 %11
               OpStore %8 %30
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after_transformation, context.get());
}

TEST(TransformationAddDeadBreakTest, BreakOutOfInnermostSwitch) {
  // Checks some allowed and disallowed scenarios for a nested switch.

  // The SPIR-V for this test is adapted from the following GLSL:
  //
  // TODO

  std::string shader = R"(
               TODO
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  // TODO: assert some applicable and inapplicable transformations.

  // TODO: apply some transformations, checking validity after each one as
  // follows:
  CheckValid(env, context.get());

  std::string after_transformation = R"(
               TODO
  )";

  CheckEqual(env, after_transformation, context.get());
}

TEST(TransformationAddDeadBreakTest, BreakOutOfLoopNest) {
  // Checks some allowed and disallowed scenarios for a nest of loops, including
  // breaking several loops simultaneously, and breaking from an if or switch
  // right out of a loop.

  // The SPIR-V for this test is adapted from the following GLSL:
  //
  // TODO

  std::string shader = R"(
               TODO
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  // TODO: assert some applicable and inapplicable transformations.

  // TODO: apply some transformations, checking validity after each one as
  // follows:
  CheckValid(env, context.get());

  std::string after_transformation = R"(
               TODO
  )";

  CheckEqual(env, after_transformation, context.get());
}

TEST(TransformationAddDeadBreakTest, PhiInstructions) {
  // Checks that the transformation works in the presence of phi instructions.

  // The SPIR-V for this test is adapted from the following GLSL:
  //
  // TODO

  std::string shader = R"(
               TODO
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  // TODO: assert some applicable and inapplicable transformations.

  // TODO: apply some transformations, checking validity after each one as
  // follows:
  CheckValid(env, context.get());

  std::string after_transformation = R"(
               TODO
  )";

  CheckEqual(env, after_transformation, context.get());
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
