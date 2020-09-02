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

#include "source/fuzz/transformation_duplicate_region_with_selection.h"
#include "test/fuzz/fuzz_test_util.h"

/*to be removed*/
#include "source/fuzz/fuzzer_pass_duplicate_regions_with_selections.h"
#include "source/fuzz/pseudo_random_generator.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationDuplicateRegionWithSelectionTest, BasicUseTest) {
  // This test handles a case where the ids from the original region are used in
  // subsequent block.

  std::string shader = R"(
               OpCapability Shader
               OpCapability VariablePointers
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %12 "b"
               OpName %18 "c"
               OpName %20 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %14 = OpConstant %6 2
         %16 = OpTypeBool
         %17 = OpTypePointer Function %16
         %19 = OpConstantTrue %16
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpVariable %17 Function
         %20 = OpVariable %7 Function
               OpStore %18 %19
               OpStore %20 %14
         %21 = OpFunctionCall %2 %10 %20
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
               OpBranch %800
        %800 = OpLabel
         %13 = OpLoad %6 %9
         %15 = OpIAdd %6 %13 %14
               OpStore %12 %15
               OpBranch %900
         %900 = OpLabel
         %901 = OpIAdd %6 %15 %13
         %902 = OpISub %6 %13 %15
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

  TransformationDuplicateRegionWithSelection transformation_good_1 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 800, 800, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 301}, {15, 302}});

  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));
  std::string expected_shader = R"(
               OpCapability Shader
               OpCapability VariablePointers
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %12 "b"
               OpName %18 "c"
               OpName %20 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %14 = OpConstant %6 2
         %16 = OpTypeBool
         %17 = OpTypePointer Function %16
         %19 = OpConstantTrue %16
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpVariable %17 Function
         %20 = OpVariable %7 Function
               OpStore %18 %19
               OpStore %20 %14
         %21 = OpFunctionCall %2 %10 %20
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
               OpBranch %500
        %500 = OpLabel
               OpSelectionMerge %501 None
               OpBranchConditional %19 %800 %100
        %800 = OpLabel
         %13 = OpLoad %6 %9
         %15 = OpIAdd %6 %13 %14
               OpStore %12 %15
               OpBranch %501
        %100 = OpLabel
        %201 = OpLoad %6 %9
        %202 = OpIAdd %6 %201 %14
               OpStore %12 %202
               OpBranch %501
        %501 = OpLabel
        %301 = OpPhi %6 %13 %800 %201 %100
        %302 = OpPhi %6 %15 %800 %202 %100
               OpBranch %900
        %900 = OpLabel
        %901 = OpIAdd %6 %302 %301
        %902 = OpISub %6 %301 %302
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationDuplicateRegionWithSelectionTest, BasicExitBlockTest) {
  // This test handles a case where the exit block of the region is the exit
  // block of the containing function.

  std::string shader = R"(
               OpCapability Shader
               OpCapability VariablePointers
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %12 "b"
               OpName %18 "c"
               OpName %20 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %14 = OpConstant %6 2
         %16 = OpTypeBool
         %17 = OpTypePointer Function %16
         %19 = OpConstantTrue %16
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpVariable %17 Function
         %20 = OpVariable %7 Function
               OpStore %18 %19
               OpStore %20 %14
         %21 = OpFunctionCall %2 %10 %20
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
               OpBranch %800
        %800 = OpLabel
         %13 = OpLoad %6 %9
         %15 = OpIAdd %6 %13 %14
               OpStore %12 %15
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

  TransformationDuplicateRegionWithSelection transformation_good_1 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 800, 800, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 301}, {15, 302}});

  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));

  std::string expected_shader = R"(
   OpCapability Shader
               OpCapability VariablePointers
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %12 "b"
               OpName %18 "c"
               OpName %20 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %14 = OpConstant %6 2
         %16 = OpTypeBool
         %17 = OpTypePointer Function %16
         %19 = OpConstantTrue %16
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpVariable %17 Function
         %20 = OpVariable %7 Function
               OpStore %18 %19
               OpStore %20 %14
         %21 = OpFunctionCall %2 %10 %20
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
               OpBranch %500
        %500 = OpLabel
               OpSelectionMerge %501 None
               OpBranchConditional %19 %800 %100
        %800 = OpLabel
         %13 = OpLoad %6 %9
         %15 = OpIAdd %6 %13 %14
               OpStore %12 %15
               OpBranch %501
        %100 = OpLabel
        %201 = OpLoad %6 %9
        %202 = OpIAdd %6 %201 %14
               OpStore %12 %202
               OpBranch %501
        %501 = OpLabel
        %301 = OpPhi %6 %13 %800 %201 %100
        %302 = OpPhi %6 %15 %800 %202 %100
               OpReturn
               OpFunctionEnd

  )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationDuplicateRegionWithSelectionTest, NotApplicableCFGTest) {
  // This test handles few cases where the transformation is not applicable
  // because of the control flow graph or layout of the blocks.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %18 "b"
               OpName %25 "c"
               OpName %27 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %13 = OpConstant %6 2
         %14 = OpTypeBool
         %24 = OpTypePointer Function %14
         %26 = OpConstantTrue %14
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %25 = OpVariable %24 Function
         %27 = OpVariable %7 Function
               OpStore %25 %26
               OpStore %27 %13
         %28 = OpFunctionCall %2 %10 %27
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %18 = OpVariable %7 Function
         %12 = OpLoad %6 %9
         %15 = OpSLessThan %14 %12 %13
               OpSelectionMerge %17 None
               OpBranchConditional %15 %16 %21
         %16 = OpLabel
         %19 = OpLoad %6 %9
         %20 = OpIAdd %6 %19 %13
               OpStore %18 %20
               OpBranch %17
         %21 = OpLabel
         %22 = OpLoad %6 %9
         %23 = OpISub %6 %22 %13
               OpStore %18 %23
               OpBranch %17
         %17 = OpLabel
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

  // Bad: |entry_block_id| refers to the entry block of the function (this
  // transformation currently avoids such cases).
  TransformationDuplicateRegionWithSelection transformation_bad_1 =
      TransformationDuplicateRegionWithSelection(
          500, 26, 501, 11, 11, {{11, 100}}, {{18, 201}, {12, 202}, {15, 203}},
          {{18, 301}, {12, 302}, {15, 303}});
  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: The block with id 16 does not dominate the block with id 21.
  TransformationDuplicateRegionWithSelection transformation_bad_2 =
      TransformationDuplicateRegionWithSelection(
          500, 26, 501, 16, 21, {{16, 100}, {21, 101}},
          {{19, 201}, {20, 202}, {22, 203}, {23, 204}},
          {{19, 301}, {20, 302}, {22, 303}, {23, 304}});
  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: The block with id 21 does not post-dominate the block with id 11.
  TransformationDuplicateRegionWithSelection transformation_bad_3 =
      TransformationDuplicateRegionWithSelection(
          500, 26, 501, 11, 21, {{11, 100}, {21, 101}},
          {{18, 201}, {12, 202}, {15, 203}, {22, 204}, {23, 205}},
          {{18, 301}, {12, 302}, {15, 303}, {22, 304}, {23, 305}});
  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));

  // Bad: The block with id 5 is contained in a different function than the
  // block with id 11.
  TransformationDuplicateRegionWithSelection transformation_bad_4 =
      TransformationDuplicateRegionWithSelection(
          500, 26, 501, 5, 11, {{5, 100}, {11, 101}},
          {{25, 201}, {27, 202}, {28, 203}, {18, 204}, {12, 205}, {15, 206}},
          {{25, 301}, {27, 302}, {28, 303}, {18, 304}, {12, 305}, {15, 306}});
  ASSERT_FALSE(
      transformation_bad_4.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationDuplicateRegionWithSelectionTest, NotApplicableIdTest) {
  // This test handles a case where the supplied ids are either not fresh, not
  // distinct, not valid in their context or do not refer to the existing
  // instructions.

  std::string shader = R"(
               OpCapability Shader
               OpCapability VariablePointers
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %12 "b"
               OpName %18 "c"
               OpName %20 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %14 = OpConstant %6 2
         %16 = OpTypeBool
         %17 = OpTypePointer Function %16
         %19 = OpConstantTrue %16
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpVariable %17 Function
         %20 = OpVariable %7 Function
               OpStore %18 %19
               OpStore %20 %14
         %21 = OpFunctionCall %2 %10 %20
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
               OpBranch %800
        %800 = OpLabel
         %13 = OpLoad %6 %9
         %15 = OpIAdd %6 %13 %14
               OpStore %12 %15
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

  // Bad: A value in the |original_label_to_duplicate_label| is not a fresh id.
  TransformationDuplicateRegionWithSelection transformation_bad_1 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 800, 800, {{800, 21}}, {{13, 201}, {15, 202}},
          {{13, 301}, {15, 302}});

  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: Values in the |original_id_to_duplicate_id| are not distinct.
  TransformationDuplicateRegionWithSelection transformation_bad_2 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 800, 800, {{800, 100}}, {{13, 201}, {15, 201}},
          {{13, 301}, {15, 302}});
  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: Values in the |original_id_to_phi_id| are not fresh and are not
  // distinct with previous values.
  TransformationDuplicateRegionWithSelection transformation_bad_3 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 800, 800, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 18}, {15, 202}});
  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));

  // Bad: |entry_block_id| does not refer to an existing instruction.
  TransformationDuplicateRegionWithSelection transformation_bad_4 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 802, 800, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 301}, {15, 302}});
  ASSERT_FALSE(
      transformation_bad_4.IsApplicable(context.get(), transformation_context));

  // Bad: |exit_block_id| does not refer to a block.
  TransformationDuplicateRegionWithSelection transformation_bad_5 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 800, 9, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 301}, {15, 302}});
  ASSERT_FALSE(
      transformation_bad_5.IsApplicable(context.get(), transformation_context));

  // Bad: |new_entry_fresh_id| is not fresh.
  TransformationDuplicateRegionWithSelection transformation_bad_6 =
      TransformationDuplicateRegionWithSelection(
          20, 19, 501, 800, 800, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 301}, {15, 302}});
  ASSERT_FALSE(
      transformation_bad_6.IsApplicable(context.get(), transformation_context));

  // Bad: |merge_label_fresh_id| is not fresh.
  TransformationDuplicateRegionWithSelection transformation_bad_7 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 20, 800, 800, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 301}, {15, 302}});
  ASSERT_FALSE(
      transformation_bad_7.IsApplicable(context.get(), transformation_context));

  // Bad: Instruction with id 15 is from the original region and is available
  // at the end of the region but it is not present in the
  // |original_id_to_phi_id|.
  TransformationDuplicateRegionWithSelection transformation_bad_8 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 800, 800, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 301}});
  ASSERT_FALSE(
      transformation_bad_8.IsApplicable(context.get(), transformation_context));

  // Bad: Instruction with id 15 is from the original region but it is
  // not present in the |original_id_to_duplicate_id|.
  TransformationDuplicateRegionWithSelection transformation_bad_9 =
      TransformationDuplicateRegionWithSelection(500, 19, 501, 800, 800,
                                                 {{800, 100}}, {{13, 201}},
                                                 {{13, 301}, {15, 302}});
  ASSERT_FALSE(
      transformation_bad_9.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationDuplicateRegionWithSelectionTest, NotApplicableCFGTest2) {
  // This test handles few cases where the transformation is not applicable
  // because of the control flow graph or the layout of the blocks.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun("
               OpName %10 "s"
               OpName %12 "i"
               OpName %29 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %11 = OpConstant %8 0
         %19 = OpConstant %8 10
         %20 = OpTypeBool
         %26 = OpConstant %8 1
         %28 = OpTypePointer Function %20
         %30 = OpConstantTrue %20
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %29 = OpVariable %28 Function
               OpStore %29 %30
         %31 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %10 = OpVariable %9 Function
         %12 = OpVariable %9 Function
               OpStore %10 %11
               OpStore %12 %11
               OpBranch %13
         %13 = OpLabel
               OpLoopMerge %15 %16 None
               OpBranch %17
         %17 = OpLabel
         %18 = OpLoad %8 %12
         %21 = OpSLessThan %20 %18 %19
               OpBranchConditional %21 %14 %15
         %14 = OpLabel
         %22 = OpLoad %8 %10
         %23 = OpLoad %8 %12
         %24 = OpIAdd %8 %22 %23
               OpStore %10 %24
               OpBranch %16
         %16 = OpLabel
               OpBranch %50
         %50 = OpLabel
         %25 = OpLoad %8 %12
         %27 = OpIAdd %8 %25 %26
               OpStore %12 %27
               OpBranch %13
         %15 = OpLabel
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
  // Bad: The exit block cannot be a header of a loop, because the region won't
  // be a single-entry, single-exit region.
  TransformationDuplicateRegionWithSelection transformation_bad_1 =
      TransformationDuplicateRegionWithSelection(500, 30, 501, 13, 13,
                                                 {{13, 100}}, {{}}, {{}});
  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: The block with id 13, the loop header, is in the region. The block
  // with id 15, the loop merge block, is not in the region.
  TransformationDuplicateRegionWithSelection transformation_bad_2 =
      TransformationDuplicateRegionWithSelection(
          500, 30, 501, 13, 17, {{13, 100}, {17, 101}}, {{18, 201}, {21, 202}},
          {{18, 301}, {21, 302}});
  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: The block with id 13, the loop header, is not in the region. The block
  // with id 16, the loop continue target, is in the region.
  TransformationDuplicateRegionWithSelection transformation_bad_3 =
      TransformationDuplicateRegionWithSelection(
          500, 30, 501, 16, 50, {{16, 100}, {50, 101}}, {{25, 201}, {27, 202}},
          {{25, 301}, {27, 302}});
  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationDuplicateRegionWithSelectionTest, NotApplicableCFGTest3) {
  // This test handles a case where for the block which is not the exit block,
  // not all successors are in the region.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun("
               OpName %14 "a"
               OpName %19 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeBool
          %9 = OpConstantTrue %8
         %12 = OpTypeInt 32 1
         %13 = OpTypePointer Function %12
         %15 = OpConstant %12 2
         %17 = OpConstant %12 3
         %18 = OpTypePointer Function %8
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %19 = OpVariable %18 Function
               OpStore %19 %9
         %20 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %14 = OpVariable %13 Function
               OpSelectionMerge %11 None
               OpBranchConditional %9 %10 %16
         %10 = OpLabel
               OpStore %14 %15
               OpBranch %11
         %16 = OpLabel
               OpStore %14 %17
               OpBranch %11
         %11 = OpLabel
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
  // Bad: The block with id 7, which is not an exit block, has two successors:
  // the block with id 10 and the block with id 16. The block with id 16 is not
  // in the region.
  TransformationDuplicateRegionWithSelection transformation_bad_1 =
      TransformationDuplicateRegionWithSelection(
          500, 30, 501, 7, 10, {{13, 100}}, {{14, 201}}, {{14, 301}});
  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));
}
TEST(TransformationDuplicateRegionWithSelectionTest, MultipleBlocksLoopTest) {
  // This test handles a case where the region consists of multiple blocks
  // (they form a loop). The transformation is applicable and the region is
  // duplicated.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun("
               OpName %10 "s"
               OpName %12 "i"
               OpName %29 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %11 = OpConstant %8 0
         %19 = OpConstant %8 10
         %20 = OpTypeBool
         %26 = OpConstant %8 1
         %28 = OpTypePointer Function %20
         %30 = OpConstantTrue %20
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %29 = OpVariable %28 Function
               OpStore %29 %30
         %31 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %10 = OpVariable %9 Function
         %12 = OpVariable %9 Function
               OpStore %10 %11
               OpStore %12 %11
               OpBranch %50
         %50 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpLoopMerge %15 %16 None
               OpBranch %17
         %17 = OpLabel
         %18 = OpLoad %8 %12
         %21 = OpSLessThan %20 %18 %19
               OpBranchConditional %21 %14 %15
         %14 = OpLabel
         %22 = OpLoad %8 %10
         %23 = OpLoad %8 %12
         %24 = OpIAdd %8 %22 %23
               OpStore %10 %24
               OpBranch %16
         %16 = OpLabel
         %25 = OpLoad %8 %12
         %27 = OpIAdd %8 %25 %26
               OpStore %12 %27
               OpBranch %13
         %15 = OpLabel
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

  TransformationDuplicateRegionWithSelection transformation_good_1 =
      TransformationDuplicateRegionWithSelection(
          500, 30, 501, 50, 15,
          {{50, 100}, {13, 101}, {14, 102}, {15, 103}, {16, 104}, {17, 105}},
          {{22, 201},
           {23, 202},
           {24, 203},
           {25, 204},
           {27, 205},
           {18, 206},
           {21, 207}},
          {{22, 301},
           {23, 302},
           {24, 303},
           {25, 304},
           {27, 305},
           {18, 306},
           {21, 307}});
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
               OpName %6 "fun("
               OpName %10 "s"
               OpName %12 "i"
               OpName %29 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %11 = OpConstant %8 0
         %19 = OpConstant %8 10
         %20 = OpTypeBool
         %26 = OpConstant %8 1
         %28 = OpTypePointer Function %20
         %30 = OpConstantTrue %20
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %29 = OpVariable %28 Function
               OpStore %29 %30
         %31 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %10 = OpVariable %9 Function
         %12 = OpVariable %9 Function
               OpStore %10 %11
               OpStore %12 %11
               OpBranch %500
        %500 = OpLabel
               OpSelectionMerge %501 None
               OpBranchConditional %30 %50 %100
         %50 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpLoopMerge %15 %16 None
               OpBranch %17
         %17 = OpLabel
         %18 = OpLoad %8 %12
         %21 = OpSLessThan %20 %18 %19
               OpBranchConditional %21 %14 %15
         %14 = OpLabel
         %22 = OpLoad %8 %10
         %23 = OpLoad %8 %12
         %24 = OpIAdd %8 %22 %23
               OpStore %10 %24
               OpBranch %16
         %16 = OpLabel
         %25 = OpLoad %8 %12
         %27 = OpIAdd %8 %25 %26
               OpStore %12 %27
               OpBranch %13
         %15 = OpLabel
               OpBranch %501
        %100 = OpLabel
               OpBranch %101
        %101 = OpLabel
               OpLoopMerge %103 %104 None
               OpBranch %105
        %105 = OpLabel
        %206 = OpLoad %8 %12
        %207 = OpSLessThan %20 %206 %19
               OpBranchConditional %207 %102 %103
        %102 = OpLabel
        %201 = OpLoad %8 %10
        %202 = OpLoad %8 %12
        %203 = OpIAdd %8 %201 %202
               OpStore %10 %203
               OpBranch %104
        %104 = OpLabel
        %204 = OpLoad %8 %12
        %205 = OpIAdd %8 %204 %26
               OpStore %12 %205
               OpBranch %101
        %103 = OpLabel
               OpBranch %501
        %501 = OpLabel
        %306 = OpPhi %8 %18 %15 %206 %103
        %307 = OpPhi %20 %21 %15 %207 %103
               OpReturn
               OpFunctionEnd
    )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationDuplicateRegionWithSelectionTest,
     MultipleBlocksNestedLoopTest) {
  // This test handles a case where the region consists of multiple blocks
  // (they form a nested loop). The transformation is applicable and the region
  // is duplicated.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun("
               OpName %10 "s"
               OpName %12 "i"
               OpName %22 "j"
               OpName %38 "r"
               OpName %42 "t"
               OpName %47 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %11 = OpConstant %8 0
         %19 = OpConstant %8 10
         %20 = OpTypeBool
         %34 = OpConstant %8 1
         %44 = OpConstant %8 2
         %46 = OpTypePointer Function %20
         %48 = OpConstantTrue %20
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %47 = OpVariable %46 Function
               OpStore %47 %48
         %49 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %10 = OpVariable %9 Function
         %12 = OpVariable %9 Function
         %22 = OpVariable %9 Function
         %38 = OpVariable %9 Function
         %42 = OpVariable %9 Function
               OpStore %10 %11
               OpStore %12 %11
               OpBranch %13
         %13 = OpLabel
               OpLoopMerge %15 %16 None
               OpBranch %17
         %17 = OpLabel
         %18 = OpLoad %8 %12
         %21 = OpSLessThan %20 %18 %19
               OpBranchConditional %21 %14 %15
         %14 = OpLabel
               OpStore %22 %11
               OpBranch %23
         %23 = OpLabel
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %28 = OpLoad %8 %22
         %29 = OpSLessThan %20 %28 %19
               OpBranchConditional %29 %24 %25
         %24 = OpLabel
         %30 = OpLoad %8 %10
         %31 = OpLoad %8 %12
         %32 = OpIAdd %8 %30 %31
               OpStore %10 %32
               OpBranch %26
         %26 = OpLabel
         %33 = OpLoad %8 %22
         %35 = OpIAdd %8 %33 %34
               OpStore %22 %35
               OpBranch %23
         %25 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %36 = OpLoad %8 %12
         %37 = OpIAdd %8 %36 %34
               OpStore %12 %37
               OpBranch %13
         %15 = OpLabel
         %39 = OpLoad %8 %10
         %40 = OpLoad %8 %10
         %41 = OpIMul %8 %39 %40
               OpStore %38 %41
         %43 = OpLoad %8 %10
         %45 = OpIAdd %8 %43 %44
               OpStore %42 %45
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

  TransformationDuplicateRegionWithSelection transformation_good_1 =
      TransformationDuplicateRegionWithSelection(500, 48, 501, 13, 15,
                                                 {{13, 100},
                                                  {13, 101},
                                                  {17, 102},
                                                  {14, 103},
                                                  {23, 104},
                                                  {27, 105},
                                                  {24, 106},
                                                  {26, 107},
                                                  {25, 108},
                                                  {16, 109},
                                                  {15, 110}},
                                                 {{18, 201},
                                                  {21, 202},
                                                  {28, 203},
                                                  {29, 204},
                                                  {30, 205},
                                                  {31, 206},
                                                  {32, 207},
                                                  {33, 208},
                                                  {35, 209},
                                                  {36, 210},
                                                  {37, 211},
                                                  {39, 212},
                                                  {40, 213},
                                                  {41, 214},
                                                  {43, 215},
                                                  {45, 216}},
                                                 {{18, 301},
                                                  {21, 302},
                                                  {28, 303},
                                                  {29, 304},
                                                  {30, 305},
                                                  {31, 306},
                                                  {32, 307},
                                                  {33, 308},
                                                  {35, 309},
                                                  {36, 310},
                                                  {37, 311},
                                                  {39, 312},
                                                  {40, 313},
                                                  {41, 314},
                                                  {43, 315},
                                                  {45, 316}});
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
               OpName %6 "fun("
               OpName %10 "s"
               OpName %12 "i"
               OpName %22 "j"
               OpName %38 "r"
               OpName %42 "t"
               OpName %47 "b"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %11 = OpConstant %8 0
         %19 = OpConstant %8 10
         %20 = OpTypeBool
         %34 = OpConstant %8 1
         %44 = OpConstant %8 2
         %46 = OpTypePointer Function %20
         %48 = OpConstantTrue %20
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %47 = OpVariable %46 Function
               OpStore %47 %48
         %49 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %10 = OpVariable %9 Function
         %12 = OpVariable %9 Function
         %22 = OpVariable %9 Function
         %38 = OpVariable %9 Function
         %42 = OpVariable %9 Function
               OpStore %10 %11
               OpStore %12 %11
               OpBranch %500
        %500 = OpLabel
               OpSelectionMerge %501 None
               OpBranchConditional %48 %13 %100
         %13 = OpLabel
               OpLoopMerge %15 %16 None
               OpBranch %17
         %17 = OpLabel
         %18 = OpLoad %8 %12
         %21 = OpSLessThan %20 %18 %19
               OpBranchConditional %21 %14 %15
         %14 = OpLabel
               OpStore %22 %11
               OpBranch %23
         %23 = OpLabel
               OpLoopMerge %25 %26 None
               OpBranch %27
         %27 = OpLabel
         %28 = OpLoad %8 %22
         %29 = OpSLessThan %20 %28 %19
               OpBranchConditional %29 %24 %25
         %24 = OpLabel
         %30 = OpLoad %8 %10
         %31 = OpLoad %8 %12
         %32 = OpIAdd %8 %30 %31
               OpStore %10 %32
               OpBranch %26
         %26 = OpLabel
         %33 = OpLoad %8 %22
         %35 = OpIAdd %8 %33 %34
               OpStore %22 %35
               OpBranch %23
         %25 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %36 = OpLoad %8 %12
         %37 = OpIAdd %8 %36 %34
               OpStore %12 %37
               OpBranch %13
         %15 = OpLabel
         %39 = OpLoad %8 %10
         %40 = OpLoad %8 %10
         %41 = OpIMul %8 %39 %40
               OpStore %38 %41
         %43 = OpLoad %8 %10
         %45 = OpIAdd %8 %43 %44
               OpStore %42 %45
               OpBranch %501
        %100 = OpLabel
               OpLoopMerge %110 %109 None
               OpBranch %102
        %102 = OpLabel
        %201 = OpLoad %8 %12
        %202 = OpSLessThan %20 %201 %19
               OpBranchConditional %202 %103 %110
        %103 = OpLabel
               OpStore %22 %11
               OpBranch %104
        %104 = OpLabel
               OpLoopMerge %108 %107 None
               OpBranch %105
        %105 = OpLabel
        %203 = OpLoad %8 %22
        %204 = OpSLessThan %20 %203 %19
               OpBranchConditional %204 %106 %108
        %106 = OpLabel
        %205 = OpLoad %8 %10
        %206 = OpLoad %8 %12
        %207 = OpIAdd %8 %205 %206
               OpStore %10 %207
               OpBranch %107
        %107 = OpLabel
        %208 = OpLoad %8 %22
        %209 = OpIAdd %8 %208 %34
               OpStore %22 %209
               OpBranch %104
        %108 = OpLabel
               OpBranch %109
        %109 = OpLabel
        %210 = OpLoad %8 %12
        %211 = OpIAdd %8 %210 %34
               OpStore %12 %211
               OpBranch %100
        %110 = OpLabel
        %212 = OpLoad %8 %10
        %213 = OpLoad %8 %10
        %214 = OpIMul %8 %212 %213
               OpStore %38 %214
        %215 = OpLoad %8 %10
        %216 = OpIAdd %8 %215 %44
               OpStore %42 %216
               OpBranch %501
        %501 = OpLabel
        %312 = OpPhi %8 %39 %15 %212 %110
        %313 = OpPhi %8 %40 %15 %213 %110
        %314 = OpPhi %8 %41 %15 %214 %110
        %315 = OpPhi %8 %43 %15 %215 %110
        %316 = OpPhi %8 %45 %15 %216 %110
        %301 = OpPhi %8 %18 %15 %201 %110
        %302 = OpPhi %20 %21 %15 %202 %110

               OpReturn
               OpFunctionEnd
        )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(FuzzerPassTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun("
               OpName %10 "s"
               OpName %12 "r"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %11 = OpConstant %8 0
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %16 = OpFunctionCall %2 %6
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %10 = OpVariable %9 Function
         %12 = OpVariable %9 Function
               OpBranch %30
         %30 = OpLabel
               OpStore %10 %11
         %13 = OpLoad %8 %10
         %14 = OpLoad %8 %10
         %15 = OpIAdd %8 %13 %14
               OpStore %12 %15
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;

  auto prng = MakeUnique<PseudoRandomGenerator>(0);

  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  FuzzerContext fuzzer_context(prng.get(), 100);
  protobufs::TransformationSequence transformation_sequence;

  for (uint32_t i = 0; i < 100; i++) {
    FuzzerPassDuplicateRegionsWithSelections fuzzer_pass(
        context.get(), &transformation_context, &fuzzer_context,
        &transformation_sequence);

    fuzzer_pass.Apply();
  }
  // We just check that the result is valid.

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
