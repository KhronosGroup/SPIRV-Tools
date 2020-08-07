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

#include "source/fuzz/transformation_flatten_conditional_branch.h"

#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationFlattenConditionalBranchTest, Inapplicable) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main" %3
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeInt 32 1
          %7 = OpTypeInt 32 0
          %8 = OpConstant %7 0
          %9 = OpTypeBool
         %10 = OpConstantTrue %9
         %11 = OpTypePointer Function %6
         %12 = OpTypePointer Workgroup %6
          %3 = OpVariable %12 Workgroup
         %13 = OpConstant %6 2
          %2 = OpFunction %4 None %5
         %14 = OpLabel
               OpBranch %15
         %15 = OpLabel
               OpSelectionMerge %16 None
               OpSwitch %13 %17 2 %18
         %17 = OpLabel
               OpBranch %16
         %18 = OpLabel
               OpBranch %16
         %16 = OpLabel
               OpLoopMerge %19 %16 None
               OpBranchConditional %10 %16 %19
         %19 = OpLabel
               OpSelectionMerge %20 None
               OpBranchConditional %10 %21 %20
         %21 = OpLabel
               OpReturn
         %20 = OpLabel
               OpSelectionMerge %22 None
               OpBranchConditional %10 %23 %22
         %23 = OpLabel
               OpSelectionMerge %24 None
               OpBranchConditional %10 %25 %24
         %25 = OpLabel
               OpBranch %24
         %24 = OpLabel
               OpBranch %22
         %22 = OpLabel
               OpSelectionMerge %26 None
               OpBranchConditional %10 %26 %27
         %27 = OpLabel
               OpBranch %28
         %28 = OpLabel
               OpLoopMerge %29 %28 None
               OpBranchConditional %10 %28 %29
         %29 = OpLabel
               OpBranch %26
         %26 = OpLabel
               OpSelectionMerge %30 None
               OpBranchConditional %10 %30 %31
         %31 = OpLabel
               OpBranch %32
         %32 = OpLabel
         %33 = OpAtomicLoad %6 %3 %8 %8
               OpBranch %30
         %30 = OpLabel
               OpSelectionMerge %34 None
               OpBranchConditional %10 %35 %34
         %35 = OpLabel
               OpMemoryBarrier %8 %8
               OpBranch %34
         %34 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Block %15 does not end with OpBranchConditional.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(15).IsApplicable(
      context.get(), transformation_context));

  // Block %17 is not a selection header.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(17).IsApplicable(
      context.get(), transformation_context));

  // Block %16 is a loop header, not a selection header.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(16).IsApplicable(
      context.get(), transformation_context));

  // Block %19 and the corresponding merge block do not describe a single-entry,
  // single-exit region.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(19).IsApplicable(
      context.get(), transformation_context));

  // Block %20 is the header of a construct containing an inner selection
  // construct.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(20).IsApplicable(
      context.get(), transformation_context));

  // Block %22 is the header of a construct containing an inner loop.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(22).IsApplicable(
      context.get(), transformation_context));

  // Block %26 is the header of a construct containing atomic instructions.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(26).IsApplicable(
      context.get(), transformation_context));

  // Block %30 is the header of a construct containing a barrier instruction.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(30).IsApplicable(
      context.get(), transformation_context));

  // %33 is not a block.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(33).IsApplicable(
      context.get(), transformation_context));
}

TEST(TransformationFlattenConditionalBranchTest, Simple) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeBool
          %4 = OpConstantTrue %3
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %2 = OpFunction %5 None %6
          %7 = OpLabel
               OpSelectionMerge %8 None
               OpBranchConditional %4 %9 %10
         %10 = OpLabel
               OpBranch %8
          %9 = OpLabel
         %11 = OpCopyObject %3 %4
               OpBranch %8
          %8 = OpLabel
         %12 = OpPhi %3 %11 %9 %4 %10
         %23 = OpPhi %3 %4 %9 %4 %10
               OpBranch %13
         %13 = OpLabel
         %14 = OpCopyObject %3 %4
               OpSelectionMerge %15 None
               OpBranchConditional %4 %16 %17
         %16 = OpLabel
               OpBranch %18
         %18 = OpLabel
               OpBranch %19
         %17 = OpLabel
         %20 = OpCopyObject %3 %4
               OpBranch %19
         %19 = OpLabel
         %21 = OpPhi %3 %4 %18 %20 %17
               OpBranch %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto transformation1 = TransformationFlattenConditionalBranch(7);
  ASSERT_TRUE(
      transformation1.IsApplicable(context.get(), transformation_context));
  transformation1.Apply(context.get(), &transformation_context);

  auto transformation2 = TransformationFlattenConditionalBranch(13);
  ASSERT_TRUE(
      transformation2.IsApplicable(context.get(), transformation_context));
  transformation2.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformations = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeBool
          %4 = OpConstantTrue %3
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %2 = OpFunction %5 None %6
          %7 = OpLabel
               OpBranch %9
          %9 = OpLabel
         %11 = OpCopyObject %3 %4
               OpBranch %10
         %10 = OpLabel
               OpBranch %8
          %8 = OpLabel
         %12 = OpSelect %3 %4 %11 %4
         %23 = OpSelect %3 %4 %4 %4
               OpBranch %13
         %13 = OpLabel
         %14 = OpCopyObject %3 %4
               OpBranch %16
         %16 = OpLabel
               OpBranch %18
         %18 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %20 = OpCopyObject %3 %4
               OpBranch %19
         %19 = OpLabel
         %21 = OpSelect %3 %4 %4 %20
               OpBranch %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  ASSERT_TRUE(IsEqual(env, after_transformations, context.get()));
}

TEST(TransformationFlattenConditionalBranchTest, LoadStoreFunctionCall) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
               OpName %3 "func("
               OpName %4 "a"
               OpName %5 "b"
               OpDecorate %3 RelaxedPrecision
               OpDecorate %4 RelaxedPrecision
               OpDecorate %6 RelaxedPrecision
               OpDecorate %7 RelaxedPrecision
               OpDecorate %8 RelaxedPrecision
               OpDecorate %5 RelaxedPrecision
          %9 = OpTypeVoid
         %10 = OpTypeFunction %9
         %11 = OpTypeInt 32 1
         %12 = OpTypeFunction %11
         %13 = OpConstant %11 1
         %14 = OpTypeBool
         %15 = OpConstantTrue %14
         %16 = OpTypePointer Function %11
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 2
         %19 = OpTypeArray %11 %18
         %20 = OpTypePointer Function %19
          %2 = OpFunction %9 None %10
         %21 = OpLabel
          %4 = OpVariable %16 Function
          %5 = OpVariable %20 Function
               OpSelectionMerge %22 None
               OpBranchConditional %15 %22 %23
         %23 = OpLabel
          %6 = OpLoad %11 %4
          %7 = OpIAdd %11 %6 %13
               OpStore %4 %7
               OpBranch %22
         %22 = OpLabel
         %30 = OpPhi %11 %13 %23 %13 %21
               OpSelectionMerge %24 None
               OpBranchConditional %15 %26 %25
         %25 = OpLabel
          %8 = OpFunctionCall %11 %3
               OpStore %4 %8
               OpBranch %27
         %26 = OpLabel
         %28 = OpAccessChain %16 %5 %13
               OpStore %28 %13
               OpBranch %27
         %27 = OpLabel
               OpStore %4 %13
               OpBranch %24
         %24 = OpLabel
               OpStore %4 %13
               OpReturn
               OpFunctionEnd
          %3 = OpFunction %11 None %12
         %29 = OpLabel
               OpReturnValue %13
               OpFunctionEnd
)";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // The selection construct with header block %21 contains an OpLoad and an
  // OpStore, requiring some fresh ids.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(21).IsApplicable(
      context.get(), transformation_context));

  // Not all of the instructions are given in the map and there are not enough
  // overflow ids.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(
                   21, {{MakeInstructionDescriptor(6, SpvOpLoad, 0),
                         {100, 101, 102, 103, 104}}})
                   .IsApplicable(context.get(), transformation_context));

  auto transformation1 = TransformationFlattenConditionalBranch(
      21,
      {{MakeInstructionDescriptor(6, SpvOpLoad, 0), {100, 101, 102, 103, 104}},
       {MakeInstructionDescriptor(6, SpvOpStore, 0), {105, 106, 107}}});

  ASSERT_TRUE(
      transformation1.IsApplicable(context.get(), transformation_context));

  transformation1.Apply(context.get(), &transformation_context);

  auto transformation2 = TransformationFlattenConditionalBranch(
      22,
      {{MakeInstructionDescriptor(8, SpvOpFunctionCall, 0),
        {108, 109, 110, 111, 112}},
       {MakeInstructionDescriptor(8, SpvOpStore, 0), {113, 114}}},
      {115, 116});

  ASSERT_TRUE(
      transformation2.IsApplicable(context.get(), transformation_context));

  transformation2.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformations = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
               OpName %3 "func("
               OpName %4 "a"
               OpName %5 "b"
               OpDecorate %3 RelaxedPrecision
               OpDecorate %4 RelaxedPrecision
               OpDecorate %6 RelaxedPrecision
               OpDecorate %7 RelaxedPrecision
               OpDecorate %8 RelaxedPrecision
               OpDecorate %5 RelaxedPrecision
          %9 = OpTypeVoid
         %10 = OpTypeFunction %9
         %11 = OpTypeInt 32 1
         %12 = OpTypeFunction %11
         %13 = OpConstant %11 1
         %14 = OpTypeBool
         %15 = OpConstantTrue %14
         %16 = OpTypePointer Function %11
         %17 = OpTypeInt 32 0
         %18 = OpConstant %17 2
         %19 = OpTypeArray %11 %18
         %20 = OpTypePointer Function %19
          %2 = OpFunction %9 None %10
         %21 = OpLabel
          %4 = OpVariable %16 Function
          %5 = OpVariable %20 Function
               OpBranch %23
         %23 = OpLabel
               OpSelectionMerge %101 None
               OpBranchConditional %15 %102 %100
        %100 = OpLabel
        %103 = OpLoad %11 %4
               OpBranch %101
        %102 = OpLabel
        %104 = OpUndef %11
               OpBranch %101
        %101 = OpLabel
          %6 = OpPhi %11 %103 %100 %104 %102
          %7 = OpIAdd %11 %6 %13
               OpSelectionMerge %106 None
               OpBranchConditional %15 %106 %105
        %105 = OpLabel
               OpStore %4 %7
               OpBranch %106
        %106 = OpLabel
               OpBranch %22
         %22 = OpLabel
         %30 = OpSelect %11 %15 %13 %13
               OpBranch %26
         %26 = OpLabel
         %28 = OpAccessChain %16 %5 %13
               OpSelectionMerge %116 None
               OpBranchConditional %15 %115 %116
        %115 = OpLabel
               OpStore %28 %13
               OpBranch %116
        %116 = OpLabel
               OpBranch %25
         %25 = OpLabel
               OpSelectionMerge %109 None
               OpBranchConditional %15 %110 %108
        %108 = OpLabel
        %111 = OpFunctionCall %11 %3
               OpBranch %109
        %110 = OpLabel
        %112 = OpUndef %11
               OpBranch %109
        %109 = OpLabel
          %8 = OpPhi %11 %111 %108 %112 %110
               OpSelectionMerge %114 None
               OpBranchConditional %15 %114 %113
        %113 = OpLabel
               OpStore %4 %8
               OpBranch %114
        %114 = OpLabel
               OpBranch %27
         %27 = OpLabel
               OpStore %4 %13
               OpBranch %24
         %24 = OpLabel
               OpStore %4 %13
               OpReturn
               OpFunctionEnd
          %3 = OpFunction %11 None %12
         %29 = OpLabel
               OpReturnValue %13
               OpFunctionEnd
)";

  ASSERT_TRUE(IsEqual(env, after_transformations, context.get()));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
