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
          %9 = OpLabel
         %11 = OpCopyObject %3 %4
               OpBranch %8
         %10 = OpLabel
               OpBranch %8
          %8 = OpLabel
         %12 = OpCopyObject %3 %4
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
         %21 = OpCopyObject %3 %4
               OpBranch %15
         %15 = OpLabel
         %22 = OpCopyObject %3 %4
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

  ASSERT_TRUE(TransformationFlattenConditionalBranch(7).IsApplicable(
      context.get(), transformation_context));

  ASSERT_TRUE(TransformationFlattenConditionalBranch(13).IsApplicable(
      context.get(), transformation_context));
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
               OpBranchConditional %15 %23 %22
         %23 = OpLabel
          %6 = OpLoad %11 %4
          %7 = OpIAdd %11 %6 %13
               OpStore %4 %7
               OpBranch %22
         %22 = OpLabel
         %30 = OpPhi %11 %13 %23 %13 %21
               OpSelectionMerge %24 None
               OpBranchConditional %15 %25 %26
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

  std::cout << ToString(env, context.get()) << "\n\n";

  ASSERT_TRUE(IsValid(env, context.get()));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
