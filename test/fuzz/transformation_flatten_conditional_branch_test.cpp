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
               OpSelectionMerge %22 None
               OpBranchConditional %4 %22 %22
         %22 = OpLabel
               OpSelectionMerge %25 None
               OpBranchConditional %4 %24 %24
         %24 = OpLabel
               OpBranch %25
         %25 = OpLabel
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

  auto transformation3 = TransformationFlattenConditionalBranch(15);
  ASSERT_TRUE(
      transformation3.IsApplicable(context.get(), transformation_context));
  transformation3.Apply(context.get(), &transformation_context);

  auto transformation4 = TransformationFlattenConditionalBranch(22);
  ASSERT_TRUE(
      transformation4.IsApplicable(context.get(), transformation_context));
  transformation4.Apply(context.get(), &transformation_context);

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
               OpBranch %22
         %22 = OpLabel
               OpBranch %24
         %24 = OpLabel
               OpBranch %25
         %25 = OpLabel
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
         %12 = OpTypeVector %11 4
         %13 = OpTypeFunction %11
         %14 = OpConstant %11 1
         %15 = OpTypeFloat 32
         %16 = OpTypeVector %15 2
         %17 = OpConstant %15 1
         %18 = OpConstantComposite %16 %17 %17
         %19 = OpTypeBool
         %20 = OpConstantTrue %19
         %21 = OpTypePointer Function %11
         %22 = OpTypeSampler
         %23 = OpTypeImage %9 2D 2 0 0 1 Unknown
         %24 = OpTypeSampledImage %23
         %25 = OpTypePointer Function %23
         %26 = OpTypePointer Function %22
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 2
         %29 = OpTypeArray %11 %28
         %30 = OpTypePointer Function %29
          %2 = OpFunction %9 None %10
         %31 = OpLabel
          %4 = OpVariable %21 Function
          %5 = OpVariable %30 Function
         %32 = OpVariable %25 Function
         %33 = OpVariable %26 Function
         %34 = OpLoad %23 %32
         %35 = OpLoad %22 %33
               OpSelectionMerge %36 None
               OpBranchConditional %20 %37 %36
         %37 = OpLabel
         %38 = OpSampledImage %24 %34 %35
         %39 = OpImageSampleImplicitLod %12 %38 %18
          %6 = OpLoad %11 %4
         %40 = OpSampledImage %24 %34 %35
         %41 = OpImageSampleImplicitLod %12 %40 %18
          %7 = OpIAdd %11 %6 %14
               OpStore %4 %7
               OpBranch %36
         %36 = OpLabel
         %42 = OpPhi %11 %14 %37 %14 %31
               OpSelectionMerge %43 None
               OpBranchConditional %20 %44 %45
         %44 = OpLabel
          %8 = OpFunctionCall %11 %3
               OpStore %4 %8
               OpBranch %46
         %45 = OpLabel
         %47 = OpAccessChain %21 %5 %14
               OpStore %47 %14
               OpBranch %46
         %46 = OpLabel
               OpStore %4 %14
               OpBranch %43
         %43 = OpLabel
               OpStore %4 %14
               OpSelectionMerge %48 None
               OpBranchConditional %20 %49 %48
         %49 = OpLabel
               OpBranch %48
         %48 = OpLabel
               OpSelectionMerge %50 None
               OpBranchConditional %20 %51 %50
         %51 = OpLabel
         %52 = OpSampledImage %24 %34 %35
         %53 = OpLoad %11 %4
         %54 = OpImageSampleImplicitLod %12 %52 %18
               OpBranch %50
         %50 = OpLabel
               OpReturn
               OpFunctionEnd
          %3 = OpFunction %11 None %13
         %55 = OpLabel
               OpReturnValue %14
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

  // The selection construct with header block %31 contains an OpLoad and an
  // OpStore, requiring some fresh ids.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(31).IsApplicable(
      context.get(), transformation_context));

  // Not all of the instructions are given in the map and there are not enough
  // overflow ids.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(
                   31, {{MakeInstructionDescriptor(6, SpvOpLoad, 0),
                         {100, 101, 102, 103, 104}}})
                   .IsApplicable(context.get(), transformation_context));

  // The map maps from an instruction to a list with not enough fresh ids.
  ASSERT_FALSE(
      TransformationFlattenConditionalBranch(
          31,
          {{MakeInstructionDescriptor(6, SpvOpLoad, 0), {100, 101, 102, 103}}},
          {105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115})
          .IsApplicable(context.get(), transformation_context));

  // Not all fresh ids given are distinct.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(
                   31,
                   {{MakeInstructionDescriptor(6, SpvOpLoad, 0),
                     {100, 101, 102, 103, 104}}},
                   {103, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115})
                   .IsApplicable(context.get(), transformation_context));

  // %48 heads a construct where an OpSampledImage instruction is separated from
  // its use by an OpLoad instruction, so the block cannot be split around the
  // OpLoad and, thus, the transformation is not applicable.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(
                   48, {{MakeInstructionDescriptor(53, SpvOpLoad, 0),
                         {100, 101, 102, 103, 104}}})
                   .IsApplicable(context.get(), transformation_context));

  auto transformation1 = TransformationFlattenConditionalBranch(
      31,
      {{MakeInstructionDescriptor(6, SpvOpLoad, 0), {100, 101, 102, 103, 104}},
       {MakeInstructionDescriptor(6, SpvOpStore, 0), {105, 106, 107}}});

  ASSERT_TRUE(
      transformation1.IsApplicable(context.get(), transformation_context));

  transformation1.Apply(context.get(), &transformation_context);

  auto transformation2 = TransformationFlattenConditionalBranch(
      36,
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
         %12 = OpTypeVector %11 4
         %13 = OpTypeFunction %11
         %14 = OpConstant %11 1
         %15 = OpTypeFloat 32
         %16 = OpTypeVector %15 2
         %17 = OpConstant %15 1
         %18 = OpConstantComposite %16 %17 %17
         %19 = OpTypeBool
         %20 = OpConstantTrue %19
         %21 = OpTypePointer Function %11
         %22 = OpTypeSampler
         %23 = OpTypeImage %9 2D 2 0 0 1 Unknown
         %24 = OpTypeSampledImage %23
         %25 = OpTypePointer Function %23
         %26 = OpTypePointer Function %22
         %27 = OpTypeInt 32 0
         %28 = OpConstant %27 2
         %29 = OpTypeArray %11 %28
         %30 = OpTypePointer Function %29
          %2 = OpFunction %9 None %10
         %31 = OpLabel
          %4 = OpVariable %21 Function
          %5 = OpVariable %30 Function
         %32 = OpVariable %25 Function
         %33 = OpVariable %26 Function
         %34 = OpLoad %23 %32
         %35 = OpLoad %22 %33
               OpBranch %37
         %37 = OpLabel
         %38 = OpSampledImage %24 %34 %35
         %39 = OpImageSampleImplicitLod %12 %38 %18
               OpSelectionMerge %101 None
               OpBranchConditional %20 %100 %102
        %100 = OpLabel
        %103 = OpLoad %11 %4
               OpBranch %101
        %102 = OpLabel
        %104 = OpUndef %11
               OpBranch %101
        %101 = OpLabel
          %6 = OpPhi %11 %103 %100 %104 %102
         %40 = OpSampledImage %24 %34 %35
         %41 = OpImageSampleImplicitLod %12 %40 %18
          %7 = OpIAdd %11 %6 %14
               OpSelectionMerge %106 None
               OpBranchConditional %20 %105 %106
        %105 = OpLabel
               OpStore %4 %7
               OpBranch %106
        %106 = OpLabel
               OpBranch %36
         %36 = OpLabel
         %42 = OpSelect %11 %20 %14 %14
               OpBranch %44
         %44 = OpLabel
               OpSelectionMerge %109 None
               OpBranchConditional %20 %108 %110
        %108 = OpLabel
        %111 = OpFunctionCall %11 %3
               OpBranch %109
        %110 = OpLabel
        %112 = OpUndef %11
               OpBranch %109
        %109 = OpLabel
          %8 = OpPhi %11 %111 %108 %112 %110
               OpSelectionMerge %114 None
               OpBranchConditional %20 %113 %114
        %113 = OpLabel
               OpStore %4 %8
               OpBranch %114
        %114 = OpLabel
               OpBranch %45
         %45 = OpLabel
         %47 = OpAccessChain %21 %5 %14
               OpSelectionMerge %116 None
               OpBranchConditional %20 %116 %115
        %115 = OpLabel
               OpStore %47 %14
               OpBranch %116
        %116 = OpLabel
               OpBranch %46
         %46 = OpLabel
               OpStore %4 %14
               OpBranch %43
         %43 = OpLabel
               OpStore %4 %14
               OpSelectionMerge %48 None
               OpBranchConditional %20 %49 %48
         %49 = OpLabel
               OpBranch %48
         %48 = OpLabel
               OpSelectionMerge %50 None
               OpBranchConditional %20 %51 %50
         %51 = OpLabel
         %52 = OpSampledImage %24 %34 %35
         %53 = OpLoad %11 %4
         %54 = OpImageSampleImplicitLod %12 %52 %18
               OpBranch %50
         %50 = OpLabel
               OpReturn
               OpFunctionEnd
          %3 = OpFunction %11 None %13
         %55 = OpLabel
               OpReturnValue %14
               OpFunctionEnd
)";

  ASSERT_TRUE(IsEqual(env, after_transformations, context.get()));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
