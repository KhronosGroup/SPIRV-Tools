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
  ASSERT_FALSE(TransformationFlattenConditionalBranch(15, {}).IsApplicable(
      context.get(), transformation_context));

  // Block %17 is not a selection header.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(17, {}).IsApplicable(
      context.get(), transformation_context));

  // Block %16 is a loop header, not a selection header.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(16, {}).IsApplicable(
      context.get(), transformation_context));

  // Block %19 and the corresponding merge block do not describe a single-entry,
  // single-exit region.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(19, {}).IsApplicable(
      context.get(), transformation_context));

  // Block %20 is the header of a construct containing an inner selection
  // construct.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(20, {}).IsApplicable(
      context.get(), transformation_context));

  // Block %22 is the header of a construct containing an inner loop.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(22, {}).IsApplicable(
      context.get(), transformation_context));

  // Block %26 is the header of a construct containing atomic instructions.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(26, {}).IsApplicable(
      context.get(), transformation_context));

  // Block %30 is the header of a construct containing a barrier instruction.
  ASSERT_FALSE(TransformationFlattenConditionalBranch(30, {}).IsApplicable(
      context.get(), transformation_context));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
