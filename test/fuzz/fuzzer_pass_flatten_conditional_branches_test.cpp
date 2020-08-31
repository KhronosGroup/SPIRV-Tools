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

#include "source/fuzz/fuzzer_pass_flatten_conditional_branches.h"
#include "source/fuzz/pseudo_random_generator.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %7 = OpTypeInt 32 1
          %8 = OpTypePointer Function %7
          %9 = OpConstant %7 1
         %10 = OpConstant %7 10
         %11 = OpConstant %7 2
          %2 = OpFunction %3 None %4
         %12 = OpLabel
               OpSelectionMerge %13 None
               OpBranchConditional %6 %14 %15
         %14 = OpLabel
               OpBranch %13
         %15 = OpLabel
               OpBranch %16
         %16 = OpLabel
               OpLoopMerge %17 %18 None
               OpBranch %19
         %19 = OpLabel
               OpBranchConditional %6 %20 %17
         %20 = OpLabel
               OpSelectionMerge %21 None
               OpBranchConditional %6 %22 %23
         %22 = OpLabel
               OpBranch %21
         %23 = OpLabel
               OpBranch %21
         %21 = OpLabel
               OpBranch %18
         %18 = OpLabel
               OpBranch %16
         %17 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
)";

TEST(FuzzerPassFlattenConditionalBranches, NestingDepth) {
  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Check the nesting depth of a few blocks.
  ASSERT_EQ(
      FuzzerPassFlattenConditionalBranches::NestingDepth(context.get(), 22), 3);
  ASSERT_EQ(
      FuzzerPassFlattenConditionalBranches::NestingDepth(context.get(), 21), 2);
  ASSERT_EQ(
      FuzzerPassFlattenConditionalBranches::NestingDepth(context.get(), 20), 2);
  ASSERT_EQ(
      FuzzerPassFlattenConditionalBranches::NestingDepth(context.get(), 19), 2);
  ASSERT_EQ(
      FuzzerPassFlattenConditionalBranches::NestingDepth(context.get(), 16), 1);
  ASSERT_EQ(
      FuzzerPassFlattenConditionalBranches::NestingDepth(context.get(), 15), 1);
  ASSERT_EQ(
      FuzzerPassFlattenConditionalBranches::NestingDepth(context.get(), 13), 0);
  ASSERT_EQ(
      FuzzerPassFlattenConditionalBranches::NestingDepth(context.get(), 12), 0);
}

TEST(FuzzerPassFlattenConditionalBranches, Comparator) {
  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Check that, sorting using the comparator, the blocks are ordered from more
  // deeply nested to less deeply nested.
  // 17 has depth 1, 20 has depth 2, 13 has depth 0.
  std::vector<opt::BasicBlock*> blocks = {context->get_instr_block(17),
                                          context->get_instr_block(20),
                                          context->get_instr_block(13)};

  std::sort(blocks.begin(), blocks.end(),
            FuzzerPassFlattenConditionalBranches::LessIfNestedMoreDeeply(
                context.get()));

  // Check that the blocks are in the correct order.
  ASSERT_EQ(blocks[0]->id(), 20);
  ASSERT_EQ(blocks[1]->id(), 17);
  ASSERT_EQ(blocks[2]->id(), 13);
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
