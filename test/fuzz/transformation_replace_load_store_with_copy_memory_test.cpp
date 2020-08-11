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

#include "source/fuzz/transformation_replace_load_store_with_copy_memory.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationReplaceLoadStoreWithCopyMemoryTest, BasicScenarios) {
  // This is a simple transformation and this test handles the main cases.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %34 %35
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun1("
               OpName %10 "a"
               OpName %12 "b"
               OpName %15 "a"
               OpName %17 "b"
               OpName %19 "c"
               OpName %21 "d"
               OpName %25 "e"
               OpName %27 "f"
               OpName %34 "i1"
               OpName %35 "i2"
               OpDecorate %34 Location 0
               OpDecorate %35 Location 1
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %11 = OpConstant %8 0
         %13 = OpConstant %8 1
         %16 = OpConstant %8 2
         %18 = OpConstant %8 3
         %20 = OpConstant %8 4
         %22 = OpConstant %8 5
         %23 = OpTypeFloat 32
         %24 = OpTypePointer Function %23
         %26 = OpConstant %23 2
         %28 = OpConstant %23 3
         %33 = OpTypePointer Output %8
         %34 = OpVariable %33 Output
         %35 = OpVariable %33 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpVariable %9 Function
         %17 = OpVariable %9 Function
         %19 = OpVariable %9 Function
         %21 = OpVariable %9 Function
         %25 = OpVariable %24 Function
         %27 = OpVariable %24 Function
               OpStore %15 %16
               OpStore %17 %18
               OpStore %19 %20
               OpStore %21 %22
               OpStore %25 %26
               OpStore %27 %28
         %29 = OpLoad %8 %15
               OpCopyMemory %17 %15
               OpStore %17 %29
         %30 = OpLoad %8 %19
               OpStore %21 %30
         %31 = OpLoad %23 %25
               OpStore %27 %31
         %32 = OpFunctionCall %2 %6
               OpStore %34 %13
               OpStore %35 %13
         %36 = OpLoad %8 %34
               OpMemoryBarrier %11 %11
               OpStore %35 %36
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %10 = OpVariable %9 Function
         %12 = OpVariable %9 Function
               OpStore %10 %11
               OpStore %12 %13
         %14 = OpLoad %8 %10
               OpStore %12 %14
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);
  ASSERT_TRUE(IsValid(env, context.get()));

  auto bad_instruction_descriptor_1 =
      MakeInstructionDescriptor(11, SpvOpConstant, 0);

  auto load_instruction_descriptor_1 =
      MakeInstructionDescriptor(29, SpvOpLoad, 0);
  auto load_instruction_descriptor_2 =
      MakeInstructionDescriptor(30, SpvOpLoad, 0);
  auto load_instruction_descriptor_3 =
      MakeInstructionDescriptor(31, SpvOpLoad, 0);
  auto load_instruction_descriptor_unsafe =
      MakeInstructionDescriptor(36, SpvOpLoad, 0);

  auto store_instruction_descriptor_1 =
      MakeInstructionDescriptor(29, SpvOpStore, 0);
  auto store_instruction_descriptor_2 =
      MakeInstructionDescriptor(30, SpvOpStore, 0);
  auto store_instruction_descriptor_3 =
      MakeInstructionDescriptor(31, SpvOpStore, 0);
  auto store_instruction_descriptor_unsafe =
      MakeInstructionDescriptor(36, SpvOpStore, 0);

  // Bad: |load_instruction_descriptor| is incorrect.
  auto transformation_bad_1 = TransformationReplaceLoadStoreWithCopyMemory(
      bad_instruction_descriptor_1, store_instruction_descriptor_1);
  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: |store_instruction_descriptor| is incorrect.
  auto transformation_bad_2 = TransformationReplaceLoadStoreWithCopyMemory(
      load_instruction_descriptor_1, bad_instruction_descriptor_1);
  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: Intermediate values of the OpLoad and the OpStore don't match.
  auto transformation_bad_3 = TransformationReplaceLoadStoreWithCopyMemory(
      load_instruction_descriptor_1, store_instruction_descriptor_2);
  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));

  // Bad: There is an interfering OpCopyMemory instruction between the OpLoad
  // and the OpStore.
  auto transformation_bad_4 = TransformationReplaceLoadStoreWithCopyMemory(
      load_instruction_descriptor_1, store_instruction_descriptor_1);
  ASSERT_FALSE(
      transformation_bad_4.IsApplicable(context.get(), transformation_context));

  // Bad: There is an interfering OpMemoryBarrier instruction between the OpLoad
  // and the OpStore.
  auto transformation_bad_5 = TransformationReplaceLoadStoreWithCopyMemory(
      load_instruction_descriptor_unsafe, store_instruction_descriptor_unsafe);
  ASSERT_FALSE(
      transformation_bad_5.IsApplicable(context.get(), transformation_context));

  auto transformation_good_1 = TransformationReplaceLoadStoreWithCopyMemory(
      load_instruction_descriptor_2, store_instruction_descriptor_2);
  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  auto transformation_good_2 = TransformationReplaceLoadStoreWithCopyMemory(
      load_instruction_descriptor_3, store_instruction_descriptor_3);
  ASSERT_TRUE(transformation_good_2.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_2.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformations = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %34 %35
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "fun1("
               OpName %10 "a"
               OpName %12 "b"
               OpName %15 "a"
               OpName %17 "b"
               OpName %19 "c"
               OpName %21 "d"
               OpName %25 "e"
               OpName %27 "f"
               OpName %34 "i1"
               OpName %35 "i2"
               OpDecorate %34 Location 0
               OpDecorate %35 Location 1
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %11 = OpConstant %8 0
         %13 = OpConstant %8 1
         %16 = OpConstant %8 2
         %18 = OpConstant %8 3
         %20 = OpConstant %8 4
         %22 = OpConstant %8 5
         %23 = OpTypeFloat 32
         %24 = OpTypePointer Function %23
         %26 = OpConstant %23 2
         %28 = OpConstant %23 3
         %33 = OpTypePointer Output %8
         %34 = OpVariable %33 Output
         %35 = OpVariable %33 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpVariable %9 Function
         %17 = OpVariable %9 Function
         %19 = OpVariable %9 Function
         %21 = OpVariable %9 Function
         %25 = OpVariable %24 Function
         %27 = OpVariable %24 Function
               OpStore %15 %16
               OpStore %17 %18
               OpStore %19 %20
               OpStore %21 %22
               OpStore %25 %26
               OpStore %27 %28
         %29 = OpLoad %8 %15
               OpCopyMemory %17 %15
               OpStore %17 %29
         %30 = OpLoad %8 %19
               OpCopyMemory %21 %19
         %31 = OpLoad %23 %25
               OpCopyMemory %27 %25
         %32 = OpFunctionCall %2 %6
               OpStore %34 %13
               OpStore %35 %13
         %36 = OpLoad %8 %34
               OpMemoryBarrier %11 %11
               OpStore %35 %36
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %10 = OpVariable %9 Function
         %12 = OpVariable %9 Function
               OpStore %10 %11
               OpStore %12 %13
         %14 = OpLoad %8 %10
               OpStore %12 %14
               OpReturn
               OpFunctionEnd
    )";
  ASSERT_TRUE(IsEqual(env, after_transformations, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
