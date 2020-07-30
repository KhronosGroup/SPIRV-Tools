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

#include "source/fuzz/transformation_composite_insert.h"

#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationCompositeInsertTest, BasicScenarios) {
  // This is a simple transformation and this test handles the main cases.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "i1"
               OpName %10 "i2"
               OpName %12 "base"
               OpMemberName %12 0 "a1"
               OpMemberName %12 1 "a2"
               OpName %14 "b"
               OpName %18 "level_1"
               OpMemberName %18 0 "b1"
               OpMemberName %18 1 "b2"
               OpName %20 "l1"
               OpName %24 "level_2"
               OpMemberName %24 0 "c1"
               OpMemberName %24 1 "c2"
               OpName %26 "l2"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %11 = OpConstant %6 2
         %12 = OpTypeStruct %6 %6
         %13 = OpTypePointer Function %12
         %18 = OpTypeStruct %12 %12
         %19 = OpTypePointer Function %18
         %24 = OpTypeStruct %18 %18
         %25 = OpTypePointer Function %24
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %13 Function
         %20 = OpVariable %19 Function
         %26 = OpVariable %25 Function
               OpStore %8 %9
               OpStore %10 %11
         %15 = OpLoad %6 %8
         %16 = OpLoad %6 %10
         %17 = OpCompositeConstruct %12 %15 %16
               OpStore %14 %17
         %21 = OpLoad %12 %14
         %22 = OpLoad %12 %14
         %23 = OpCompositeConstruct %18 %21 %22
               OpStore %20 %23
         %27 = OpLoad %18 %20
         %28 = OpLoad %18 %20
         %29 = OpCompositeConstruct %24 %27 %28
               OpStore %26 %29
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
  protobufs::InstructionDescriptor instruction_to_insert_before;
  uint32_t fresh_id, composite_id, object_id;
  std::vector<uint32_t> index;

  // Bad: |fresh_id| is not fresh.
  instruction_to_insert_before = MakeInstructionDescriptor(29, SpvOpStore, 0);
  fresh_id = 20;
  composite_id = 29;
  object_id = 11;
  index = {1, 0, 0};
  auto transformation_bad_1 =
      TransformationCompositeInsert(instruction_to_insert_before, fresh_id,
                                    composite_id, object_id, std::move(index));
  ASSERT_FALSE(
      transformation_bad_1.IsApplicable(context.get(), transformation_context));

  // Bad: |composite_id| does not refer to a composite value.
  instruction_to_insert_before = MakeInstructionDescriptor(29, SpvOpStore, 0);
  fresh_id = 50;
  composite_id = 9;
  object_id = 11;
  index = {1, 0, 0};
  auto transformation_bad_2 =
      TransformationCompositeInsert(instruction_to_insert_before, fresh_id,
                                    composite_id, object_id, std::move(index));
  ASSERT_FALSE(
      transformation_bad_2.IsApplicable(context.get(), transformation_context));

  // Bad: |index| is not a valid index.
  instruction_to_insert_before = MakeInstructionDescriptor(29, SpvOpStore, 0);
  fresh_id = 50;
  composite_id = 29;
  object_id = 11;
  index = {2, 0, 0};
  auto transformation_bad_3 =
      TransformationCompositeInsert(instruction_to_insert_before, fresh_id,
                                    composite_id, object_id, std::move(index));
  ASSERT_FALSE(
      transformation_bad_3.IsApplicable(context.get(), transformation_context));

  // Bad: Type id of the object to be inserted and the type id of the
  // componenent at |index| are not the same.
  instruction_to_insert_before = MakeInstructionDescriptor(29, SpvOpStore, 0);
  fresh_id = 50;
  composite_id = 29;
  object_id = 11;
  index = {1, 0};
  auto transformation_bad_4 =
      TransformationCompositeInsert(instruction_to_insert_before, fresh_id,
                                    composite_id, object_id, std::move(index));
  ASSERT_FALSE(
      transformation_bad_4.IsApplicable(context.get(), transformation_context));

  // Bad: |instruction_to_insert_before| does not refer to a valid instruction.
  instruction_to_insert_before = MakeInstructionDescriptor(29, SpvOpIMul, 0);
  fresh_id = 50;
  composite_id = 29;
  object_id = 11;
  index = {1, 0, 0};
  auto transformation_bad_5 =
      TransformationCompositeInsert(instruction_to_insert_before, fresh_id,
                                    composite_id, object_id, std::move(index));
  ASSERT_FALSE(
      transformation_bad_5.IsApplicable(context.get(), transformation_context));

  instruction_to_insert_before = MakeInstructionDescriptor(29, SpvOpStore, 0);
  fresh_id = 50;
  composite_id = 29;
  object_id = 11;
  index = {1, 0, 0};
  auto transformation_good_1 =
      TransformationCompositeInsert(instruction_to_insert_before, fresh_id,
                                    composite_id, object_id, std::move(index));
  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  instruction_to_insert_before = MakeInstructionDescriptor(50, SpvOpStore, 0);
  fresh_id = 51;
  composite_id = 50;
  object_id = 11;
  index = {0, 1, 1};
  auto transformation_good_2 =
      TransformationCompositeInsert(instruction_to_insert_before, fresh_id,
                                    composite_id, object_id, std::move(index));
  ASSERT_TRUE(transformation_good_2.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_2.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "i1"
               OpName %10 "i2"
               OpName %12 "base"
               OpMemberName %12 0 "a1"
               OpMemberName %12 1 "a2"
               OpName %14 "b"
               OpName %18 "level_1"
               OpMemberName %18 0 "b1"
               OpMemberName %18 1 "b2"
               OpName %20 "l1"
               OpName %24 "level_2"
               OpMemberName %24 0 "c1"
               OpMemberName %24 1 "c2"
               OpName %26 "l2"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %11 = OpConstant %6 2
         %12 = OpTypeStruct %6 %6
         %13 = OpTypePointer Function %12
         %18 = OpTypeStruct %12 %12
         %19 = OpTypePointer Function %18
         %24 = OpTypeStruct %18 %18
         %25 = OpTypePointer Function %24
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %13 Function
         %20 = OpVariable %19 Function
         %26 = OpVariable %25 Function
               OpStore %8 %9
               OpStore %10 %11
         %15 = OpLoad %6 %8
         %16 = OpLoad %6 %10
         %17 = OpCompositeConstruct %12 %15 %16
               OpStore %14 %17
         %21 = OpLoad %12 %14
         %22 = OpLoad %12 %14
         %23 = OpCompositeConstruct %18 %21 %22
               OpStore %20 %23
         %27 = OpLoad %18 %20
         %28 = OpLoad %18 %20
         %29 = OpCompositeConstruct %24 %27 %28
         %50 = OpCompositeInsert %24 %11 %29 1 0 0
         %51 = OpCompositeInsert %24 %11 %50 0 1 1
               OpStore %26 %29
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
