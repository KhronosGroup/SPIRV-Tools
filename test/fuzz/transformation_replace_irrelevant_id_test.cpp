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

#include "source/fuzz/transformation_replace_irrelevant_id.h"

#include "source/fuzz/id_use_descriptor.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {
const std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
               OpName %3 "a"
               OpName %4 "b"
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %9 = OpTypeInt 32 1
         %10 = OpTypePointer Function %9
         %11 = OpConstant %9 2
         %12 = OpTypeStruct %9
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 3
         %15 = OpTypeArray %12 %14
         %16 = OpTypePointer Function %15
         %17 = OpConstant %9 0
          %2 = OpFunction %5 None %6
         %18 = OpLabel
          %3 = OpVariable %10 Function
          %4 = OpVariable %10 Function
         %19 = OpVariable %16 Function
               OpStore %3 %11
         %20 = OpLoad %9 %3
         %21 = OpAccessChain %10 %19 %20 %17
         %22 = OpLoad %9 %21
               OpStore %4 %22
         %23 = OpLoad %9 %4
         %24 = OpIAdd %9 %20 %23
         %25 = OpISub %9 %23 %20
               OpReturn
               OpFunctionEnd
)";

void SetUpIrrelevantIdFacts(FactManager* fact_manager) {
  fact_manager->AddFactIdIsIrrelevant(17);
  fact_manager->AddFactIdIsIrrelevant(23);
  fact_manager->AddFactIdIsIrrelevant(24);
  fact_manager->AddFactIdIsIrrelevant(25);
}

TEST(TransformationReplaceIrrelevantIdTest, Inapplicable) {
  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);
  SetUpIrrelevantIdFacts(transformation_context.GetFactManager());

  auto instruction_21_descriptor =
      MakeInstructionDescriptor(21, SpvOpAccessChain, 0);
  auto instruction_24_descriptor = MakeInstructionDescriptor(24, SpvOpIAdd, 0);

  // %20 has not been declared as irrelevant.
  ASSERT_FALSE(TransformationReplaceIrrelevantId(
                   MakeIdUseDescriptor(20, instruction_24_descriptor, 0), 23)
                   .IsApplicable(context.get(), transformation_context));

  // %22 is not used in %24.
  ASSERT_FALSE(TransformationReplaceIrrelevantId(
                   MakeIdUseDescriptor(22, instruction_24_descriptor, 1), 20)
                   .IsApplicable(context.get(), transformation_context));

  // Replacement id %50 does not exist.
  ASSERT_FALSE(TransformationReplaceIrrelevantId(
                   MakeIdUseDescriptor(23, instruction_24_descriptor, 1), 50)
                   .IsApplicable(context.get(), transformation_context));

  // %25 is not available to use at %24.
  ASSERT_FALSE(TransformationReplaceIrrelevantId(
                   MakeIdUseDescriptor(23, instruction_24_descriptor, 1), 25)
                   .IsApplicable(context.get(), transformation_context));

  // %24 is not available to use at %24.
  ASSERT_FALSE(TransformationReplaceIrrelevantId(
                   MakeIdUseDescriptor(23, instruction_24_descriptor, 1), 24)
                   .IsApplicable(context.get(), transformation_context));

  // %8 has not the same type as %23.
  ASSERT_FALSE(TransformationReplaceIrrelevantId(
                   MakeIdUseDescriptor(23, instruction_24_descriptor, 1), 8)
                   .IsApplicable(context.get(), transformation_context));

  // %17 is an index to a struct in an access chain, so it can't be replaced.
  ASSERT_FALSE(TransformationReplaceIrrelevantId(
                   MakeIdUseDescriptor(17, instruction_21_descriptor, 2), 20)
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationReplaceIrrelevantIdTest, Apply) {
  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);
  SetUpIrrelevantIdFacts(transformation_context.GetFactManager());

  auto instruction_24_descriptor = MakeInstructionDescriptor(24, SpvOpIAdd, 0);

  // Replace the use of %23 in %24 with %22.
  auto transformation = TransformationReplaceIrrelevantId(
      MakeIdUseDescriptor(23, instruction_24_descriptor, 1), 22);
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  ApplyAndCheckFreshIds(transformation, context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
               OpName %3 "a"
               OpName %4 "b"
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %9 = OpTypeInt 32 1
         %10 = OpTypePointer Function %9
         %11 = OpConstant %9 2
         %12 = OpTypeStruct %9
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 3
         %15 = OpTypeArray %12 %14
         %16 = OpTypePointer Function %15
         %17 = OpConstant %9 0
          %2 = OpFunction %5 None %6
         %18 = OpLabel
          %3 = OpVariable %10 Function
          %4 = OpVariable %10 Function
         %19 = OpVariable %16 Function
               OpStore %3 %11
         %20 = OpLoad %9 %3
         %21 = OpAccessChain %10 %19 %20 %17
         %22 = OpLoad %9 %21
               OpStore %4 %22
         %23 = OpLoad %9 %4
         %24 = OpIAdd %9 %20 %22
         %25 = OpISub %9 %23 %20
               OpReturn
               OpFunctionEnd
)";

  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
