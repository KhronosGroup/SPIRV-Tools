// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_move_instruction_down.h"

#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationMoveInstructionDownTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %9 = OpConstant %6 0
         %16 = OpTypeBool
         %17 = OpConstantFalse %16
         %20 = OpUndef %6
         %13 = OpTypePointer Function %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %12 = OpVariable %13 Function
         %10 = OpIAdd %6 %9 %9
         %11 = OpISub %6 %9 %10
               OpStore %12 %10
         %14 = OpLoad %6 %12
         %15 = OpIMul %6 %9 %14
               OpSelectionMerge %19 None
               OpBranchConditional %17 %18 %19
         %18 = OpLabel
               OpBranch %19
         %19 = OpLabel
         %22 = OpIAdd %6 %15 %15
         %21 = OpIAdd %6 %15 %15
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Instruction descriptor is invalid.
  ASSERT_FALSE(TransformationMoveInstructionDown(
                   MakeInstructionDescriptor(30, SpvOpNop, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Opcode is not supported.
  ASSERT_FALSE(TransformationMoveInstructionDown(
                   MakeInstructionDescriptor(5, SpvOpLabel, 0))
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationMoveInstructionDown(
                   MakeInstructionDescriptor(12, SpvOpVariable, 0))
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationMoveInstructionDown(
                   MakeInstructionDescriptor(11, SpvOpStore, 0))
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationMoveInstructionDown(
                   MakeInstructionDescriptor(14, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Can't move the last instruction in the block.
  ASSERT_FALSE(TransformationMoveInstructionDown(
                   MakeInstructionDescriptor(15, SpvOpBranchConditional, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Can't move the instruction if the next instruction is the last one in the
  // block.
  ASSERT_FALSE(TransformationMoveInstructionDown(
                   MakeInstructionDescriptor(21, SpvOpIAdd, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Can't insert instruction's opcode after its successor.
  ASSERT_FALSE(TransformationMoveInstructionDown(
                   MakeInstructionDescriptor(15, SpvOpIMul, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Instruction's successor depends on the instruction.
  ASSERT_FALSE(TransformationMoveInstructionDown(
                   MakeInstructionDescriptor(10, SpvOpIAdd, 0))
                   .IsApplicable(context.get(), transformation_context));

  {
    TransformationMoveInstructionDown transformation(
        MakeInstructionDescriptor(11, SpvOpISub, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
  }
  {
    TransformationMoveInstructionDown transformation(
        MakeInstructionDescriptor(22, SpvOpIAdd, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
  }

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %9 = OpConstant %6 0
         %16 = OpTypeBool
         %17 = OpConstantFalse %16
         %20 = OpUndef %6
         %13 = OpTypePointer Function %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %12 = OpVariable %13 Function
         %10 = OpIAdd %6 %9 %9
               OpStore %12 %10
         %11 = OpISub %6 %9 %10
         %14 = OpLoad %6 %12
         %15 = OpIMul %6 %9 %14
               OpSelectionMerge %19 None
               OpBranchConditional %17 %18 %19
         %18 = OpLabel
               OpBranch %19
         %19 = OpLabel
         %21 = OpIAdd %6 %15 %15
         %22 = OpIAdd %6 %15 %15
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
