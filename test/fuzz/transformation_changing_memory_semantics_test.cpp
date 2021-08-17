// Copyright (c) 2021 Mostafa Ashraf
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

#include "source/fuzz/transformation_changing_memory_semantics.h"

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationChangingMemorySemanticsTest, NotApplicable) {
  const std::string shader = R"(
               OpCapability Shader
               OpCapability Int8
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 320
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeInt 8 1
          %9 = OpTypeInt 32 0
         %26 = OpTypeFloat 32
          %8 = OpTypeStruct %6
         %10 = OpTypePointer StorageBuffer %8
         %11 = OpVariable %10 StorageBuffer
         %19 = OpConstant %26 0
         %18 = OpConstant %9 1
         %12 = OpConstant %6 0
         %13 = OpTypePointer StorageBuffer %6
         %15 = OpConstant %6 4
         %16 = OpConstant %6 7
         %17 = OpConstant %7 4
         %20 = OpConstant %6 80 ; SequentiallyConsistent | UniformMemory
         %97 = OpConstant %6 64 ; None | UniformMemory
         %98 = OpConstant %6 66 ; Acquire | UniformMemory
         %99 = OpConstant %6 68 ; Release | UniformMemory
        %100 = OpConstant %6 72 ; AcquireRelease | UniformMemory
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %21 = OpAtomicLoad %6 %14 %15 %20
         %23 = OpAtomicExchange %6 %14 %15 %100 %16
         %25 = OpAtomicCompareExchange %6 %14 %15 %20 %97 %16 %15
         %24 = OpAccessChain %13 %11 %12
               OpAtomicStore %14 %15 %99 %16
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  // Bad: Instruction does not exist.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(150, SpvOpAtomicLoad, 0), 0, 100)
                   .IsApplicable(context.get(), transformation_context));

#ifndef NDEBUG

  // Bad: Instruction exists, but not an atomic or barrier instruction.
  ASSERT_DEATH(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(19, SpvOpConstant, 0), 0, 101),
               "The instruction does not have any memory semantics operands.");

  // Bad: Operand position does not equal 0 or 1.
  ASSERT_DEATH(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 2, 100),
               "The operand position is out of bounds.");

  // Bad: The SpvOpAtomicLoad takes one memory semantic value and the operand
  // position passed is 1.
  ASSERT_DEATH(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 1, 100),
               "The operand position is out of bounds.");
#endif

  // Bad: The new Id instruction does not exist.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 0, 101)
                   .IsApplicable(context.get(), transformation_context));

  // Bad: The new Id instruction exists, but not OpConstant.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 0, 13)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpAtomicStore Release | UniformMemory
  // to:
  // OpAtomicStore AcquireRelease | UniformMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(24, SpvOpAtomicStore, 0), 0, 100)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpAtomicExchange AcquireRelease | UniformMemory
  // to:
  // OpAtomicExchange None           | UniformMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpAtomicExchange, 0), 0, 97)
                   .IsApplicable(context.get(), transformation_context));

  // Value (one) is pointing to the unequal position operand.
  auto memory_semantics_operand_second_position = 1;
  // Bad: The SpvOpAtomicCompareExchange(read-modify-write) takes two memory
  // semantics operands. The second operand cannot be AcquireRelease value.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(25, SpvOpAtomicCompareExchange, 0),
                   memory_semantics_operand_second_position, 100)
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationChangingMemorySemanticsTest, AtomicInstructionsTestCases) {
  const std::string shader = R"(
               OpCapability Shader
               OpCapability Int8
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 320
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeInt 8 1
          %9 = OpTypeInt 32 0
          %8 = OpTypeStruct %6
         %10 = OpTypePointer StorageBuffer %8
         %11 = OpVariable %10 StorageBuffer
         %18 = OpConstant %9 1
         %12 = OpConstant %6 0
         %13 = OpTypePointer StorageBuffer %6
         %15 = OpConstant %6 4
         %16 = OpConstant %6 7
         %17 = OpConstant %7 4
         %250 = OpConstant %6 66  ; Acquire | UniformMemory
         %26 = OpConstant %6 258 ; Acquire | WorkgroupMemory
         %27 = OpConstant %6 68  ; Release | UniformMemory
         %28 = OpConstant %6 80  ; SequentiallyConsistent | UniformMemory
         %29 = OpConstant %6 64  ; None | UniformMemory
         %30 = OpConstant %6 256 ; None | WorkgroupMemory
         %31 = OpConstant %6 72 ; AcquireRelease | UniformMemory
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %21 = OpAtomicLoad %6 %14 %15 %250
         %22 = OpAtomicLoad %6 %14 %15 %29
         %23 = OpAtomicLoad %6 %14 %15 %27
         %32 = OpAtomicExchange %6 %14 %15 %31 %16
         %33 = OpAtomicCompareExchange %6 %14 %15 %28 %29 %16 %15
         %24 = OpAccessChain %13 %11 %12
               OpAtomicStore %14 %15 %27 %16
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  // Bad transformation:
  // OpAtomicLoad Acquire | UniformMemory
  // to:
  // OpAtomicLoad Acquire | WorkgroupMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 0, 26)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpAtomicLoad Acquire    | UniformMemory
  // to:
  // OpAtomicLoad Release    | UniformMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 0, 27)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpAtomicLoad None    | UniformMemory
  // to:
  // OpAtomicLoad None    | WorkgroupMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(22, SpvOpAtomicLoad, 0), 0, 30)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpAtomicLoad None    | UniformMemory
  // to:
  // OpAtomicLoad Release | UniformMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(22, SpvOpAtomicLoad, 0), 0, 27)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpAtomicLoad Release | UniformMemory
  // to:
  // OpAtomicLoad None    | UniformMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpAtomicLoad, 0), 0, 29)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpAtomicLoad Release | UniformMemory
  // to:
  // OpAtomicLoad Release | UniformMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpAtomicLoad, 0), 0, 27)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpAtomicLoad Release | UniformMemory
  // to:
  // OpAtomicLoad Acquire | UniformMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpAtomicLoad, 0), 0, 250)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpAtomicLoad Release                | UniformMemory
  // to:
  // OpAtomicLoad SequentiallyConsistent | UniformMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpAtomicLoad, 0), 0, 28)
                   .IsApplicable(context.get(), transformation_context));

  // Successful transformations.
  {
    // OpAtomicLoad Acquire                | UniformMemory
    // to:
    // OpAtomicLoad SequentiallyConsistent | UniformMemory
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 0, 28);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }

  {
    // OpAtomicLoad None    | UniformMemory
    // to:
    // OpAtomicLoad Acquire | UniformMemory
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(22, SpvOpAtomicLoad, 0), 0, 250);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }

  {
    // OpAtomicStore Release                | UniformMemory
    // to:
    // OpAtomicStore SequentiallyConsistent | UniformMemory
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(24, SpvOpAtomicStore, 0), 0, 28);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }

  {
    // OpAtomicExchange AcquireRelease         | UniformMemory
    // to:
    // OpAtomicExchange SequentiallyConsistent | UniformMemory
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(32, SpvOpAtomicExchange, 0), 0, 28);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }

  {
    // Value (one) is pointing to the unequal position operand.
    auto memory_semantics_operand_second_position = 1;
    // OpAtomicCompareExchange None                   | UniformMemory
    // to:
    // OpAtomicCompareExchange SequentiallyConsistent | UniformMemory
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(33, SpvOpAtomicCompareExchange, 0),
        memory_semantics_operand_second_position, 28);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }

  const std::string after_transformation = R"(
               OpCapability Shader
               OpCapability Int8
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 320
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeInt 8 1
          %9 = OpTypeInt 32 0
          %8 = OpTypeStruct %6
         %10 = OpTypePointer StorageBuffer %8
         %11 = OpVariable %10 StorageBuffer
         %18 = OpConstant %9 1
         %12 = OpConstant %6 0
         %13 = OpTypePointer StorageBuffer %6
         %15 = OpConstant %6 4
         %16 = OpConstant %6 7
         %17 = OpConstant %7 4
         %250 = OpConstant %6 66  ; Acquire | UniformMemory
         %26 = OpConstant %6 258 ; Acquire | WorkgroupMemory
         %27 = OpConstant %6 68  ; Release | UniformMemory
         %28 = OpConstant %6 80  ; SequentiallyConsistent | UniformMemory
         %29 = OpConstant %6 64  ; None | UniformMemory
         %30 = OpConstant %6 256 ; None | WorkgroupMemory
         %31 = OpConstant %6 72 ; AcquireRelease | UniformMemory
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %21 = OpAtomicLoad %6 %14 %15 %28
         %22 = OpAtomicLoad %6 %14 %15 %250
         %23 = OpAtomicLoad %6 %14 %15 %27
         %32 = OpAtomicExchange %6 %14 %15 %28 %16
         %33 = OpAtomicCompareExchange %6 %14 %15 %28 %28 %16 %15
         %24 = OpAccessChain %13 %11 %12
               OpAtomicStore %14 %15 %28 %16
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationChangingMemorySemanticsTest,
     NotApplicableSequentiallyConsistentWithVulkanMemoryModel) {
  const std::string shader = R"(
               OpCapability Shader
               OpCapability VulkanMemoryModel
               OpExtension "SPV_KHR_vulkan_memory_model"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %4 "main"
               OpExecutionMode %4 LocalSize 1 1 1
               OpSource GLSL 450
               OpSourceExtension "GL_KHR_memory_scope_semantics"
               OpName %4 "main"
               OpName %9 "vertexUniformBuffer"
               OpMemberName %9 0 "transform"
               OpName %11 ""
               OpMemberDecorate %9 0 ColMajor
               OpMemberDecorate %9 0 Offset 0
               OpMemberDecorate %9 0 MatrixStride 16
               OpDecorate %9 Block
               OpDecorate %11 DescriptorSet 0
               OpDecorate %11 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %20 = OpTypeInt 32 1
          %7 = OpTypeVector %6 2
          %8 = OpTypeMatrix %7 2
          %9 = OpTypeStruct %8
         %19 = OpTypeStruct %20
         %10 = OpTypePointer Uniform %9
         %18 = OpTypePointer StorageBuffer %19
         %17 = OpVariable %18 StorageBuffer
         %11 = OpVariable %10 Uniform
         %13 = OpTypePointer StorageBuffer %20
         %12 = OpConstant %20 0
         %25 = OpConstant %20 66  ; Acquire | UniformMemory
         %26 = OpConstant %20 80  ; SequentiallyConsistent | UniformMemory
         %15 = OpConstant %20 4
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %17 %12
         %21 = OpAtomicLoad %20 %14 %15 %25
         %24 = OpAccessChain %13 %17 %12
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);
  // Bad: Can't use Sequentially Consistent with Vulkan memory model.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 0, 26)
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationChangingMemorySemanticsTest, OpMemoryBarrierTestCases) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main"
               OpExecutionMode %4 LocalSize 16 1 1
               OpSource ESSL 320
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 0
          %7 = OpConstant %6 2
         %25 = OpConstant %6 66  ; Acquire | UniformMemory
         %27 = OpConstant %6 68  ; Release | UniformMemory
         %26 = OpConstant %6 264 ; AcquireRelease | WorkgroupMemory
         %28 = OpConstant %6 72  ; AcquireRelease | UniformMemory
         %20 = OpTypePointer Function %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %21 = OpVariable %20 Function %7
          %9 = OpFunctionCall %2 %12
          %8 = OpCopyObject %6 %7
         %22 = OpLoad %6 %21
         %23 = OpFunctionCall %2 %12
               OpMemoryBarrier %7 %25
         %24 = OpFunctionCall %2 %12
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

#ifndef NDEBUG

  // Bad: Instruction exists, but not an atomic or barrier instruction.
  ASSERT_DEATH(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(7, SpvOpConstant, 0), 0, 27),
               "The instruction does not have any memory semantics operands.");

  // Bad: Operand position does not equal 0 or 1.
  ASSERT_DEATH(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpMemoryBarrier, 0), 3, 27),
               "The operand position is out of bounds.");

  // Bad: The SpvOpMemoryBarrier takes one memory semantic value and the operand
  // position passed is 1.
  ASSERT_DEATH(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpMemoryBarrier, 0), 1, 27),
               "The operand position is out of bounds.");
#endif

  // Bad transformation:
  // OpMemoryBarrier Acquire | UniformMemory
  // to:
  // OpMemoryBarrier Release | UniformMemory
  // This will change the semantics.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpMemoryBarrier, 0), 0, 27)
                   .IsApplicable(context.get(), transformation_context));

  // Bad transformation:
  // OpMemoryBarrier Acquire        | UniformMemory
  // to:
  // OpMemoryBarrier AcquireRelease | WorkgroupMemory
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpMemoryBarrier, 0), 0, 26)
                   .IsApplicable(context.get(), transformation_context));

  // Successful transformations.
  {
    // OpMemoryBarrier Acquire        | UniformMemory
    // to:
    // OpMemoryBarrier AcquireRelease | UniformMemory
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(23, SpvOpMemoryBarrier, 0), 0, 28);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main"
               OpExecutionMode %4 LocalSize 16 1 1
               OpSource ESSL 320
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 0
          %7 = OpConstant %6 2
         %25 = OpConstant %6 66  ; Acquire | UniformMemory
         %27 = OpConstant %6 68  ; Release | UniformMemory
         %26 = OpConstant %6 264 ; AcquireRelease | WorkgroupMemory
         %28 = OpConstant %6 72  ; AcquireRelease | UniformMemory
         %20 = OpTypePointer Function %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %21 = OpVariable %20 Function %7
          %9 = OpFunctionCall %2 %12
          %8 = OpCopyObject %6 %7
         %22 = OpLoad %6 %21
         %23 = OpFunctionCall %2 %12
               OpMemoryBarrier %7 %28
         %24 = OpFunctionCall %2 %12
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %2 None %3
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
