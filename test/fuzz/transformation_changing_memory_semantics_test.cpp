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
         %20 = OpConstant %6 16 ; SequentiallyConsistent mode
         %96 = OpConstant %6 64 ; UniformMemory mode
         %97 = OpConstant %6 0 ; None mode
         %98 = OpConstant %6 2 ; Acquire mode
         %99 = OpConstant %6 4 ; Release mode
        %100 = OpConstant %6 8 ; AcquireRelease mode
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %21 = OpAtomicLoad %6 %14 %15 %20
         %22 = OpAtomicLoad %6 %14 %15 %98
         %23 = OpAtomicExchange %6 %14 %15 %98 %16
         %25 = OpAtomicCompareExchange %6 %14 %15 %20 %20 %16 %15
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

  // Instruction does not exist.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(150, SpvOpAtomicLoad, 0), 2, 100)
                   .IsApplicable(context.get(), transformation_context));

  // Operand index does not equal 2 or 3.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 1, 100)
                   .IsApplicable(context.get(), transformation_context));

  // The Instruction is not an atomic operation.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(14, SpvOpAccessChain, 0), 2, 100)
                   .IsApplicable(context.get(), transformation_context));

  // The atomic operation takes one memory semantic value and the operand index
  // passed is 3.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 3, 100)
                   .IsApplicable(context.get(), transformation_context));

  // The new Id instruction does not exist.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 2, 101)
                   .IsApplicable(context.get(), transformation_context));

  // The new Id instruction exists, but not OpConstant.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 2, 13)
                   .IsApplicable(context.get(), transformation_context));

  // The new memory semantics value is smaller than old.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(21, SpvOpAtomicLoad, 0), 2, 98)
                   .IsApplicable(context.get(), transformation_context));

  // The OpAtomicLoad instruction cannot be in the release mode.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(22, SpvOpAtomicLoad, 0), 2, 99)
                   .IsApplicable(context.get(), transformation_context));

  // The OpAtomicStore instruction cannot be in the AcquireRelease mode.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(24, SpvOpAtomicStore, 0), 2, 100)
                   .IsApplicable(context.get(), transformation_context));

  // The OpAtomicExchange(read-modify-write) instruction cannot be in the
  // UniformMemory mode.
  ASSERT_FALSE(TransformationChangingMemorySemantics(
                   MakeInstructionDescriptor(23, SpvOpAtomicExchange, 0), 2, 96)
                   .IsApplicable(context.get(), transformation_context));

  // The SpvOpAtomicCompareExchange(read-modify-write) instruction that takes
  // two memory semantics, one of the value must be stronger than the other.
  ASSERT_FALSE(
      TransformationChangingMemorySemantics(
          MakeInstructionDescriptor(25, SpvOpAtomicCompareExchange, 0), 2, 100)
          .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationChangingMemorySemanticsTest, IsApplicable) {
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
         %20 = OpConstant %6 16 ; SequentiallyConsistent mode
         %97 = OpConstant %6 0 ; None mode
         %98 = OpConstant %6 2 ; Acquire mode
         %99 = OpConstant %6 4 ; Release mode
        %100 = OpConstant %6 8 ; AcquireRelease mode
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %22 = OpAtomicLoad %6 %14 %15 %98
         %23 = OpAtomicExchange %6 %14 %15 %100 %16
         %25 = OpAtomicCompareExchange %6 %14 %15 %97 %20 %16 %15
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

  // Successful transformations.
  {
    // The old memory semantics of id 98 (Acquire) mode will be a value of id 20
    // (SequentiallyConsistent) mode.
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(22, SpvOpAtomicLoad, 0), 2, 20);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }
  {
    // The old memory semantics of id 99 (Release) mode will be a value of id 20
    // (SequentiallyConsistent) mode.
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(24, SpvOpAtomicStore, 0), 2, 20);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }
  {
    // The old memory semantics of id 100 (AcquireRelease) mode will be a value
    // of id 20 (SequentiallyConsistent) mode.
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(23, SpvOpAtomicExchange, 0), 2, 20);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }

  {
    // The old memory semantics of id 97 (None-Relaxed) mode for index 2 will be
    // a value of id 100 (AcquireRelease) mode for the same index.
    TransformationChangingMemorySemantics transformation(
        MakeInstructionDescriptor(25, SpvOpAtomicCompareExchange, 0), 2, 100);
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
         %20 = OpConstant %6 16 ; SequentiallyConsistent mode
         %97 = OpConstant %6 0 ; None mode
         %98 = OpConstant %6 2 ; Acquire mode
         %99 = OpConstant %6 4 ; Release mode
        %100 = OpConstant %6 8 ; AcquireRelease mode
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %22 = OpAtomicLoad %6 %14 %15 %20
         %23 = OpAtomicExchange %6 %14 %15 %20 %16
         %25 = OpAtomicCompareExchange %6 %14 %15 %100 %20 %16 %15
         %24 = OpAccessChain %13 %11 %12
               OpAtomicStore %14 %15 %20 %16
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
