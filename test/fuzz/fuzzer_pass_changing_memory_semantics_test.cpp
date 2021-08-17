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

#include "source/fuzz/fuzzer_pass_changing_memory_semantics.h"

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/pseudo_random_generator.h"
#include "source/fuzz/transformation_changing_memory_semantics.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(FuzzerPassChangingMemorySemanticsTest,
     TestGetSuitableNewMemorySemanticsLowerBitValues) {
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
         %18 = OpConstant %6 2
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
               OpMemoryBarrier %18 %31
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

  FuzzerContext fuzzer_context(MakeUnique<PseudoRandomGenerator>(0), 100,
                               false);
  protobufs::TransformationSequence transformations;
  FuzzerPassChangingMemorySemantics fuzzer_pass_changing_memory_semantics(
      context.get(), &transformation_context, &fuzzer_context, &transformations,
      false);

  fuzzer_pass_changing_memory_semantics.Apply();

  auto ir_context = context.get();

  // Check |GetSuitableNewMemorySemanticsLowerBitValues| method, its should be
  // returns the expected vector as the examples below.

  // OpAtomicLoad: None -> [Acquire, SequentiallyConsistent].
  {
    auto atomic_load_instruction = ir_context->get_def_use_mgr()->GetDef(21);
    auto old_value = SpvMemorySemanticsMaskNone;
    auto expected = std::vector<SpvMemorySemanticsMask>{
        SpvMemorySemanticsAcquireMask,
        SpvMemorySemanticsSequentiallyConsistentMask};
    auto actual = FuzzerPassChangingMemorySemantics::
        GetSuitableNewMemorySemanticsLowerBitValues(
            context.get(), atomic_load_instruction, old_value, 0,
            SpvMemoryModelGLSL450);
    ASSERT_EQ(expected, actual);
  }

  // OpAtomicLoad: Acquire -> [SequentiallyConsistent].
  {
    auto atomic_load_instruction = ir_context->get_def_use_mgr()->GetDef(22);
    auto old_value = SpvMemorySemanticsAcquireMask;
    auto expected = std::vector<SpvMemorySemanticsMask>{
        SpvMemorySemanticsSequentiallyConsistentMask};
    auto actual = FuzzerPassChangingMemorySemantics::
        GetSuitableNewMemorySemanticsLowerBitValues(
            context.get(), atomic_load_instruction, old_value, 0,
            SpvMemoryModelGLSL450);
    ASSERT_EQ(expected, actual);
  }

  // OpAtomicLoad: Acquire -> []
  // (with Vulkan memory model).
  {
    auto atomic_load_instruction = ir_context->get_def_use_mgr()->GetDef(22);
    auto old_value = SpvMemorySemanticsAcquireMask;
    auto expected = std::vector<SpvMemorySemanticsMask>{};
    auto actual = FuzzerPassChangingMemorySemantics::
        GetSuitableNewMemorySemanticsLowerBitValues(
            context.get(), atomic_load_instruction, old_value, 0,
            SpvMemoryModelVulkan);
    ASSERT_EQ(expected, actual);
  }

  // OpAtomicExchange: AcquireRelease -> [SequentiallyConsistent].
  {
    auto atomic_exchange_instruction =
        ir_context->get_def_use_mgr()->GetDef(32);
    auto old_value = SpvMemorySemanticsAcquireReleaseMask;
    auto expected = std::vector<SpvMemorySemanticsMask>{
        SpvMemorySemanticsSequentiallyConsistentMask};
    auto actual = FuzzerPassChangingMemorySemantics::
        GetSuitableNewMemorySemanticsLowerBitValues(
            context.get(), atomic_exchange_instruction, old_value, 0,
            SpvMemoryModelGLSL450);
    ASSERT_EQ(expected, actual);
  }

  // OpAtomicCompareExchange: None -> [SequentiallyConsistent].
  {
    auto atomic_compare_exchange_instruction =
        ir_context->get_def_use_mgr()->GetDef(33);
    auto old_value = SpvMemorySemanticsMaskNone;
    auto expected = std::vector<SpvMemorySemanticsMask>{
        SpvMemorySemanticsSequentiallyConsistentMask};
    auto actual = FuzzerPassChangingMemorySemantics::
        GetSuitableNewMemorySemanticsLowerBitValues(
            context.get(), atomic_compare_exchange_instruction, old_value, 1,
            SpvMemoryModelGLSL450);
    ASSERT_EQ(expected, actual);
  }

  // OpMemoryBarrier: AcquireRelease -> [SequentiallyConsistent].
  {
    auto memory_barrier_instruction = FindInstruction(
        MakeInstructionDescriptor(24, SpvOpMemoryBarrier, 0), ir_context);
    auto old_value = SpvMemorySemanticsAcquireReleaseMask;
    auto expected = std::vector<SpvMemorySemanticsMask>{
        SpvMemorySemanticsSequentiallyConsistentMask};
    auto actual = FuzzerPassChangingMemorySemantics::
        GetSuitableNewMemorySemanticsLowerBitValues(
            context.get(), memory_barrier_instruction, old_value, 0,
            SpvMemoryModelGLSL450);
    ASSERT_EQ(expected, actual);
  }
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
