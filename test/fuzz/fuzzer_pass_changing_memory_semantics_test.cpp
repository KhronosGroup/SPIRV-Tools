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

  FuzzerContext fuzzer_context(MakeUnique<PseudoRandomGenerator>(0), 100,
                               false);
  protobufs::TransformationSequence transformations;
  FuzzerPassChangingMemorySemantics fuzzer_pass_changing_memory_semantics(
      context.get(), &transformation_context, &fuzzer_context, &transformations,
      false);

  fuzzer_pass_changing_memory_semantics.Apply();

  // Check |GetSuitableNewMemorySemanticsLowerBitValues| method, its should be
  // returns the expected vector below.
  auto ir_context = context.get();
  auto needed_instruction = ir_context->get_def_use_mgr()->GetDef(21);
  auto memory_model = static_cast<SpvMemoryModel>(
      ir_context->module()->GetMemoryModel()->GetSingleWordInOperand(1));
  uint32_t memory_semantics_operand_position = 0;

  auto memory_semantics_value =
      ir_context->get_def_use_mgr()
          ->GetDef(needed_instruction->GetSingleWordInOperand(2))
          ->GetSingleWordInOperand(0);

  auto lower_bits_old_memory_semantics = static_cast<SpvMemorySemanticsMask>(
      memory_semantics_value &
      TransformationChangingMemorySemantics::kMemorySemanticsLowerBitmask);

  std::vector<SpvMemorySemanticsMask> expected_potential_new_memory_orders{
      SpvMemorySemanticsSequentiallyConsistentMask};
  auto actual_potential_new_memory_orders = FuzzerPassChangingMemorySemantics::
      GetSuitableNewMemorySemanticsLowerBitValues(
          ir_context, needed_instruction, lower_bits_old_memory_semantics,
          memory_semantics_operand_position, memory_model);
  ASSERT_TRUE(expected_potential_new_memory_orders ==
              actual_potential_new_memory_orders);
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
