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

#include "source/fuzz/transformation_add_read_modify_write_atomic_instruction.h"

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddReadModifyWriteAtomicInstructionTest, NotApplicable) {
  const std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main" %16
               OpExecutionMode %4 LocalSize 1 1 1
               OpSource GLSL 450
               OpName %4 "main"
               OpName %8 "result"
               OpName %10 "temp_val"
               OpName %13 "id"
               OpName %16 "gl_LocalInvocationID"
               OpName %21 "chunkStart"
               OpName %23 "i"
               OpName %37 "shared_memory"
               OpDecorate %16 BuiltIn LocalInvocationId
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %11 = OpTypeInt 32 0
         %12 = OpTypePointer Function %11
         %14 = OpTypeVector %11 3
         %15 = OpTypePointer Input %14
         %16 = OpVariable %15 Input
         %17 = OpConstant %11 272
         %18 = OpTypePointer Input %11
         %32 = OpTypeBool
         %34 = OpConstant %11 16
         %35 = OpTypeArray %11 %34
         %36 = OpTypePointer Workgroup %35
         %37 = OpVariable %36 Workgroup
         %39 = OpTypePointer Workgroup %11
         %41 = OpConstant %11 4
         %43 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %13 = OpVariable %12 Function
         %21 = OpVariable %12 Function
         %23 = OpVariable %12 Function
               OpStore %8 %9
               OpStore %10 %9
         %19 = OpAccessChain %18 %16 %17
         %20 = OpLoad %11 %19
               OpStore %13 %20
         %22 = OpLoad %11 %13
               OpStore %21 %22
         %24 = OpLoad %11 %21
               OpStore %23 %24
               OpBranch %25
         %25 = OpLabel
               OpLoopMerge %27 %28 None
               OpBranch %29
         %29 = OpLabel
         %30 = OpLoad %11 %23
         %31 = OpLoad %11 %21
         %33 = OpULessThan %32 %30 %31
               OpBranchConditional %33 %26 %27
         %26 = OpLabel
         %38 = OpLoad %11 %23
         %40 = OpAccessChain %39 %37 %38
         %44 = OpLoad %6 %10
         %45 = OpIAdd %6 %44 %43
               OpStore %10 %45
               OpBranch %28
         %28 = OpLabel
         %46 = OpLoad %11 %23
         %47 = OpIAdd %11 %46 %43
               OpStore %23 %47
               OpBranch %25
         %27 = OpLabel
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

  // Bad: id is not fresh.
  ASSERT_FALSE(TransformationAddReadModifyWriteAtomicInstruction(
                   40, 40, static_cast<uint32_t>(SpvOpAtomicIAdd), 41, 17, 0,
                   41, 0, MakeInstructionDescriptor(44, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));
  // Pointer does not exist.
  ASSERT_FALSE(TransformationAddReadModifyWriteAtomicInstruction(
                   100, 70, static_cast<uint32_t>(SpvOpAtomicIAdd), 41, 17, 0,
                   41, 0, MakeInstructionDescriptor(44, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // This transformation is not responsible for AtomicLoad or AtomicStore.
  ASSERT_FALSE(TransformationAddReadModifyWriteAtomicInstruction(
                   100, 40, static_cast<uint32_t>(SpvOpAtomicLoad), 41, 17, 0,
                   0, 0, MakeInstructionDescriptor(44, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: id 100 of memory scope instruction does not exist.
  ASSERT_FALSE(TransformationAddReadModifyWriteAtomicInstruction(
                   100, 40, static_cast<uint32_t>(SpvOpAtomicIAdd), 100, 17, 0,
                   41, 0, MakeInstructionDescriptor(44, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: memory scope should be |OpConstant| opcode.
  ASSERT_FALSE(TransformationAddReadModifyWriteAtomicInstruction(
                   100, 40, static_cast<uint32_t>(SpvOpAtomicIAdd), 37, 17, 0,
                   41, 0, MakeInstructionDescriptor(44, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: memory scope not SpvScopeInvocation.
  ASSERT_FALSE(TransformationAddReadModifyWriteAtomicInstruction(
                   100, 40, static_cast<uint32_t>(SpvOpAtomicIAdd), 17, 17, 0,
                   41, 0, MakeInstructionDescriptor(44, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: id 100 of memory semantics instruction does not exist.
  ASSERT_FALSE(TransformationAddReadModifyWriteAtomicInstruction(
                   100, 40, static_cast<uint32_t>(SpvOpAtomicIAdd), 41, 100, 0,
                   41, 0, MakeInstructionDescriptor(44, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: Can't insert OpLoad before the id 41 of memory scope.
  ASSERT_FALSE(TransformationAddReadModifyWriteAtomicInstruction(
                   100, 40, static_cast<uint32_t>(SpvOpAtomicIAdd), 41, 17, 0,
                   41, 0, MakeInstructionDescriptor(41, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Bad: Can't insert OpLoad before the id 41 of memory semantics.
  ASSERT_FALSE(TransformationAddReadModifyWriteAtomicInstruction(
                   100, 40, static_cast<uint32_t>(SpvOpAtomicIAdd), 41, 17, 0,
                   41, 0, MakeInstructionDescriptor(17, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddReadModifyWriteAtomicInstructionTest, IsApplicable) {
  const std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main" %16
               OpExecutionMode %4 LocalSize 1 1 1
               OpSource GLSL 450
               OpName %4 "main"
               OpName %8 "result"
               OpName %10 "temp_val"
               OpName %13 "id"
               OpName %16 "gl_LocalInvocationID"
               OpName %21 "chunkStart"
               OpName %23 "i"
               OpName %37 "shared_memory"
               OpDecorate %16 BuiltIn LocalInvocationId
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %11 = OpTypeInt 32 0
         %12 = OpTypePointer Function %11
         %14 = OpTypeVector %11 3
         %15 = OpTypePointer Input %14
         %16 = OpVariable %15 Input
         %17 = OpConstant %11 272 ; SequentiallyConsistent | WorkgroupMemory
         %62 = OpConstant %11 256 ; None                   | WorkgroupMemory
         %18 = OpTypePointer Input %11
         %32 = OpTypeBool
         %34 = OpConstant %11 16
         %35 = OpTypeArray %11 %34
         %36 = OpTypePointer Workgroup %35
         %37 = OpVariable %36 Workgroup
         %39 = OpTypePointer Workgroup %11
         %41 = OpConstant %11 4
         %43 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %13 = OpVariable %12 Function
         %21 = OpVariable %12 Function
         %23 = OpVariable %12 Function
               OpStore %8 %9
               OpStore %10 %9
         %19 = OpAccessChain %18 %16 %17
         %20 = OpLoad %11 %19
               OpStore %13 %20
         %22 = OpLoad %11 %13
               OpStore %21 %22
         %24 = OpLoad %11 %21
               OpStore %23 %24
               OpBranch %25
         %25 = OpLabel
               OpLoopMerge %27 %28 None
               OpBranch %29
         %29 = OpLabel
         %30 = OpLoad %11 %23
         %31 = OpLoad %11 %21
         %33 = OpULessThan %32 %30 %31
               OpBranchConditional %33 %26 %27
         %26 = OpLabel
         %38 = OpLoad %11 %23
         %40 = OpAccessChain %39 %37 %38
         %44 = OpLoad %6 %10
         %45 = OpIAdd %6 %44 %43
               OpStore %10 %45
               OpBranch %28
         %28 = OpLabel
         %46 = OpLoad %11 %23
         %47 = OpIAdd %11 %46 %43
               OpStore %23 %47
               OpBranch %25
         %27 = OpLabel
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

  transformation_context.GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
      40);
  // Successful transformations.
  {
    // Added |OpAtomicIAdd| instruction, takes |value_id| only.
    TransformationAddReadModifyWriteAtomicInstruction transformation(
        42, 40, static_cast<uint32_t>(SpvOpAtomicIAdd), 41, 17, 0, 41, 0,
        MakeInstructionDescriptor(44, SpvOpLoad, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }
  {
    // Added |OpAtomicIIncrement| instruction, don't need any extra ids.
    TransformationAddReadModifyWriteAtomicInstruction transformation(
        60, 40, static_cast<uint32_t>(SpvOpAtomicIIncrement), 41, 17, 0, 0, 0,
        MakeInstructionDescriptor(44, SpvOpLoad, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }
  {
    // Added |OpAtomicCompareExchange| instruction, that takes
    // 'memory_semantics_unequal', |value_id|, |comparator_id|.
    TransformationAddReadModifyWriteAtomicInstruction transformation(
        61, 40, static_cast<uint32_t>(SpvOpAtomicCompareExchange), 41, 17, 62,
        41, 17, MakeInstructionDescriptor(44, SpvOpLoad, 0));
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    ApplyAndCheckFreshIds(transformation, context.get(),
                          &transformation_context);
    ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(
        context.get(), validator_options, kConsoleMessageConsumer));
  }

  const std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main" %16
               OpExecutionMode %4 LocalSize 1 1 1
               OpSource GLSL 450
               OpName %4 "main"
               OpName %8 "result"
               OpName %10 "temp_val"
               OpName %13 "id"
               OpName %16 "gl_LocalInvocationID"
               OpName %21 "chunkStart"
               OpName %23 "i"
               OpName %37 "shared_memory"
               OpDecorate %16 BuiltIn LocalInvocationId
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %11 = OpTypeInt 32 0
         %12 = OpTypePointer Function %11
         %14 = OpTypeVector %11 3
         %15 = OpTypePointer Input %14
         %16 = OpVariable %15 Input
         %17 = OpConstant %11 272 ; SequentiallyConsistent | WorkgroupMemory
         %62 = OpConstant %11 256 ; None                   | WorkgroupMemory
         %18 = OpTypePointer Input %11
         %32 = OpTypeBool
         %34 = OpConstant %11 16
         %35 = OpTypeArray %11 %34
         %36 = OpTypePointer Workgroup %35
         %37 = OpVariable %36 Workgroup
         %39 = OpTypePointer Workgroup %11
         %41 = OpConstant %11 4
         %43 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %13 = OpVariable %12 Function
         %21 = OpVariable %12 Function
         %23 = OpVariable %12 Function
               OpStore %8 %9
               OpStore %10 %9
         %19 = OpAccessChain %18 %16 %17
         %20 = OpLoad %11 %19
               OpStore %13 %20
         %22 = OpLoad %11 %13
               OpStore %21 %22
         %24 = OpLoad %11 %21
               OpStore %23 %24
               OpBranch %25
         %25 = OpLabel
               OpLoopMerge %27 %28 None
               OpBranch %29
         %29 = OpLabel
         %30 = OpLoad %11 %23
         %31 = OpLoad %11 %21
         %33 = OpULessThan %32 %30 %31
               OpBranchConditional %33 %26 %27
         %26 = OpLabel
         %38 = OpLoad %11 %23
         %40 = OpAccessChain %39 %37 %38
         %42 = OpAtomicIAdd %11 %40 %41 %17 %41
         %60 = OpAtomicIIncrement %11 %40 %41 %17
         %61 = OpAtomicCompareExchange %11 %40 %41 %17 %62 %41 %17
         %44 = OpLoad %6 %10
         %45 = OpIAdd %6 %44 %43
               OpStore %10 %45
               OpBranch %28
         %28 = OpLabel
         %46 = OpLoad %11 %23
         %47 = OpIAdd %11 %46 %43
               OpStore %23 %47
               OpBranch %25
         %27 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
