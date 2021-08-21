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

#include "source/fuzz/transformation_changing_memory_scope.h"

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationChangingMemoryScopeTest, AtomicInstructionsTestCases) {
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
         %15 = OpConstant %6 4 ; Scope Invocation
         %19 = OpConstant %6 2 ; Scope Workgroup
         %16 = OpConstant %6 7
         %17 = OpConstant %7 4
         %25 = OpConstant %6 66
         %28 = OpConstant %6 80
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %22 = OpAtomicLoad %6 %14 %15 %25
         %32 = OpAtomicExchange %6 %14 %15 %28 %16
         %33 = OpAtomicCompareExchange %6 %14 %15 %28 %28 %16 %15
         %24 = OpAccessChain %13 %11 %12
               OpAtomicStore %14 %15 %28 %16
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
    // OpAtomicLoad Invocation
    // to:
    // OpAtomicLoad Workgroup
    TransformationChangingMemoryScope transformation(
        MakeInstructionDescriptor(22, SpvOpAtomicLoad, 0), 19);
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
         %15 = OpConstant %6 4 ; Scope Invocation
         %19 = OpConstant %6 2 ; Scope Workgroup
         %16 = OpConstant %6 7
         %17 = OpConstant %7 4
         %25 = OpConstant %6 66
         %28 = OpConstant %6 80
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %22 = OpAtomicLoad %6 %14 %19 %25
         %32 = OpAtomicExchange %6 %14 %15 %28 %16
         %33 = OpAtomicCompareExchange %6 %14 %15 %28 %28 %16 %15
         %24 = OpAccessChain %13 %11 %12
               OpAtomicStore %14 %15 %28 %16
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
