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

// Draft WIP - NOT FINAL.
TEST(TransformationAddReadModifyWriteAtomicInstructionTest, DISABLED_NotApplicable) {
  const std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 320
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1g
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
         %20 = OpConstant %9 80
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %24 = OpAccessChain %13 %11 %12
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
}

// Draft WIP - NOT FINAL.
TEST(TransformationAddReadModifyWriteAtomicInstructionTest, DISABLED_IsApplicable) {
  const std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 320
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1g
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
         %20 = OpConstant %9 80
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %24 = OpAccessChain %13 %11 %12
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

  const std::string after_transformation = R"(
                 OpCapability Shader
            %1 = OpExtInstImport "GLSL.std.450"
                 OpMemoryModel Logical GLSL450
                 OpEntryPoint Fragment %4 "main"
                 OpExecutionMode %4 OriginUpperLeft
                 OpSource ESSL 320
            %2 = OpTypeVoid
            %3 = OpTypeFunction %2
            %6 = OpTypeInt 32 1
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
           %20 = OpConstant %9 80
            %4 = OpFunction %2 None %3
            %5 = OpLabel
           %14 = OpAccessChain %13 %11 %12
           %25 = OpAtomicExchange %6 %14 %15 %20 %16
           %26 = OpAtomicCompareExchange %6 %14 %15 %20 %12 %16 %15
           %27 = OpAtomicIIncrement %6 %14 %15 %20
           %28 = OpAtomicIDecrement %6 %14 %15 %20
           %29 = OpAtomicIAdd %6  %14 %15 %20 %16
           %30 = OpAtomicISub %6  %14 %15 %20 %16
           %31 = OpAtomicSMin %6  %14 %15 %20 %16
           %32 = OpAtomicUMin %9 %90 %15 %20 %18
           %33 = OpAtomicSMax %6  %14 %15 %20 %15
           %34 = OpAtomicAnd  %6  %14 %15 %20 %16
           %35 = OpAtomicOr   %6  %14 %15 %20 %16
           %36 = OpAtomicXor  %6  %14 %15 %20 %16
           %24 = OpAccessChain %13 %11 %12
                 OpReturn
                 OpFunctionEnd
    )";
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
