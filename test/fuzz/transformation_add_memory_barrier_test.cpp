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

#include "source/fuzz/transformation_add_memory_barrier.h"

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddMemoryBarrierTest, OpMemoryBarrierTestCase) {
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
         %15 = OpConstant %6 4 ; Scope Invocation
          %7 = OpConstant %6 2
         %28 = OpConstant %6 64 ; None | UniformMemory
         %20 = OpTypePointer Function %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %21 = OpVariable %20 Function %7
          %9 = OpFunctionCall %2 %12
          %8 = OpCopyObject %6 %7
         %22 = OpLoad %6 %21
         %23 = OpFunctionCall %2 %12
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

  // Successful transformations.
  {
    TransformationAddMemoryBarrier transformation(
        15, 28, MakeInstructionDescriptor(24, SpvOpFunctionCall, 0));
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
         %15 = OpConstant %6 4 ; Scope Invocation
          %7 = OpConstant %6 2
         %28 = OpConstant %6 64 ; None | UniformMemory
         %20 = OpTypePointer Function %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %21 = OpVariable %20 Function %7
          %9 = OpFunctionCall %2 %12
          %8 = OpCopyObject %6 %7
         %22 = OpLoad %6 %21
         %23 = OpFunctionCall %2 %12
               OpMemoryBarrier %15 %28
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
