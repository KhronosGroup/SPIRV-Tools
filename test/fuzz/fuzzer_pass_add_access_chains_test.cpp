// Copyright (c) 2026 The Khronos Group Inc.
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

#include "source/fuzz/fuzzer_pass_add_access_chains.h"

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/random_generator.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

class AlwaysZeroGenerator : public RandomGenerator {
 public:
  uint32_t RandomUint32(uint32_t /*bound*/) override { return 0; }

  uint64_t RandomUint64(uint64_t /*bound*/) override { return 0; }

  uint32_t RandomPercentage() override { return 0; }

  bool RandomBool() override { return false; }

  double RandomDouble() override { return 0.0; }
};

TEST(FuzzerPassAddAccessChainsTest, RuntimeArray) {
  const std::string shader = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %runtime_array ArrayStride 4
               OpMemberDecorate %buffer_type 0 Offset 0
               OpDecorate %buffer_type Block
               OpDecorate %buffer DescriptorSet 0
               OpDecorate %buffer Binding 0
          %uint = OpTypeInt 32 0
 %runtime_array = OpTypeRuntimeArray %uint
   %buffer_type = OpTypeStruct %runtime_array
%buffer_pointer = OpTypePointer StorageBuffer %buffer_type
          %void = OpTypeVoid
 %function_type = OpTypeFunction %void
        %buffer = OpVariable %buffer_pointer StorageBuffer
          %main = OpFunction %void None %function_type
         %entry = OpLabel
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
  protobufs::TransformationSequence transformation_sequence;
  FuzzerContext fuzzer_context(MakeUnique<AlwaysZeroGenerator>(), 100, false);

  FuzzerPassAddAccessChains fuzzer_pass(context.get(), &transformation_context,
                                        &fuzzer_context,
                                        &transformation_sequence, false);
  fuzzer_pass.Apply();

  ASSERT_GT(transformation_sequence.transformation_size(), 0);
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
