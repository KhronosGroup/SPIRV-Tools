// Copyright (c) 2021 Shiyu Liu
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

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(FuzzerUtilMaybeFindBlockTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %8 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %10 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpBranch %11
         %11 = OpLabel
               OpStore %8 %9
               OpBranch %12
         %12 = OpLabel
               OpStore %8 %10
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const std::unique_ptr<opt::IRContext> context =
      BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  // Only blocks with id 11 and 12 can be found.
  // Should return nullptr when id is not a label or id was not found.
  uint32_t block_id1 = 11;
  uint32_t block_id2 = 12;
  uint32_t block_id3 = 13;
  uint32_t block_id4 = 8;

  opt::IRContext* ir_context = context.get();
  // Block with id 11 should be found.
  ASSERT_TRUE(fuzzerutil::MaybeFindBlock(ir_context, block_id1) != nullptr);
  // Block with id 12 should be found.
  ASSERT_TRUE(fuzzerutil::MaybeFindBlock(ir_context, block_id2) != nullptr);
  // Block with id 13 cannot be found.
  ASSERT_FALSE(fuzzerutil::MaybeFindBlock(ir_context, block_id3) != nullptr);
  // Block with id 8 exisits but don't not of type OpLabel.
  ASSERT_FALSE(fuzzerutil::MaybeFindBlock(ir_context, block_id4) != nullptr);
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
