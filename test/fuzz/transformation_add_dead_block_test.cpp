// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_add_dead_block.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddDeadBlockTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpConstantTrue %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  // Id 4 is already in use
  ASSERT_FALSE(TransformationAddDeadBlock(4, 5, true, {}).IsApplicable(context.get(), fact_manager));

  // Id 7 is not a block
  ASSERT_FALSE(TransformationAddDeadBlock(100, 7, true, {}).IsApplicable(context.get(), fact_manager));

  TransformationAddDeadBlock transformation(100, 5, true, {});
  ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
  transformation.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpConstantTrue %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpSelectionMerge %8
               OpBranchConditional %7 %8 %100
        %100 = OpLabel
               OpBranch %8
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

// Target block must not be merge or continue

// Source block must not be loop head

// Target block can start with OpPhi; need to give suitable ids in that case

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
