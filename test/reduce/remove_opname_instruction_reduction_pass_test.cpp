// Copyright (c) 2018 Google LLC
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

#include "reduce_test_util.h"

#include "source/opt/build_module.h"
#include "source/reduce/reduction_opportunity.h"
#include "source/reduce/remove_opname_instruction_reduction_pass.h"

namespace spvtools {
namespace reduce {
namespace {

TEST(RemoveOpnameInstructionReductionPassTest, NothingToRemove) {
  const std::string original = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, original, kReduceAssembleOption);
  const auto pass =
      TestSubclass<RemoveOpNameInstructionReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(0, ops.size());
}

TEST(RemoveOpnameInstructionReductionPassTest, RemoveSingleOpName) {
  const std::string prologue = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
  )";

  const std::string epilogue = R"(
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const std::string original = prologue + R"(
               OpName %4 "main"
  )" + epilogue;

  const std::string expected = prologue + epilogue;

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, original, kReduceAssembleOption);
  const auto pass =
      TestSubclass<RemoveOpNameInstructionReductionPass>(env);
  const auto ops = pass.WrapGetAvailableOpportunities(context.get());
  ASSERT_EQ(1, ops.size());
  ASSERT_TRUE(ops[0]->PreconditionHolds());
  ops[0]->TryToApply();

  CheckEqual(env, expected, context.get());
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools
