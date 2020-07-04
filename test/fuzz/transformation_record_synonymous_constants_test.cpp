// Copyright (c) 2020 Stefano Milizia
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

#include "source/fuzz/transformation_record_synonymous_constants.h"

#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationRecordSynonymousConstantsTest, IntConstants) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %17
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %12 "c"
               OpName %17 "color"
               OpDecorate %8 RelaxedPrecision
               OpDecorate %10 RelaxedPrecision
               OpDecorate %12 RelaxedPrecision
               OpDecorate %17 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %11 = OpConstantNull %6
         %13 = OpConstant %6 1
         %14 = OpTypeFloat 32
         %15 = OpTypeVector %14 4
         %16 = OpTypePointer Output %15
         %17 = OpVariable %16 Output
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %13
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);
  ASSERT_TRUE(IsValid(env, context.get()));

  // %3 is not a constant declaration
  ASSERT_FALSE(TransformationRecordSynonymousConstants(3, 9).IsApplicable(
      context.get(), transformation_context));

  ASSERT_FALSE(TransformationRecordSynonymousConstants(9, 3).IsApplicable(
      context.get(), transformation_context));

  // The two constants must be different
  ASSERT_FALSE(TransformationRecordSynonymousConstants(9, 9).IsApplicable(
      context.get(), transformation_context));

  // %9 and %13 are not equivalent
  ASSERT_FALSE(TransformationRecordSynonymousConstants(9, 13).IsApplicable(
      context.get(), transformation_context));

  ASSERT_FALSE(TransformationRecordSynonymousConstants(13, 9).IsApplicable(
      context.get(), transformation_context));

  // %11 and %13 are not equivalent
  ASSERT_FALSE(TransformationRecordSynonymousConstants(11, 13).IsApplicable(
      context.get(), transformation_context));

  ASSERT_FALSE(TransformationRecordSynonymousConstants(13, 11).IsApplicable(
      context.get(), transformation_context));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
