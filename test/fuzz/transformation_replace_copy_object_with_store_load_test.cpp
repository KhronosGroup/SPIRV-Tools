// Copyright (c) 2020 Google
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

#include "source/fuzz/transformation_replace_copy_object_with_store_load.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationReplaceCopyObjectWithStoreLoad, BasicScenarios) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %11
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "a"
               OpName %11 "b"
               OpDecorate %8 RelaxedPrecision
               OpDecorate %11 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %10 = OpTypePointer Private %6
         %11 = OpVariable %10 Private
         %12 = OpConstant %6 3
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %11 %12
        %13 = OpCopyObject %7 %8
        %14 = OpCopyObject %10 %11
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Invalid: fresh_variable_id=10 is not fresh.
  auto transformation_invalid_1 = TransformationReplaceCopyObjectWithStoreLoad(
      13, 10, SpvStorageClassFunction, 9);
  ASSERT_FALSE(transformation_invalid_1.IsApplicable(context.get(),
                                                     transformation_context));

  // Invalid: copy_object_result_id=10 is not a CopyObject instruction
  // result_id.
  auto transformation_invalid_2 = TransformationReplaceCopyObjectWithStoreLoad(
      10, 15, SpvStorageClassFunction, 9);
  ASSERT_FALSE(transformation_invalid_2.IsApplicable(context.get(),
                                                     transformation_context));

  // Invalid: initializer_id=8 is not a initializer result_id.
  auto transformation_invalid_3 = TransformationReplaceCopyObjectWithStoreLoad(
      13, 15, SpvStorageClassFunction, 8);
  ASSERT_FALSE(transformation_invalid_3.IsApplicable(context.get(),
                                                     transformation_context));
  // Invalid: SpvStorageClassUniform is not applicable to the transformation.
  auto transformation_invalid_4 = TransformationReplaceCopyObjectWithStoreLoad(
      13, 15, SpvStorageClassUniform, 9);
  ASSERT_FALSE(transformation_invalid_4.IsApplicable(context.get(),
                                                     transformation_context));

  auto transformation_valid_1 = TransformationReplaceCopyObjectWithStoreLoad(
      13, 15, SpvStorageClassFunction, 9);
  ASSERT_TRUE(transformation_valid_1.IsApplicable(context.get(),
                                                  transformation_context));
  transformation_valid_1.Apply(context.get(), &transformation_context);

  auto transformation_valid_2 = TransformationReplaceCopyObjectWithStoreLoad(
      14, 16, SpvStorageClassPrivate, 12);
  ASSERT_TRUE(transformation_valid_2.IsApplicable(context.get(),
                                                  transformation_context));
  transformation_valid_2.Apply(context.get(), &transformation_context);

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %11 %16
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "a"
               OpName %11 "b"
               OpDecorate %8 RelaxedPrecision
               OpDecorate %11 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 2
         %10 = OpTypePointer Private %6
         %11 = OpVariable %10 Private
         %12 = OpConstant %6 3
         %16 = OpVariable %10 Private %12
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %15 = OpVariable %7 Function %9
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %11 %12
               OpStore %15 %8
         %13 = OpLoad %7 %15
               OpStore %16 %11
         %14 = OpLoad %10 %16
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
