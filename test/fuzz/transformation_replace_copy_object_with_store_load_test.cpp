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
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "b"
               OpName %11 "a"
               OpDecorate %8 RelaxedPrecision
               OpDecorate %11 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Private %6
          %8 = OpVariable %7 Private
          %9 = OpConstant %6 3
         %10 = OpTypePointer Function %6
         %12 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %11 = OpVariable %10 Function
               OpStore %8 %9
               OpStore %11 %12
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

  uint32_t copy_object_result_id = 10;
  uint32_t fresh_variable_id = 11;
  uint32_t variable_storage_class = SpvStorageClassFunction;
  uint32_t variable_initializer_id = 9;
  auto transformation = TransformationReplaceCopyObjectWithStoreLoad(
      copy_object_result_id, fresh_variable_id, variable_storage_class,
      variable_initializer_id);
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);
  std::vector<uint32_t> actual_binary;
  context.get()->module()->ToBinary(&actual_binary, false);
  SpirvTools t(env);
  std::string actual_disassembled;
  t.Disassemble(actual_binary, &actual_disassembled, kFuzzDisassembleOption);
  std::cout << actual_disassembled;
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
