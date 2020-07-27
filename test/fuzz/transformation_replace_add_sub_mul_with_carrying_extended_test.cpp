// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_replace_add_sub_mul_with_carrying_extended.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationReplaceAddSubMulWithCarryingExtendedTest, BasicScenarios) {
  // This is a simple transformation and this test handles the main cases.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %7 "int2"
               OpMemberName %7 0 "a"
               OpMemberName %7 1 "b"
               OpName %9 "test"
               OpName %14 "a"
               OpName %15 "b"
               OpName %16 "c"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 0
          %7 = OpTypeStruct %6 %6
          %8 = OpTypePointer Function %7
         %10 = OpConstant %6 2
         %11 = OpConstant %6 3
         %12 = OpConstantComposite %7 %10 %11
         %13 = OpTypePointer Function %6
         %17 = OpTypeInt 32 1
         %18 = OpConstant %17 0
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %9 = OpVariable %8 Function
         %14 = OpVariable %13 Function
         %15 = OpVariable %13 Function
         %16 = OpVariable %13 Function
               OpStore %9 %12
               OpStore %14 %10
               OpStore %15 %11
         %19 = OpAccessChain %13 %9 %18
         %20 = OpLoad %6 %19
               OpStore %16 %20
         %21 = OpLoad %6 %14
         %22 = OpLoad %6 %15
         %23 = OpIAdd %6 %21 %22
               OpStore %16 %23
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
  auto transformation_valid_1 =
      TransformationReplaceAddSubMulWithCarryingExtended(30, 7, 23);

  ASSERT_TRUE(transformation_valid_1.IsApplicable(context.get(),
                                                  transformation_context));
  transformation_valid_1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::vector<uint32_t> actual_binary;
  context.get()->module()->ToBinary(&actual_binary, false);
  SpirvTools t(env);
  std::string actual_disassembled;
  t.Disassemble(actual_binary, &actual_disassembled, kFuzzDisassembleOption);
  std::cout << actual_disassembled;
  ASSERT_TRUE(IsValid(env, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools