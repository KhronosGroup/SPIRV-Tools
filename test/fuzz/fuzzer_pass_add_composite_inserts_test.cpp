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

#include "source/fuzz/fuzzer_pass_add_composite_inserts.h"
#include "source/fuzz/pseudo_random_generator.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {
TEST(FuzzerPassAddCompositeInsertsTest, BasicScenarios) {
  std::string shader = R"(
                OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %7 "base"
               OpMemberName %7 0 "a1"
               OpMemberName %7 1 "a2"
               OpName %9 "A"
               OpName %13 "level_1"
               OpMemberName %13 0 "b1"
               OpMemberName %13 1 "b2"
               OpName %15 "B"
               OpName %19 "level_2"
               OpMemberName %19 0 "c1"
               OpMemberName %19 1 "c2"
               OpName %21 "C"
               OpMemberDecorate %7 0 RelaxedPrecision
               OpMemberDecorate %7 1 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeStruct %6 %6
          %8 = OpTypePointer Function %7
         %10 = OpConstant %6 1
         %11 = OpConstant %6 2
         %12 = OpConstantComposite %7 %10 %11
         %13 = OpTypeStruct %7 %7
         %14 = OpTypePointer Function %13
         %19 = OpTypeStruct %13 %13
         %20 = OpTypePointer Function %19
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %9 = OpVariable %8 Function
         %15 = OpVariable %14 Function
         %21 = OpVariable %20 Function
               OpStore %9 %12
         %16 = OpLoad %7 %9
         %17 = OpLoad %7 %9
         %18 = OpCompositeConstruct %13 %16 %17
               OpStore %15 %18
         %22 = OpLoad %13 %15
         %23 = OpLoad %13 %15
         %24 = OpCompositeConstruct %19 %22 %23
               OpStore %21 %24
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);
  auto prng = MakeUnique<PseudoRandomGenerator>(9);
  FuzzerContext fuzzer_context(prng.get(), 100);
  protobufs::TransformationSequence transformation_sequence;
  FuzzerPassAddCompositeInserts fuzzer_pass(
      context.get(), &transformation_context, &fuzzer_context,
      &transformation_sequence);
  fuzzer_pass.Apply();

  std::vector<uint32_t> actual_binary;
  context.get()->module()->ToBinary(&actual_binary, false);
  SpirvTools t(env);
  std::string actual_disassembled;
  t.Disassemble(actual_binary, &actual_disassembled, kFuzzDisassembleOption);
  std::cout << actual_disassembled;
  /*
  std::string after_transformation = R"(

  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
   */
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools