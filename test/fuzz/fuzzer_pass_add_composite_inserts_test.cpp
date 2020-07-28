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
  // This is a simple fuzzer pass and this test handles the main cases.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "m1"
               OpName %18 "m2"
               OpName %26 "m3"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 3
          %8 = OpTypeMatrix %7 3
          %9 = OpTypePointer Function %8
         %11 = OpConstant %6 1
         %12 = OpConstantComposite %7 %11 %11 %11
         %13 = OpConstant %6 2
         %14 = OpConstantComposite %7 %13 %13 %13
         %15 = OpConstant %6 3
         %16 = OpConstantComposite %7 %15 %15 %15
         %17 = OpConstantComposite %8 %12 %14 %16
         %19 = OpConstant %6 4
         %20 = OpConstantComposite %7 %19 %19 %19
         %21 = OpConstant %6 5
         %22 = OpConstantComposite %7 %21 %21 %21
         %23 = OpConstant %6 6
         %24 = OpConstantComposite %7 %23 %23 %23
         %25 = OpConstantComposite %8 %20 %22 %24
         %27 = OpConstant %6 7
         %28 = OpConstantComposite %7 %27 %27 %27
         %29 = OpConstant %6 8
         %30 = OpConstantComposite %7 %29 %29 %29
         %31 = OpConstant %6 9
         %32 = OpConstantComposite %7 %31 %31 %31
         %33 = OpConstantComposite %8 %28 %30 %32
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %10 = OpVariable %9 Function
         %18 = OpVariable %9 Function
         %26 = OpVariable %9 Function
               OpStore %10 %17
               OpStore %18 %25
               OpStore %26 %33
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
  auto prng = MakeUnique<PseudoRandomGenerator>(12);
  FuzzerContext fuzzer_context(prng.get(), 100);
  protobufs::TransformationSequence transformation_sequence;
  FuzzerPassAddCompositeInserts fuzzer_pass(
      context.get(), &transformation_context, &fuzzer_context,
      &transformation_sequence);
  fuzzer_pass.Apply();
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools