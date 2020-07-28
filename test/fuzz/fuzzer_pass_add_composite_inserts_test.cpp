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
               OpName %9 "v"
               OpName %18 "m"
               OpDecorate %9 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeVector %6 3
          %8 = OpTypePointer Function %7
         %10 = OpConstant %6 2
         %11 = OpConstant %6 3
         %12 = OpConstant %6 4
         %13 = OpConstantComposite %7 %10 %11 %12
         %14 = OpTypeFloat 32
         %15 = OpTypeVector %14 3
         %16 = OpTypeMatrix %15 3
         %17 = OpTypePointer Function %16
         %19 = OpConstant %14 1.10000002
         %20 = OpConstant %14 2.0999999
         %21 = OpConstant %14 3.0999999
         %22 = OpConstantComposite %15 %19 %20 %21
         %23 = OpConstant %14 1.20000005
         %24 = OpConstant %14 2.20000005
         %25 = OpConstant %14 3.20000005
         %26 = OpConstantComposite %15 %23 %24 %25
         %27 = OpConstant %14 1.29999995
         %28 = OpConstant %14 2.29999995
         %29 = OpConstant %14 3.29999995
         %30 = OpConstantComposite %15 %27 %28 %29
         %31 = OpConstantComposite %16 %22 %26 %30
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %9 = OpVariable %8 Function
         %18 = OpVariable %17 Function
               OpStore %9 %13
               OpStore %18 %31
         %32 = OpCompositeInsert %16 %22 %31 0
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
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools