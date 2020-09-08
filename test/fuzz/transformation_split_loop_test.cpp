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

#include "source/fuzz/transformation_split_loop.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationSplitLoopTest, BasicScenarios) {
  // This is a simple transformation and this test handles the main cases.
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "s"
               OpName %10 "i"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %17 = OpConstant %6 10
         %18 = OpTypeBool
         %55 = OpConstantTrue %18
         %56 = OpConstantFalse %18
         %53 = OpTypePointer Function %18
         %24 = OpConstant %6 3
         %30 = OpConstant %6 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %50 = OpVariable %7 Function
         %54 = OpVariable %53 Function
               OpStore %8 %9
               OpStore %10 %9
               OpBranch %11
         %11 = OpLabel
               OpLoopMerge %13 %14 None
               OpBranch %15
         %15 = OpLabel
         %16 = OpLoad %6 %10
         %19 = OpSLessThan %18 %16 %17
               OpBranchConditional %19 %12 %13
         %12 = OpLabel
         %20 = OpLoad %6 %10
         %21 = OpLoad %6 %8
         %22 = OpIAdd %6 %21 %20
               OpStore %8 %22
         %23 = OpLoad %6 %10
         %25 = OpIEqual %18 %23 %24
               OpSelectionMerge %27 None
               OpBranchConditional %25 %26 %27
         %26 = OpLabel
               OpBranch %13
         %27 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %29 = OpLoad %6 %10
         %31 = OpIAdd %6 %29 %30
               OpStore %10 %31
               OpBranch %11
         %13 = OpLabel
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

  auto transformation = TransformationSplitLoop(11, 50, 54, 30, 9, 24, 101, 102,
                                                103, 104, 55, 56, 105, 106,
                                                {{11, 201},
                                                 {15, 202},
                                                 {12, 203},
                                                 {13, 204},
                                                 {26, 205},
                                                 {27, 206},
                                                 {14, 207}},
                                                {{16, 301},
                                                 {19, 302},
                                                 {20, 303},
                                                 {21, 304},
                                                 {22, 305},
                                                 {23, 306},
                                                 {25, 307},
                                                 {29, 308},
                                                 {31, 309}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
  transformation.Apply(context.get(), &transformation_context);

  std::vector<uint32_t> actual_binary;
  context.get()->module()->ToBinary(&actual_binary, false);
  SpirvTools t(env);
  std::string actual_disassembled;
  t.Disassemble(actual_binary, &actual_disassembled, kFuzzDisassembleOption);
  std::cout << actual_disassembled << std::endl;

  ASSERT_TRUE(IsValid(env, context.get()));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
