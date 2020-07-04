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

#include "source/fuzz/fuzzer_pass_interchange_zero_like_constants.h"

#include "source/fuzz/pseudo_random_generator.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

// Test that the shader is still valid after applying the transformations.
TEST(FuzzerPassInterchangeZeroLikeConstants, StillValid) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %29
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %12 "c"
               OpName %16 "d"
               OpName %20 "e"
               OpName %29 "color"
               OpDecorate %16 RelaxedPrecision
               OpDecorate %29 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %11 = OpConstant %6 1
         %13 = OpConstant %6 2
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %17 = OpConstant %14 0
         %18 = OpTypeBool
         %19 = OpTypePointer Function %18
         %21 = OpConstantFalse %18
         %27 = OpTypeVector %6 4
         %28 = OpTypePointer Output %27
         %29 = OpVariable %28 Output
         %30 = OpConstantComposite %27 %9 %11 %9 %11
         %32 = OpConstantComposite %27 %11 %9 %9 %11
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
         %16 = OpVariable %15 Function
         %20 = OpVariable %19 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %13
               OpStore %16 %17
               OpStore %20 %21
         %22 = OpLoad %6 %8
         %23 = OpLoad %6 %10
         %24 = OpFOrdEqual %18 %22 %23
               OpSelectionMerge %26 None
               OpBranchConditional %24 %25 %31
         %25 = OpLabel
               OpStore %29 %30
               OpBranch %26
         %31 = OpLabel
               OpStore %29 %32
               OpBranch %26
         %26 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;

  auto generator = MakeUnique<PseudoRandomGenerator>(0);

  for (int i = 0; i < 10; i++) {
    const auto context =
        BuildModule(env, consumer, shader, kFuzzAssembleOption);
    ASSERT_TRUE(IsValid(env, context.get()));

    FactManager fact_manager;
    spvtools::ValidatorOptions validator_options;
    TransformationContext transformation_context(&fact_manager,
                                                 validator_options);

    FuzzerContext fuzzer_context(generator.get(), 33);
    protobufs::TransformationSequence transformation_sequence;

    FuzzerPassInterchangeZeroLikeConstants fuzzer_pass(
        context.get(), &transformation_context, &fuzzer_context,
        &transformation_sequence);

    fuzzer_pass.Apply();

    // Check that the transformed shader is still valid
    ASSERT_TRUE(IsValid(env, context.get()));
  }
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools