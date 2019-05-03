// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/fuzzer_pass_add_useful_constructs.h"
#include "source/fuzz/pseudo_random_generator.h"

#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(FuzzerPassAddUsefulConstructsTest, CheckBasicStuffIsAdded) {
  // The SPIR-V came from the following empty GLSL shader:
  //
  // #version 450
  //
  // void main()
  // {
  // }

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  FuzzerContext fuzzer_context(MakeUnique<PseudoRandomGenerator>(0).get(), 100);
  protobufs::TransformationSequence transformation_sequence;

  FuzzerPassAddUsefulConstructs pass;
  pass.Apply(context.get(), &fact_manager, &fuzzer_context,
             &transformation_sequence);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
        %100 = OpTypeBool
        %101 = OpTypeInt 32 1
        %102 = OpTypeInt 32 0
        %103 = OpTypeFloat 32
        %104 = OpConstantTrue %100
        %105 = OpConstantFalse %100
        %106 = OpConstant %101 0
        %107 = OpConstant %101 1
        %108 = OpConstant %102 0
        %109 = OpConstant %102 1
        %110 = OpConstant %103 0
        %111 = OpConstant %103 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
