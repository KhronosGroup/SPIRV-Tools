// Copyright (c) 2021 Shiyu Liu
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

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(FuzzerUtilMaybeGetBoolConstantTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %36
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "b1"
               OpName %10 "b2"
               OpName %12 "b3"
               OpName %13 "b4"
               OpName %16 "f1"
               OpName %18 "f2"
               OpName %20 "cf1"
               OpName %22 "cf2"
               OpName %26 "i1"
               OpName %28 "i2"
               OpName %30 "ci1"
               OpName %32 "ci2"
               OpName %36 "value"
               OpDecorate %26 RelaxedPrecision
               OpDecorate %28 RelaxedPrecision
               OpDecorate %30 RelaxedPrecision
               OpDecorate %32 RelaxedPrecision
               OpDecorate %36 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpTypePointer Function %6
          %9 = OpConstantTrue %6
         %11 = OpConstantFalse %6
         %14 = OpTypeFloat 32
         %15 = OpTypePointer Function %14
         %17 = OpConstant %14 1.23000002
         %19 = OpConstant %14 1.11000001
         %21 = OpConstant %14 2
         %23 = OpConstant %14 3.29999995
         %24 = OpTypeInt 32 1
         %25 = OpTypePointer Function %24
         %27 = OpConstant %24 1
         %29 = OpConstant %24 100
         %31 = OpConstant %24 123
         %33 = OpConstant %24 1111
         %35 = OpTypePointer Input %14
         %36 = OpVariable %35 Input
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
         %13 = OpVariable %7 Function
         %16 = OpVariable %15 Function
         %18 = OpVariable %15 Function
         %20 = OpVariable %15 Function
         %22 = OpVariable %15 Function
         %26 = OpVariable %25 Function
         %28 = OpVariable %25 Function
         %30 = OpVariable %25 Function
         %32 = OpVariable %25 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %9
               OpStore %13 %11
               OpStore %16 %17
               OpStore %18 %19
               OpStore %20 %21
               OpStore %22 %23
               OpStore %26 %27
               OpStore %28 %29
               OpStore %30 %31
               OpStore %32 %33
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const std::unique_ptr<opt::IRContext> context =
      BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  opt::IRContext* ir_context = context.get();
  // A bool constant with value false exists and the id is 11.
  ASSERT_EQ(11, fuzzerutil::MaybeGetBoolConstant(
                    ir_context, transformation_context, false, false));
  // A bool constant with value true exists and the id is 9.
  ASSERT_EQ(9, fuzzerutil::MaybeGetBoolConstant(
                   ir_context, transformation_context, true, false));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools