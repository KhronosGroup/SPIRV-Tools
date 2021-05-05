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

TEST(FuzzerUtilMaybeGetIntegerConstantFromValueAndTypeTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %32
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "b1"
               OpName %10 "b2"
               OpName %12 "b3"
               OpName %13 "b4"
               OpName %16 "f1"
               OpName %18 "f2"
               OpName %22 "i1"
               OpName %24 "i2"
               OpName %26 "i3"
               OpName %28 "i4"
               OpName %32 "value"
               OpDecorate %22 RelaxedPrecision
               OpDecorate %24 RelaxedPrecision
               OpDecorate %26 RelaxedPrecision
               OpDecorate %28 RelaxedPrecision
               OpDecorate %32 Location 0
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
         %20 = OpTypeInt 32 1
         %21 = OpTypePointer Function %20
         %23 = OpConstant %20 1
         %25 = OpConstant %20 100
         %27 = OpConstant %20 -1
         %29 = OpConstant %20 -99
         %31 = OpTypePointer Input %14
         %32 = OpVariable %31 Input
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %12 = OpVariable %7 Function
         %13 = OpVariable %7 Function
         %16 = OpVariable %15 Function
         %18 = OpVariable %15 Function
         %22 = OpVariable %21 Function
         %24 = OpVariable %21 Function
         %26 = OpVariable %21 Function
         %28 = OpVariable %21 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %9
               OpStore %13 %11
               OpStore %16 %17
               OpStore %18 %19
               OpStore %22 %23
               OpStore %24 %25
               OpStore %26 %27
               OpStore %28 %29
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
  // All integer constants:
  //  %23 = OpConstant %20 1
  //  %25 = OpConstant %20 100
  //  %27 = OpConstant %20 -1
  //  %29 = OpConstant %20 -99
  // A 32 bit unsigned int constant with value 1 and type_id 20 exists and the
  // id is 23.
  ASSERT_TRUE(fuzzerutil::MaybeGetIntegerConstantFromValueAndType(ir_context, 1,
                                                                  20) == 23);
  // A 32 bit signed int constant with value 100 exists and the id is 25.
  ASSERT_TRUE(fuzzerutil::MaybeGetIntegerConstantFromValueAndType(
                  ir_context, 100, 20) == 25);
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools