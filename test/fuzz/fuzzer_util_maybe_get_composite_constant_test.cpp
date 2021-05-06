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

TEST(FuzzerUtilMaybeGetCompositeConstantTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %54
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "b1"
               OpName %10 "b2"
               OpName %12 "b3"
               OpName %13 "b4"
               OpName %16 "f1"
               OpName %18 "f2"
               OpName %22 "zc"
               OpName %24 "i1"
               OpName %28 "i2"
               OpName %30 "i3"
               OpName %32 "i4"
               OpName %37 "f_arr"
               OpName %47 "i_arr"
               OpName %54 "value"
               OpDecorate %22 RelaxedPrecision
               OpDecorate %24 RelaxedPrecision
               OpDecorate %28 RelaxedPrecision
               OpDecorate %30 RelaxedPrecision
               OpDecorate %32 RelaxedPrecision
               OpDecorate %47 RelaxedPrecision
               OpDecorate %54 Location 0
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
         %23 = OpConstant %20 0
         %25 = OpConstant %20 1
         %26 = OpTypeInt 32 0
         %27 = OpTypePointer Function %26
         %29 = OpConstant %26 100
         %31 = OpConstant %20 -1
         %33 = OpConstant %20 -99
         %34 = OpConstant %26 5
         %35 = OpTypeArray %14 %34
         %36 = OpTypePointer Function %35
         %38 = OpConstant %14 5.5
         %39 = OpConstant %14 4.4000001
         %40 = OpConstant %14 3.29999995
         %41 = OpConstant %14 2.20000005
         %42 = OpConstant %14 1.10000002
         %43 = OpConstantComposite %35 %38 %39 %40 %41 %42
         %44 = OpConstant %26 3
         %45 = OpTypeArray %20 %44
         %46 = OpTypePointer Function %45
         %48 = OpConstant %20 3
         %49 = OpConstant %20 7
         %50 = OpConstant %20 9
         %51 = OpConstantComposite %45 %48 %49 %50
         %53 = OpTypePointer Input %14
         %54 = OpVariable %53 Input
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
         %28 = OpVariable %27 Function
         %30 = OpVariable %21 Function
         %32 = OpVariable %21 Function
         %37 = OpVariable %36 Function
         %47 = OpVariable %46 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %12 %9
               OpStore %13 %11
               OpStore %16 %17
               OpStore %18 %19
               OpStore %22 %23
               OpStore %24 %25
               OpStore %28 %29
               OpStore %30 %31
               OpStore %32 %33
               OpStore %37 %43
               OpStore %47 %51
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

  //      %43 = OpConstantComposite %35 %38 %39 %40 %41 %42
  //    %51 = OpConstantComposite %45 %48 %49 %50
  // This should pass as a float array with 5 elements exist and its id is 43.
  ASSERT_EQ(43, fuzzerutil::MaybeGetCompositeConstant(
                    ir_context, transformation_context, {38, 39, 40, 41, 42},
                    35, false));
  // This should pass as an int array with 3 elements exist and its id is 51.
  ASSERT_EQ(51,
            fuzzerutil::MaybeGetCompositeConstant(
                ir_context, transformation_context, {48, 49, 50}, 45, false));
  // An int array with 2 elements does not exist.
  ASSERT_EQ(0, fuzzerutil::MaybeGetCompositeConstant(
                   ir_context, transformation_context, {48, 49}, 45, false));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools