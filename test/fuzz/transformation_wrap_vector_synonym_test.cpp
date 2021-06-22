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

#include "source/fuzz/transformation_wrap_vector_synonym.h"
#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationWrapVectorSynonym, SimpleTest) {

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %74
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "i1"
               OpName %10 "i2"
               OpName %14 "ui1"
               OpName %16 "ui2"
               OpName %20 "f1"
               OpName %22 "f2"
               OpName %24 "int_add"
               OpName %28 "int_sub"
               OpName %32 "int_mul"
               OpName %36 "int_div"
               OpName %40 "uint_add"
               OpName %44 "uint_sub"
               OpName %48 "uint_mul"
               OpName %52 "uint_div"
               OpName %56 "float_add"
               OpName %60 "float_sub"
               OpName %64 "float_mul"
               OpName %68 "float_div"
               OpName %74 "value"
               OpDecorate %8 RelaxedPrecision
               OpDecorate %10 RelaxedPrecision
               OpDecorate %14 RelaxedPrecision
               OpDecorate %16 RelaxedPrecision
               OpDecorate %24 RelaxedPrecision
               OpDecorate %25 RelaxedPrecision
               OpDecorate %26 RelaxedPrecision
               OpDecorate %27 RelaxedPrecision
               OpDecorate %28 RelaxedPrecision
               OpDecorate %29 RelaxedPrecision
               OpDecorate %30 RelaxedPrecision
               OpDecorate %31 RelaxedPrecision
               OpDecorate %32 RelaxedPrecision
               OpDecorate %33 RelaxedPrecision
               OpDecorate %34 RelaxedPrecision
               OpDecorate %35 RelaxedPrecision
               OpDecorate %36 RelaxedPrecision
               OpDecorate %37 RelaxedPrecision
               OpDecorate %38 RelaxedPrecision
               OpDecorate %39 RelaxedPrecision
               OpDecorate %40 RelaxedPrecision
               OpDecorate %41 RelaxedPrecision
               OpDecorate %42 RelaxedPrecision
               OpDecorate %43 RelaxedPrecision
               OpDecorate %44 RelaxedPrecision
               OpDecorate %45 RelaxedPrecision
               OpDecorate %46 RelaxedPrecision
               OpDecorate %47 RelaxedPrecision
               OpDecorate %48 RelaxedPrecision
               OpDecorate %49 RelaxedPrecision
               OpDecorate %50 RelaxedPrecision
               OpDecorate %51 RelaxedPrecision
               OpDecorate %52 RelaxedPrecision
               OpDecorate %53 RelaxedPrecision
               OpDecorate %54 RelaxedPrecision
               OpDecorate %55 RelaxedPrecision
               OpDecorate %74 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 10
         %11 = OpConstant %6 -5
         %12 = OpTypeInt 32 0
         %13 = OpTypePointer Function %12
         %15 = OpConstant %12 8
         %17 = OpConstant %12 2
         %18 = OpTypeFloat 32
         %19 = OpTypePointer Function %18
         %21 = OpConstant %18 3.29999995
         %23 = OpConstant %18 1.10000002
         %73 = OpTypePointer Input %18
         %74 = OpVariable %73 Input
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %13 Function
         %16 = OpVariable %13 Function
         %20 = OpVariable %19 Function
         %22 = OpVariable %19 Function
         %24 = OpVariable %7 Function
         %28 = OpVariable %7 Function
         %32 = OpVariable %7 Function
         %36 = OpVariable %7 Function
         %40 = OpVariable %13 Function
         %44 = OpVariable %13 Function
         %48 = OpVariable %13 Function
         %52 = OpVariable %13 Function
         %56 = OpVariable %19 Function
         %60 = OpVariable %19 Function
         %64 = OpVariable %19 Function
         %68 = OpVariable %19 Function
               OpStore %8 %9
               OpStore %10 %11
               OpStore %14 %15
               OpStore %16 %17
               OpStore %20 %21
               OpStore %22 %23
         %25 = OpLoad %6 %8
         %26 = OpLoad %6 %10
         %27 = OpIAdd %6 %25 %26
               OpStore %24 %27
         %29 = OpLoad %6 %8
         %30 = OpLoad %6 %10
         %31 = OpISub %6 %29 %30
               OpStore %28 %31
         %33 = OpLoad %6 %8
         %34 = OpLoad %6 %10
         %35 = OpIMul %6 %33 %34
               OpStore %32 %35
         %37 = OpLoad %6 %8
         %38 = OpLoad %6 %10
         %39 = OpSDiv %6 %37 %38
               OpStore %36 %39
         %41 = OpLoad %12 %14
         %42 = OpLoad %12 %16
         %43 = OpIAdd %12 %41 %42
               OpStore %40 %43
         %45 = OpLoad %12 %14
         %46 = OpLoad %12 %16
         %47 = OpISub %12 %45 %46
               OpStore %44 %47
         %49 = OpLoad %12 %14
         %50 = OpLoad %12 %16
         %51 = OpIMul %12 %49 %50
               OpStore %48 %51
         %53 = OpLoad %12 %14
         %54 = OpLoad %12 %16
         %55 = OpUDiv %12 %53 %54
               OpStore %52 %55
         %57 = OpLoad %18 %20
         %58 = OpLoad %18 %22
         %59 = OpFAdd %18 %57 %58
               OpStore %56 %59
         %61 = OpLoad %18 %20
         %62 = OpLoad %18 %22
         %63 = OpFSub %18 %61 %62
               OpStore %60 %63
         %65 = OpLoad %18 %20
         %66 = OpLoad %18 %22
         %67 = OpFMul %18 %65 %66
               OpStore %64 %67
         %69 = OpLoad %18 %20
         %70 = OpLoad %18 %22
         %71 = OpFDiv %18 %69 %70
               OpStore %68 %71
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;

  // Check context validity.
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));

  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);



  std::string after_transformation = R"(

  )";
//  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
