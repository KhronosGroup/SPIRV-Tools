// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_swap_functions.h"

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationSwapFunctions, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %72 %74
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %12 "g(f1;f1;"
               OpName %10 "x"
               OpName %11 "y"
               OpName %22 "f(f1;i1;vf2;"
               OpName %19 "x"
               OpName %20 "y"
               OpName %21 "z"
               OpName %28 "cond(i1;f1;"
               OpName %26 "a"
               OpName %27 "b"
               OpName %53 "param"
               OpName %54 "param"
               OpName %66 "param"
               OpName %67 "param"
               OpName %72 "color"
               OpName %74 "gl_FragCoord"
               OpName %75 "param"
               OpName %79 "param"
               OpName %85 "param"
               OpName %86 "param"
               OpName %91 "param"
               OpName %92 "param"
               OpName %93 "param"
               OpName %99 "param"
               OpName %100 "param"
               OpName %101 "param"
               OpDecorate %20 RelaxedPrecision
               OpDecorate %26 RelaxedPrecision
               OpDecorate %47 RelaxedPrecision
               OpDecorate %58 RelaxedPrecision
               OpDecorate %72 Location 0
               OpDecorate %74 BuiltIn FragCoord
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %8 = OpTypeVector %6 4
          %9 = OpTypeFunction %8 %7 %7
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %16 = OpTypeVector %6 2
         %17 = OpTypePointer Function %16
         %18 = OpTypeFunction %8 %7 %15 %17
         %24 = OpTypeBool
         %25 = OpTypeFunction %24 %15 %7
         %31 = OpConstant %6 255
         %33 = OpConstant %6 0
         %34 = OpConstant %6 1
         %42 = OpTypeInt 32 0
         %43 = OpConstant %42 0
         %49 = OpConstant %42 1
         %64 = OpConstant %14 4
         %65 = OpConstant %6 5
         %71 = OpTypePointer Output %8
         %72 = OpVariable %71 Output
         %73 = OpTypePointer Input %8
         %74 = OpVariable %73 Input
         %76 = OpTypePointer Input %6
         %84 = OpConstant %14 5
         %90 = OpConstant %6 3
         %98 = OpConstant %6 4
        %206 = OpTypeFunction %2 %14 %16
        %223 = OpTypeFunction %2 %6 %8
        %224 = OpTypeFunction %2 %8 %6
        %233 = OpTypeFunction %2 %42 %24
        %234 = OpTypeFunction %2 %24 %42
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %66 = OpVariable %15 Function
         %67 = OpVariable %7 Function
         %75 = OpVariable %7 Function
         %79 = OpVariable %7 Function
         %85 = OpVariable %15 Function
         %86 = OpVariable %7 Function
         %91 = OpVariable %7 Function
         %92 = OpVariable %15 Function
         %93 = OpVariable %17 Function
         %99 = OpVariable %7 Function
        %100 = OpVariable %15 Function
        %101 = OpVariable %17 Function
               OpStore %66 %64
               OpStore %67 %65
         %68 = OpFunctionCall %24 %28 %66 %67
               OpSelectionMerge %70 None
               OpBranchConditional %68 %69 %83
         %69 = OpLabel
         %77 = OpAccessChain %76 %74 %43
         %78 = OpLoad %6 %77
               OpStore %75 %78
         %80 = OpAccessChain %76 %74 %49
         %81 = OpLoad %6 %80
               OpStore %79 %81
         %82 = OpFunctionCall %8 %12 %75 %79
               OpStore %72 %82
               OpBranch %70
         %83 = OpLabel
               OpStore %85 %84
               OpStore %86 %65
         %87 = OpFunctionCall %24 %28 %85 %86
               OpSelectionMerge %89 None
               OpBranchConditional %87 %88 %97
         %88 = OpLabel
               OpStore %91 %90
               OpStore %92 %64
         %94 = OpLoad %8 %74
         %95 = OpVectorShuffle %16 %94 %94 0 1
               OpStore %93 %95
         %96 = OpFunctionCall %8 %22 %91 %92 %93
               OpStore %72 %96
               OpBranch %89
         %97 = OpLabel
               OpStore %99 %98
               OpStore %100 %84
        %102 = OpLoad %8 %74
        %103 = OpVectorShuffle %16 %102 %102 0 1
               OpStore %101 %103
        %104 = OpFunctionCall %8 %22 %99 %100 %101
               OpStore %72 %104
               OpBranch %89
         %89 = OpLabel
               OpBranch %70
         %70 = OpLabel
               OpReturn
               OpFunctionEnd
         
         ; Function0
         ; adjust type of the function in-place
         %12 = OpFunction %8 None %9
         %10 = OpFunctionParameter %7
         %11 = OpFunctionParameter %7
         %13 = OpLabel
         %30 = OpLoad %6 %10
         %32 = OpFDiv %6 %30 %31
         %35 = OpLoad %6 %11
         %36 = OpFDiv %6 %35 %31
         %37 = OpFSub %6 %34 %36
         %38 = OpCompositeConstruct %8 %32 %33 %37 %34
               OpReturnValue %38
               OpFunctionEnd
         
         ; Function1
         %22 = OpFunction %8 None %18
         %19 = OpFunctionParameter %7
         %20 = OpFunctionParameter %15
         %21 = OpFunctionParameter %17
         %23 = OpLabel
         %53 = OpVariable %7 Function
         %54 = OpVariable %7 Function
         %41 = OpLoad %6 %19
         %44 = OpAccessChain %7 %21 %43
         %45 = OpLoad %6 %44
         %46 = OpFAdd %6 %41 %45
         %47 = OpLoad %14 %20
         %48 = OpConvertSToF %6 %47
         %50 = OpAccessChain %7 %21 %49
         %51 = OpLoad %6 %50
         %52 = OpFAdd %6 %48 %51
               OpStore %53 %46
               OpStore %54 %52
         %55 = OpFunctionCall %8 %12 %53 %54
               OpReturnValue %55
               OpFunctionEnd
         
         ; Function2
         %28 = OpFunction %24 None %25
         %26 = OpFunctionParameter %15
         %27 = OpFunctionParameter %7
         %29 = OpLabel
         %58 = OpLoad %14 %26
         %59 = OpConvertSToF %6 %58
         %60 = OpLoad %6 %27
         %61 = OpFOrdLessThan %24 %59 %60
               OpReturnValue %61
               OpFunctionEnd
        
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  // Try random permutations

  ASSERT_TRUE(TransformationSwapFunctions({68, 87}).IsApplicable(
      context.get(), transformation_context));

  ASSERT_TRUE(TransformationSwapFunctions({68, 82}).IsApplicable(
      context.get(), transformation_context));

  ASSERT_TRUE(TransformationSwapFunctions({96, 104}).IsApplicable(
      context.get(), transformation_context));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %72 %74
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %12 "g(f1;f1;"
               OpName %10 "x"
               OpName %11 "y"
               OpName %22 "f(f1;i1;vf2;"
               OpName %19 "x"
               OpName %20 "y"
               OpName %21 "z"
               OpName %28 "cond(i1;f1;"
               OpName %26 "a"
               OpName %27 "b"
               OpName %53 "param"
               OpName %54 "param"
               OpName %66 "param"
               OpName %67 "param"
               OpName %72 "color"
               OpName %74 "gl_FragCoord"
               OpName %75 "param"chrome://downloadswe
               OpName %79 "param"
               OpName %85 "param"
               OpName %86 "param"
               OpName %91 "param"
               OpName %92 "param"
               OpName %93 "param"
               OpName %99 "param"
               OpName %100 "param"
               OpName %101 "param"
               OpDecorate %20 RelaxedPrecision
               OpDecorate %26 RelaxedPrecision
               OpDecorate %47 RelaxedPrecision
               OpDecorate %58 RelaxedPrecision
               OpDecorate %72 Location 0
               OpDecorate %74 BuiltIn FragCoord
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %8 = OpTypeVector %6 4
          %9 = OpTypeFunction %8 %7 %7
         %14 = OpTypeInt 32 1
         %15 = OpTypePointer Function %14
         %16 = OpTypeVector %6 2
         %17 = OpTypePointer Function %16
         %18 = OpTypeFunction %8 %7 %15 %17
         %24 = OpTypeBool
         %25 = OpTypeFunction %24 %15 %7
         %31 = OpConstant %6 255
         %33 = OpConstant %6 0
         %34 = OpConstant %6 1
         %42 = OpTypeInt 32 0
         %43 = OpConstant %42 0
         %49 = OpConstant %42 1
         %64 = OpConstant %14 4
         %65 = OpConstant %6 5
         %71 = OpTypePointer Output %8
         %72 = OpVariable %71 Output
         %73 = OpTypePointer Input %8
         %74 = OpVariable %73 Input
         %76 = OpTypePointer Input %6
         %84 = OpConstant %14 5
         %90 = OpConstant %6 3
         %98 = OpConstant %6 4
        %206 = OpTypeFunction %2 %14 %16
        %223 = OpTypeFunction %2 %6 %8
        %224 = OpTypeFunction %2 %8 %6
        %233 = OpTypeFunction %2 %42 %24
        %234 = OpTypeFunction %2 %24 %42
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %66 = OpVariable %15 Function
         %67 = OpVariable %7 Function
         %75 = OpVariable %7 Function
         %79 = OpVariable %7 Function
         %85 = OpVariable %15 Function
         %86 = OpVariable %7 Function
         %91 = OpVariable %7 Function
         %92 = OpVariable %15 Function
         %93 = OpVariable %17 Function
         %99 = OpVariable %7 Function
        %100 = OpVariable %15 Function
        %101 = OpVariable %17 Function
               OpStore %66 %64
               OpStore %67 %65
        
        %87 = OpFunctionCall %24 %28 %85 %86
            OpSelectionMerge %89 None
            OpBranchConditional %87 %88 %97

         
         %69 = OpLabel
         %77 = OpAccessChain %76 %74 %43
         %78 = OpLoad %6 %77
               OpStore %75 %78
         %80 = OpAccessChain %76 %74 %49
         %81 = OpLoad %6 %80
               OpStore %79 %81
         
         %68 = OpFunctionCall %24 %28 %66 %67
               OpSelectionMerge %70 None
               OpBranchConditional %68 %69 %83
        
         %83 = OpLabel
               OpStore %85 %84
               OpStore %86 %65
        
        %82 = OpFunctionCall %8 %12 %75 %79
               OpStore %72 %82
               OpBranch %70
         %88 = OpLabel
               OpStore %91 %90
               OpStore %92 %64
         %94 = OpLoad %8 %74
         %95 = OpVectorShuffle %16 %94 %94 0 1
               OpStore %93 %95
         
         %104 = OpFunctionCall %8 %22 %99 %100 %101
               OpStore %72 %104
               OpBranch %89
        
         %97 = OpLabel
               OpStore %99 %98
               OpStore %100 %84
        %102 = OpLoad %8 %74
        %103 = OpVectorShuffle %16 %102 %102 0 1
               OpStore %101 %103
    
        %96 = OpFunctionCall %8 %22 %91 %92 %93
            OpStore %72 %96
            OpBranch %89
            
         %89 = OpLabel
               OpBranch %70
         %70 = OpLabel
               OpReturn
               OpFunctionEnd
         
         ; Function0
         ; adjust type of the function in-place
         %12 = OpFunction %8 None %9
         %10 = OpFunctionParameter %7
         %11 = OpFunctionParameter %7
         %13 = OpLabel
         %30 = OpLoad %6 %10
         %32 = OpFDiv %6 %30 %31
         %35 = OpLoad %6 %11
         %36 = OpFDiv %6 %35 %31
         %37 = OpFSub %6 %34 %36
         %38 = OpCompositeConstruct %8 %32 %33 %37 %34
               OpReturnValue %38
               OpFunctionEnd
         
         ; Function1
         %22 = OpFunction %8 None %18
         %19 = OpFunctionParameter %7
         %20 = OpFunctionParameter %15
         %21 = OpFunctionParameter %17
         %23 = OpLabel
         %53 = OpVariable %7 Function
         %54 = OpVariable %7 Function
         %41 = OpLoad %6 %19
         %44 = OpAccessChain %7 %21 %43
         %45 = OpLoad %6 %44
         %46 = OpFAdd %6 %41 %45
         %47 = OpLoad %14 %20
         %48 = OpConvertSToF %6 %47
         %50 = OpAccessChain %7 %21 %49
         %51 = OpLoad %6 %50
         %52 = OpFAdd %6 %48 %51
               OpStore %53 %46
               OpStore %54 %52
         %55 = OpFunctionCall %8 %12 %53 %54
               OpReturnValue %55
               OpFunctionEnd
         
         ; Function2
         %28 = OpFunction %24 None %25
         %26 = OpFunctionParameter %15
         %27 = OpFunctionParameter %7
         %29 = OpLabel
         %58 = OpLoad %14 %26
         %59 = OpConvertSToF %6 %58
         %60 = OpLoad %6 %27
         %61 = OpFOrdLessThan %24 %59 %60
               OpReturnValue %61
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
