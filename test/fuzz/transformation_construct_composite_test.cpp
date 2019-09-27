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

#include "source/fuzz/transformation_construct_composite.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationConstructCompositeTest, ConstructVectors) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %9 "v2"
               OpName %27 "v3"
               OpName %46 "v4"
               OpName %53 "iv2"
               OpName %61 "uv3"
               OpName %72 "bv4"
               OpName %88 "uv2"
               OpName %95 "bv3"
               OpName %104 "bv2"
               OpName %116 "iv3"
               OpName %124 "iv4"
               OpName %133 "uv4"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 2
          %8 = OpTypePointer Function %7
         %10 = OpConstant %6 1
         %11 = OpConstant %6 2
         %12 = OpConstantComposite %7 %10 %11
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 0
         %15 = OpTypePointer Function %6
         %18 = OpConstant %13 1
         %21 = OpTypeBool
         %25 = OpTypeVector %6 3
         %26 = OpTypePointer Function %25
         %33 = OpConstant %6 3
         %34 = OpConstant %6 -0.756802499
         %38 = OpConstant %13 2
         %44 = OpTypeVector %6 4
         %45 = OpTypePointer Function %44
         %50 = OpTypeInt 32 1
         %51 = OpTypeVector %50 2
         %52 = OpTypePointer Function %51
         %57 = OpTypePointer Function %50
         %59 = OpTypeVector %13 3
         %60 = OpTypePointer Function %59
         %65 = OpConstant %13 3
         %67 = OpTypePointer Function %13
         %70 = OpTypeVector %21 4
         %71 = OpTypePointer Function %70
         %73 = OpConstantTrue %21
         %74 = OpTypePointer Function %21
         %86 = OpTypeVector %13 2
         %87 = OpTypePointer Function %86
         %93 = OpTypeVector %21 3
         %94 = OpTypePointer Function %93
        %102 = OpTypeVector %21 2
        %103 = OpTypePointer Function %102
        %111 = OpConstantFalse %21
        %114 = OpTypeVector %50 3
        %115 = OpTypePointer Function %114
        %117 = OpConstant %50 3
        %122 = OpTypeVector %50 4
        %123 = OpTypePointer Function %122
        %131 = OpTypeVector %13 4
        %132 = OpTypePointer Function %131
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %9 = OpVariable %8 Function
         %27 = OpVariable %26 Function
         %46 = OpVariable %45 Function
         %53 = OpVariable %52 Function
         %61 = OpVariable %60 Function
         %72 = OpVariable %71 Function
         %88 = OpVariable %87 Function
         %95 = OpVariable %94 Function
        %104 = OpVariable %103 Function
        %116 = OpVariable %115 Function
        %124 = OpVariable %123 Function
        %133 = OpVariable %132 Function
               OpStore %9 %12
         %16 = OpAccessChain %15 %9 %14
         %17 = OpLoad %6 %16
         %19 = OpAccessChain %15 %9 %18
         %20 = OpLoad %6 %19
         %22 = OpFOrdGreaterThan %21 %17 %20
               OpSelectionMerge %24 None
               OpBranchConditional %22 %23 %101
         %23 = OpLabel
         %28 = OpAccessChain %15 %9 %14
         %29 = OpLoad %6 %28
         %30 = OpAccessChain %15 %9 %18
         %31 = OpLoad %6 %30
         %32 = OpFAdd %6 %29 %31
         %35 = OpCompositeConstruct %25 %32 %33 %34
               OpStore %27 %35
         %36 = OpAccessChain %15 %27 %14
         %37 = OpLoad %6 %36
         %39 = OpAccessChain %15 %27 %38
         %40 = OpLoad %6 %39
         %41 = OpFOrdLessThan %21 %37 %40
               OpSelectionMerge %43 None
               OpBranchConditional %41 %42 %69
         %42 = OpLabel
         %47 = OpAccessChain %15 %9 %18
         %48 = OpLoad %6 %47
         %49 = OpAccessChain %15 %46 %14
               OpStore %49 %48
         %54 = OpAccessChain %15 %27 %38
         %55 = OpLoad %6 %54
         %56 = OpConvertFToS %50 %55
         %58 = OpAccessChain %57 %53 %14
               OpStore %58 %56
         %62 = OpAccessChain %15 %46 %14
         %63 = OpLoad %6 %62
         %64 = OpConvertFToU %13 %63
         %66 = OpIAdd %13 %64 %65
         %68 = OpAccessChain %67 %61 %14
               OpStore %68 %66
               OpBranch %43
         %69 = OpLabel
         %75 = OpAccessChain %74 %72 %14
               OpStore %75 %73
         %76 = OpAccessChain %74 %72 %14
         %77 = OpLoad %21 %76
         %78 = OpLogicalNot %21 %77
         %79 = OpAccessChain %74 %72 %18
               OpStore %79 %78
         %80 = OpAccessChain %74 %72 %14
         %81 = OpLoad %21 %80
         %82 = OpAccessChain %74 %72 %18
         %83 = OpLoad %21 %82
         %84 = OpLogicalAnd %21 %81 %83
         %85 = OpAccessChain %74 %72 %38
               OpStore %85 %84
         %89 = OpAccessChain %67 %88 %14
         %90 = OpLoad %13 %89
         %91 = OpINotEqual %21 %90 %14
         %92 = OpAccessChain %74 %72 %65
               OpStore %92 %91
               OpBranch %43
         %43 = OpLabel
         %96 = OpLoad %70 %72
         %97 = OpCompositeExtract %21 %96 0
         %98 = OpCompositeExtract %21 %96 1
         %99 = OpCompositeExtract %21 %96 2
        %100 = OpCompositeConstruct %93 %97 %98 %99
               OpStore %95 %100
               OpBranch %24
        %101 = OpLabel
        %105 = OpAccessChain %67 %88 %14
        %106 = OpLoad %13 %105
        %107 = OpINotEqual %21 %106 %14
        %108 = OpCompositeConstruct %102 %107 %107
               OpStore %104 %108
               OpBranch %24
         %24 = OpLabel
        %109 = OpAccessChain %74 %104 %18
        %110 = OpLoad %21 %109
        %112 = OpLogicalOr %21 %110 %111
        %113 = OpAccessChain %74 %104 %14
               OpStore %113 %112
        %118 = OpAccessChain %57 %116 %14
               OpStore %118 %117
        %119 = OpAccessChain %57 %116 %14
        %120 = OpLoad %50 %119
        %121 = OpAccessChain %57 %53 %18
               OpStore %121 %120
        %125 = OpAccessChain %57 %116 %14
        %126 = OpLoad %50 %125
        %127 = OpAccessChain %57 %53 %18
        %128 = OpLoad %50 %127
        %129 = OpIAdd %50 %126 %128
        %130 = OpAccessChain %57 %124 %65
               OpStore %130 %129
        %134 = OpAccessChain %57 %116 %14
        %135 = OpLoad %50 %134
        %136 = OpBitcast %13 %135
        %137 = OpAccessChain %67 %133 %14
               OpStore %137 %136
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  TransformationConstructComposite make_vec2(7, {17, 11}, 100, 1, 200);
  // Bad: not enough data for a vec2
  TransformationConstructComposite make_vec2_bad(7, {11}, 100, 1, 200);
  ASSERT_TRUE(make_vec2.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_vec2_bad.IsApplicable(context.get(), fact_manager));
  make_vec2.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_vec3(25, {12, 32}, 35, 0, 201);
  // Bad: too much data for a vec3
  TransformationConstructComposite make_vec3_bad(25, {12, 32, 32}, 35, 0, 201);
  ASSERT_TRUE(make_vec3.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_vec3_bad.IsApplicable(context.get(), fact_manager));
  make_vec3.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_vec4(44, {32, 32, 10, 11}, 75, 0, 202);
  // Bad: id 48 is not available at the insertion points
  TransformationConstructComposite make_vec4_bad(44, {48, 32, 10, 11}, 75, 0,
                                                 202);
  ASSERT_TRUE(make_vec4.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_vec4_bad.IsApplicable(context.get(), fact_manager));
  make_vec4.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_ivec2(51, {126, 120}, 128, 0, 203);
  // Bad: if 128 is not available at the instruction that defines 128
  TransformationConstructComposite make_ivec2_bad(51, {128, 120}, 128, 0, 203);
  ASSERT_TRUE(make_ivec2.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_ivec2_bad.IsApplicable(context.get(), fact_manager));
  make_ivec2.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_ivec3(114, {56, 117, 56}, 66, 1, 204);
  // Bad because 1300 is not an id
  TransformationConstructComposite make_ivec3_bad(114, {56, 117, 1300}, 66, 1,
                                                  204);
  ASSERT_TRUE(make_ivec3.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_ivec3_bad.IsApplicable(context.get(), fact_manager));
  make_ivec3.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_ivec4(122, {56, 117, 117, 117}, 66, 0,
                                              205);
  // Bad because 86 is the wrong type.
  TransformationConstructComposite make_ivec4_bad(86, {56, 117, 117, 117}, 66,
                                                  0, 205);
  ASSERT_TRUE(make_ivec4.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_ivec4_bad.IsApplicable(context.get(), fact_manager));
  make_ivec4.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_uvec2(86, {18, 38}, 133, 2, 206);
  TransformationConstructComposite make_uvec2_bad(86, {18, 38}, 133, 200, 206);
  ASSERT_TRUE(make_uvec2.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_uvec2_bad.IsApplicable(context.get(), fact_manager));
  make_uvec2.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_uvec3(59, {14, 18, 136}, 137, 2, 207);
  // Bad because 1300 is not an id
  TransformationConstructComposite make_uvec3_bad(59, {14, 18, 1300}, 137, 2,
                                                  207);
  ASSERT_TRUE(make_uvec3.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_uvec3_bad.IsApplicable(context.get(), fact_manager));
  make_uvec3.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_uvec4(131, {14, 18, 136, 136}, 137, 0,
                                              208);
  // Bad because 86 is the wrong type.
  TransformationConstructComposite make_uvec4_bad(86, {14, 18, 136, 136}, 137,
                                                  0, 208);
  ASSERT_TRUE(make_uvec4.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_uvec4_bad.IsApplicable(context.get(), fact_manager));
  make_uvec4.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_bvec2(102,
                                              {
                                                  111,
                                                  41,
                                              },
                                              75, 0, 209);
  // Bad because 0 is not a valid base instruction id
  TransformationConstructComposite make_bvec2_bad(102,
                                                  {
                                                      111,
                                                      41,
                                                  },
                                                  0, 0, 209);
  ASSERT_TRUE(make_bvec2.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_bvec2_bad.IsApplicable(context.get(), fact_manager));
  make_bvec2.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_bvec3(93, {108, 73}, 108, 1, 210);
  // Bad because there are too many components for a bvec3
  TransformationConstructComposite make_bvec3_bad(93, {108, 108}, 108, 1, 210);
  ASSERT_TRUE(make_bvec3.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_bvec3_bad.IsApplicable(context.get(), fact_manager));
  make_bvec3.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  TransformationConstructComposite make_bvec4(70, {108, 108}, 108, 3, 211);
  // Bad because 21 is a type, not a result id
  TransformationConstructComposite make_bvec4_bad(70, {21, 108}, 108, 3, 211);
  ASSERT_TRUE(make_bvec4.IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(make_bvec4_bad.IsApplicable(context.get(), fact_manager));
  make_bvec4.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %9 "v2"
               OpName %27 "v3"
               OpName %46 "v4"
               OpName %53 "iv2"
               OpName %61 "uv3"
               OpName %72 "bv4"
               OpName %88 "uv2"
               OpName %95 "bv3"
               OpName %104 "bv2"
               OpName %116 "iv3"
               OpName %124 "iv4"
               OpName %133 "uv4"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 2
          %8 = OpTypePointer Function %7
         %10 = OpConstant %6 1
         %11 = OpConstant %6 2
         %12 = OpConstantComposite %7 %10 %11
         %13 = OpTypeInt 32 0
         %14 = OpConstant %13 0
         %15 = OpTypePointer Function %6
         %18 = OpConstant %13 1
         %21 = OpTypeBool
         %25 = OpTypeVector %6 3
         %26 = OpTypePointer Function %25
         %33 = OpConstant %6 3
         %34 = OpConstant %6 -0.756802499
         %38 = OpConstant %13 2
         %44 = OpTypeVector %6 4
         %45 = OpTypePointer Function %44
         %50 = OpTypeInt 32 1
         %51 = OpTypeVector %50 2
         %52 = OpTypePointer Function %51
         %57 = OpTypePointer Function %50
         %59 = OpTypeVector %13 3
         %60 = OpTypePointer Function %59
         %65 = OpConstant %13 3
         %67 = OpTypePointer Function %13
         %70 = OpTypeVector %21 4
         %71 = OpTypePointer Function %70
         %73 = OpConstantTrue %21
         %74 = OpTypePointer Function %21
         %86 = OpTypeVector %13 2
         %87 = OpTypePointer Function %86
         %93 = OpTypeVector %21 3
         %94 = OpTypePointer Function %93
        %102 = OpTypeVector %21 2
        %103 = OpTypePointer Function %102
        %111 = OpConstantFalse %21
        %114 = OpTypeVector %50 3
        %115 = OpTypePointer Function %114
        %117 = OpConstant %50 3
        %122 = OpTypeVector %50 4
        %123 = OpTypePointer Function %122
        %131 = OpTypeVector %13 4
        %132 = OpTypePointer Function %131
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %9 = OpVariable %8 Function
         %27 = OpVariable %26 Function
         %46 = OpVariable %45 Function
         %53 = OpVariable %52 Function
         %61 = OpVariable %60 Function
         %72 = OpVariable %71 Function
         %88 = OpVariable %87 Function
         %95 = OpVariable %94 Function
        %104 = OpVariable %103 Function
        %116 = OpVariable %115 Function
        %124 = OpVariable %123 Function
        %133 = OpVariable %132 Function
               OpStore %9 %12
        %206 = OpCompositeConstruct %86 %18 %38
         %16 = OpAccessChain %15 %9 %14
         %17 = OpLoad %6 %16
         %19 = OpAccessChain %15 %9 %18
         %20 = OpLoad %6 %19
         %22 = OpFOrdGreaterThan %21 %17 %20
               OpSelectionMerge %24 None
               OpBranchConditional %22 %23 %101
         %23 = OpLabel
         %28 = OpAccessChain %15 %9 %14
         %29 = OpLoad %6 %28
         %30 = OpAccessChain %15 %9 %18
         %31 = OpLoad %6 %30
         %32 = OpFAdd %6 %29 %31
        %201 = OpCompositeConstruct %25 %12 %32
         %35 = OpCompositeConstruct %25 %32 %33 %34
               OpStore %27 %35
         %36 = OpAccessChain %15 %27 %14
         %37 = OpLoad %6 %36
         %39 = OpAccessChain %15 %27 %38
         %40 = OpLoad %6 %39
         %41 = OpFOrdLessThan %21 %37 %40
               OpSelectionMerge %43 None
               OpBranchConditional %41 %42 %69
         %42 = OpLabel
         %47 = OpAccessChain %15 %9 %18
         %48 = OpLoad %6 %47
         %49 = OpAccessChain %15 %46 %14
               OpStore %49 %48
         %54 = OpAccessChain %15 %27 %38
         %55 = OpLoad %6 %54
         %56 = OpConvertFToS %50 %55
         %58 = OpAccessChain %57 %53 %14
               OpStore %58 %56
         %62 = OpAccessChain %15 %46 %14
         %63 = OpLoad %6 %62
         %64 = OpConvertFToU %13 %63
        %205 = OpCompositeConstruct %122 %56 %117 %117 %117
         %66 = OpIAdd %13 %64 %65
        %204 = OpCompositeConstruct %114 %56 %117 %56
         %68 = OpAccessChain %67 %61 %14
               OpStore %68 %66
               OpBranch %43
         %69 = OpLabel
        %202 = OpCompositeConstruct %44 %32 %32 %10 %11
        %209 = OpCompositeConstruct %102 %111 %41
         %75 = OpAccessChain %74 %72 %14
               OpStore %75 %73
         %76 = OpAccessChain %74 %72 %14
         %77 = OpLoad %21 %76
         %78 = OpLogicalNot %21 %77
         %79 = OpAccessChain %74 %72 %18
               OpStore %79 %78
         %80 = OpAccessChain %74 %72 %14
         %81 = OpLoad %21 %80
         %82 = OpAccessChain %74 %72 %18
         %83 = OpLoad %21 %82
         %84 = OpLogicalAnd %21 %81 %83
         %85 = OpAccessChain %74 %72 %38
               OpStore %85 %84
         %89 = OpAccessChain %67 %88 %14
         %90 = OpLoad %13 %89
         %91 = OpINotEqual %21 %90 %14
         %92 = OpAccessChain %74 %72 %65
               OpStore %92 %91
               OpBranch %43
         %43 = OpLabel
         %96 = OpLoad %70 %72
         %97 = OpCompositeExtract %21 %96 0
         %98 = OpCompositeExtract %21 %96 1
         %99 = OpCompositeExtract %21 %96 2
        %100 = OpCompositeConstruct %93 %97 %98 %99
        %200 = OpCompositeConstruct %7 %17 %11
               OpStore %95 %100
               OpBranch %24
        %101 = OpLabel
        %105 = OpAccessChain %67 %88 %14
        %106 = OpLoad %13 %105
        %107 = OpINotEqual %21 %106 %14
        %108 = OpCompositeConstruct %102 %107 %107
        %210 = OpCompositeConstruct %93 %108 %73
               OpStore %104 %108
        %211 = OpCompositeConstruct %70 %108 %108
               OpBranch %24
         %24 = OpLabel
        %109 = OpAccessChain %74 %104 %18
        %110 = OpLoad %21 %109
        %112 = OpLogicalOr %21 %110 %111
        %113 = OpAccessChain %74 %104 %14
               OpStore %113 %112
        %118 = OpAccessChain %57 %116 %14
               OpStore %118 %117
        %119 = OpAccessChain %57 %116 %14
        %120 = OpLoad %50 %119
        %121 = OpAccessChain %57 %53 %18
               OpStore %121 %120
        %125 = OpAccessChain %57 %116 %14
        %126 = OpLoad %50 %125
        %127 = OpAccessChain %57 %53 %18
        %203 = OpCompositeConstruct %51 %126 %120
        %128 = OpLoad %50 %127
        %129 = OpIAdd %50 %126 %128
        %130 = OpAccessChain %57 %124 %65
               OpStore %130 %129
        %134 = OpAccessChain %57 %116 %14
        %135 = OpLoad %50 %134
        %136 = OpBitcast %13 %135
        %208 = OpCompositeConstruct %131 %14 %18 %136 %136
        %137 = OpAccessChain %67 %133 %14
               OpStore %137 %136
        %207 = OpCompositeConstruct %59 %14 %18 %136
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
