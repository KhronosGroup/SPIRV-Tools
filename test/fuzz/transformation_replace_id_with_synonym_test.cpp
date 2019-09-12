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

#include "source/fuzz/transformation_replace_id_with_synonym.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

// The following shader was obtained from this GLSL, which was then optimized with spirv-opt -O and
// manually edited to include some uses of OpCopyObject (to introduce id synonyms).
//
// #version 310 es
//
// precision highp int;
// precision highp float;
//
// layout(set = 0, binding = 0) uniform buf {
//   int a;
//   int b;
//   int c;
// };
//
// layout(location = 0) out vec4 color;
//
// void main() {
//   int x = a;
//   float f = 0.0;
//   while (x < b) {
//     switch(x % 4) {
//       case 0:
//         color[0] = f;
//         break;
//       case 1:
//         color[1] = f;
//         break;
//       case 2:
//         color[2] = f;
//         break;
//       case 3:
//         color[3] = f;
//         break;
//       default:
//         break;
//     }
//     if (x > c) {
//       x++;
//     } else {
//       x += 2;
//     }
//   }
//   color[0] += color[1] + float(x);
// }
const std::string kComplexShader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %42
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %9 "buf"
               OpMemberName %9 0 "a"
               OpMemberName %9 1 "b"
               OpMemberName %9 2 "c"
               OpName %11 ""
               OpName %42 "color"
               OpMemberDecorate %9 0 Offset 0
               OpMemberDecorate %9 1 Offset 4
               OpMemberDecorate %9 2 Offset 8
               OpDecorate %9 Block
               OpDecorate %11 DescriptorSet 0
               OpDecorate %11 Binding 0
               OpDecorate %42 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %9 = OpTypeStruct %6 %6 %6
         %10 = OpTypePointer Uniform %9
         %11 = OpVariable %10 Uniform
         %12 = OpConstant %6 0
         %13 = OpTypePointer Uniform %6
         %16 = OpTypeFloat 32
         %19 = OpConstant %16 0
         %26 = OpConstant %6 1
         %29 = OpTypeBool
         %32 = OpConstant %6 4
         %40 = OpTypeVector %16 4
         %41 = OpTypePointer Output %40
         %42 = OpVariable %41 Output
         %44 = OpTypeInt 32 0
         %45 = OpConstant %44 0
         %46 = OpTypePointer Output %16
         %50 = OpConstant %44 1
         %54 = OpConstant %44 2
         %58 = OpConstant %44 3
         %64 = OpConstant %6 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %15 = OpLoad %6 %14
        %200 = OpCopyObject %6 %15
               OpBranch %20
         %20 = OpLabel
         %84 = OpPhi %6 %15 %5 %86 %69
         %27 = OpAccessChain %13 %11 %26
         %28 = OpLoad %6 %27
        %209 = OpCopyObject %6 %84
        %201 = OpCopyObject %6 %15
         %30 = OpSLessThan %29 %84 %28
               OpLoopMerge %22 %69 None
               OpBranchConditional %30 %21 %22
         %21 = OpLabel
         %33 = OpSMod %6 %84 %32
        %208 = OpCopyObject %6 %33
               OpSelectionMerge %39 None
               OpSwitch %33 %38 0 %34 1 %35 2 %36 3 %37
         %38 = OpLabel
        %202 = OpCopyObject %6 %15
               OpBranch %39
         %34 = OpLabel
         %47 = OpAccessChain %46 %42 %45
               OpStore %47 %19
               OpBranch %39
         %35 = OpLabel
         %51 = OpAccessChain %46 %42 %50
               OpStore %51 %19
               OpBranch %39
         %36 = OpLabel
        %204 = OpCopyObject %44 %54
         %55 = OpAccessChain %46 %42 %54
        %203 = OpCopyObject %46 %55
               OpStore %55 %19
               OpBranch %39
         %37 = OpLabel
         %59 = OpAccessChain %46 %42 %58
               OpStore %59 %19
               OpBranch %39
         %39 = OpLabel
         %65 = OpAccessChain %13 %11 %64
         %66 = OpLoad %6 %65
         %67 = OpSGreaterThan %29 %84 %66
               OpSelectionMerge %69 None
               OpBranchConditional %67 %68 %72
         %68 = OpLabel
         %71 = OpIAdd %6 %84 %26
               OpBranch %69
         %72 = OpLabel
         %74 = OpIAdd %6 %84 %64
        %205 = OpCopyObject %6 %74
               OpBranch %69
         %69 = OpLabel
         %86 = OpPhi %6 %71 %68 %74 %72
               OpBranch %20
         %22 = OpLabel
         %75 = OpAccessChain %46 %42 %50
         %76 = OpLoad %16 %75
         %78 = OpConvertSToF %16 %84
         %80 = OpAccessChain %46 %42 %45
        %206 = OpCopyObject %16 %78
         %81 = OpLoad %16 %80
         %79 = OpFAdd %16 %76 %78
         %82 = OpFAdd %16 %81 %79
               OpStore %80 %82
               OpReturn
               OpFunctionEnd
)";

TEST(TransformationReplaceIdWithSynonymTest, IllegalTransformations) {

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, kComplexShader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  // Synonym does not dominate use.
  // Id in use is not synonymous
  // Synonym use is not in synonym definition (%174 = OpCopyObject %int %174)
  // Not allowed to replace an access chain index into a structure
//  FAIL();
}

TEST(TransformationReplaceIdWithSynonymTest, LegalTransformations) {
  // Synonym of global constant
  // Synonym of global variable
  // Synonym of local expr
  // Synonym of local variable
  // Allowed to replace an access chain index into a vector
  FAIL();
}

TEST(TransformationReplaceIdWithSynonymTest, OpPhi) {
  // Transformation should affect parent block
  FAIL();
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
