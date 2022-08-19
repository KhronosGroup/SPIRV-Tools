// GENERATED FILE - DO NOT EDIT.
// Generated by generate_tests.py
//
// Copyright (c) 2022 Google LLC.
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

#include "../diff_test_utils.h"

#include "gtest/gtest.h"

namespace spvtools {
namespace diff {
namespace {

// Tests OpSpecConstantComposite matching.
constexpr char kSrc[] = R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %sc SpecId 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
         %sc = OpSpecConstant %uint 10
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd)";
constexpr char kDst[] = R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
         %ss = OpSpecConstant %uint 10
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd)";

TEST(DiffTest, SpecConstantSpecid) {
  constexpr char kDiff[] = R"( ; SPIR-V
 ; Version: 1.6
 ; Generator: Khronos SPIR-V Tools Assembler; 0
-; Bound: 8
+; Bound: 9
 ; Schema: 0
 OpCapability Shader
 OpMemoryModel Logical GLSL450
 OpEntryPoint GLCompute %1 "main"
 OpExecutionMode %1 LocalSize 1 1 1
-OpDecorate %2 SpecId 0
 %4 = OpTypeVoid
 %3 = OpTypeFunction %4
 %6 = OpTypeInt 32 0
 %7 = OpTypeVector %6 3
-%2 = OpSpecConstant %6 10
+%8 = OpSpecConstant %6 10
 %1 = OpFunction %4 None %3
 %5 = OpLabel
 OpReturn
 OpFunctionEnd
)";
  Options options;
  DoStringDiffTest(kSrc, kDst, kDiff, options);
}

TEST(DiffTest, SpecConstantSpecidNoDebug) {
  constexpr char kSrcNoDebug[] = R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %sc SpecId 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
         %sc = OpSpecConstant %uint 10
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd)";
  constexpr char kDstNoDebug[] = R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
         %ss = OpSpecConstant %uint 10
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd)";
  constexpr char kDiff[] = R"( ; SPIR-V
 ; Version: 1.6
 ; Generator: Khronos SPIR-V Tools Assembler; 0
-; Bound: 8
+; Bound: 9
 ; Schema: 0
 OpCapability Shader
 OpMemoryModel Logical GLSL450
 OpEntryPoint GLCompute %1 "main"
 OpExecutionMode %1 LocalSize 1 1 1
-OpDecorate %2 SpecId 0
 %4 = OpTypeVoid
 %3 = OpTypeFunction %4
 %6 = OpTypeInt 32 0
 %7 = OpTypeVector %6 3
-%2 = OpSpecConstant %6 10
+%8 = OpSpecConstant %6 10
 %1 = OpFunction %4 None %3
 %5 = OpLabel
 OpReturn
 OpFunctionEnd
)";
  Options options;
  DoStringDiffTest(kSrcNoDebug, kDstNoDebug, kDiff, options);
}

}  // namespace
}  // namespace diff
}  // namespace spvtools
