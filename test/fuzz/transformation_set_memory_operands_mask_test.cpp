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

#include "source/fuzz/transformation_set_memory_operands_mask.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationSetMemoryOperandsMaskTest, PreSpirv14) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %7 "Point3D"
               OpMemberName %7 0 "x"
               OpMemberName %7 1 "y"
               OpMemberName %7 2 "z"
               OpName %12 "global_points"
               OpName %15 "block"
               OpMemberName %15 0 "in_points"
               OpMemberName %15 1 "in_point"
               OpName %17 ""
               OpName %133 "local_points"
               OpMemberDecorate %7 0 Offset 0
               OpMemberDecorate %7 1 Offset 4
               OpMemberDecorate %7 2 Offset 8
               OpDecorate %10 ArrayStride 16
               OpMemberDecorate %15 0 Offset 0
               OpMemberDecorate %15 1 Offset 192
               OpDecorate %15 Block
               OpDecorate %17 DescriptorSet 0
               OpDecorate %17 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeStruct %6 %6 %6
          %8 = OpTypeInt 32 0
          %9 = OpConstant %8 12
         %10 = OpTypeArray %7 %9
         %11 = OpTypePointer Private %10
         %12 = OpVariable %11 Private
         %15 = OpTypeStruct %10 %7
         %16 = OpTypePointer Uniform %15
         %17 = OpVariable %16 Uniform
         %18 = OpTypeInt 32 1
         %19 = OpConstant %18 0
         %20 = OpTypePointer Uniform %10
         %24 = OpTypePointer Private %7
         %27 = OpTypePointer Private %6
         %30 = OpConstant %18 1
        %132 = OpTypePointer Function %10
        %135 = OpTypePointer Uniform %7
        %145 = OpTypePointer Function %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %133 = OpVariable %132 Function
         %21 = OpAccessChain %20 %17 %19
               OpCopyMemory %12 %21 Aligned 16
               OpCopyMemory %133 %12 Volatile
        %136 = OpAccessChain %135 %17 %30
        %138 = OpAccessChain %24 %12 %19
               OpCopyMemory %138 %136 None
        %146 = OpAccessChain %145 %133 %30
        %147 = OpLoad %7 %146 Volatile|Nontemporal|Aligned 16
        %148 = OpAccessChain %24 %12 %19
               OpStore %148 %147 Nontemporal
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  // TODO - add test content

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %7 "Point3D"
               OpMemberName %7 0 "x"
               OpMemberName %7 1 "y"
               OpMemberName %7 2 "z"
               OpName %12 "global_points"
               OpName %15 "block"
               OpMemberName %15 0 "in_points"
               OpMemberName %15 1 "in_point"
               OpName %17 ""
               OpName %133 "local_points"
               OpMemberDecorate %7 0 Offset 0
               OpMemberDecorate %7 1 Offset 4
               OpMemberDecorate %7 2 Offset 8
               OpDecorate %10 ArrayStride 16
               OpMemberDecorate %15 0 Offset 0
               OpMemberDecorate %15 1 Offset 192
               OpDecorate %15 Block
               OpDecorate %17 DescriptorSet 0
               OpDecorate %17 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeStruct %6 %6 %6
          %8 = OpTypeInt 32 0
          %9 = OpConstant %8 12
         %10 = OpTypeArray %7 %9
         %11 = OpTypePointer Private %10
         %12 = OpVariable %11 Private
         %15 = OpTypeStruct %10 %7
         %16 = OpTypePointer Uniform %15
         %17 = OpVariable %16 Uniform
         %18 = OpTypeInt 32 1
         %19 = OpConstant %18 0
         %20 = OpTypePointer Uniform %10
         %24 = OpTypePointer Private %7
         %27 = OpTypePointer Private %6
         %30 = OpConstant %18 1
        %132 = OpTypePointer Function %10
        %135 = OpTypePointer Uniform %7
        %145 = OpTypePointer Function %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %133 = OpVariable %132 Function
         %21 = OpAccessChain %20 %17 %19
               OpCopyMemory %12 %21 Aligned 16
               OpCopyMemory %133 %12 Volatile
        %136 = OpAccessChain %135 %17 %30
        %138 = OpAccessChain %24 %12 %19
               OpCopyMemory %138 %136 None
        %146 = OpAccessChain %145 %133 %30
        %147 = OpLoad %7 %146 Volatile|Nontemporal|Aligned 16
        %148 = OpAccessChain %24 %12 %19
               OpStore %148 %147 Nontemporal
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
  FAIL(); // Remove once test is implemented
}

TEST(TransformationSetMemoryOperandsMaskTest, Spirv14) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %12 %17
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %7 "Point3D"
               OpMemberName %7 0 "x"
               OpMemberName %7 1 "y"
               OpMemberName %7 2 "z"
               OpName %12 "global_points"
               OpName %15 "block"
               OpMemberName %15 0 "in_points"
               OpMemberName %15 1 "in_point"
               OpName %17 ""
               OpName %133 "local_points"
               OpMemberDecorate %7 0 Offset 0
               OpMemberDecorate %7 1 Offset 4
               OpMemberDecorate %7 2 Offset 8
               OpDecorate %10 ArrayStride 16
               OpMemberDecorate %15 0 Offset 0
               OpMemberDecorate %15 1 Offset 192
               OpDecorate %15 Block
               OpDecorate %17 DescriptorSet 0
               OpDecorate %17 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeStruct %6 %6 %6
          %8 = OpTypeInt 32 0
          %9 = OpConstant %8 12
         %10 = OpTypeArray %7 %9
         %11 = OpTypePointer Private %10
         %12 = OpVariable %11 Private
         %15 = OpTypeStruct %10 %7
         %16 = OpTypePointer Uniform %15
         %17 = OpVariable %16 Uniform
         %18 = OpTypeInt 32 1
         %19 = OpConstant %18 0
         %20 = OpTypePointer Uniform %10
         %24 = OpTypePointer Private %7
         %27 = OpTypePointer Private %6
         %30 = OpConstant %18 1
        %132 = OpTypePointer Function %10
        %135 = OpTypePointer Uniform %7
        %145 = OpTypePointer Function %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %133 = OpVariable %132 Function
         %21 = OpAccessChain %20 %17 %19
               OpCopyMemory %12 %21 Aligned 16 Nontemporal|Aligned 16
               OpCopyMemory %133 %12 Volatile
        %136 = OpAccessChain %135 %17 %30
        %138 = OpAccessChain %24 %12 %19
               OpCopyMemory %138 %136 None Aligned 16
        %146 = OpAccessChain %145 %133 %30
        %147 = OpLoad %7 %146 Volatile|Nontemporal|Aligned 16
        %148 = OpAccessChain %24 %12 %19
               OpStore %148 %147 Nontemporal
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  // TODO - add test content

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %12 %17
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %7 "Point3D"
               OpMemberName %7 0 "x"
               OpMemberName %7 1 "y"
               OpMemberName %7 2 "z"
               OpName %12 "global_points"
               OpName %15 "block"
               OpMemberName %15 0 "in_points"
               OpMemberName %15 1 "in_point"
               OpName %17 ""
               OpName %133 "local_points"
               OpMemberDecorate %7 0 Offset 0
               OpMemberDecorate %7 1 Offset 4
               OpMemberDecorate %7 2 Offset 8
               OpDecorate %10 ArrayStride 16
               OpMemberDecorate %15 0 Offset 0
               OpMemberDecorate %15 1 Offset 192
               OpDecorate %15 Block
               OpDecorate %17 DescriptorSet 0
               OpDecorate %17 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeStruct %6 %6 %6
          %8 = OpTypeInt 32 0
          %9 = OpConstant %8 12
         %10 = OpTypeArray %7 %9
         %11 = OpTypePointer Private %10
         %12 = OpVariable %11 Private
         %15 = OpTypeStruct %10 %7
         %16 = OpTypePointer Uniform %15
         %17 = OpVariable %16 Uniform
         %18 = OpTypeInt 32 1
         %19 = OpConstant %18 0
         %20 = OpTypePointer Uniform %10
         %24 = OpTypePointer Private %7
         %27 = OpTypePointer Private %6
         %30 = OpConstant %18 1
        %132 = OpTypePointer Function %10
        %135 = OpTypePointer Uniform %7
        %145 = OpTypePointer Function %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %133 = OpVariable %132 Function
         %21 = OpAccessChain %20 %17 %19
               OpCopyMemory %12 %21 Aligned 16 Nontemporal|Aligned 16
               OpCopyMemory %133 %12 Volatile
        %136 = OpAccessChain %135 %17 %30
        %138 = OpAccessChain %24 %12 %19
               OpCopyMemory %138 %136 None Aligned 16
        %146 = OpAccessChain %145 %133 %30
        %147 = OpLoad %7 %146 Volatile|Nontemporal|Aligned 16
        %148 = OpAccessChain %24 %12 %19
               OpStore %148 %147 Nontemporal
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
  FAIL(); // Remove once test is implemented
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
