// Copyright (c) 2021 Google LLC
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

#include <string>

#include "gmock/gmock.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

struct ExecutionModelAndBuiltIn {
  const char* execution_model;
  const char* built_in;
};

using AddVolatileDecorationTest =
    PassTest<::testing::TestWithParam<ExecutionModelAndBuiltIn>>;

TEST_P(AddVolatileDecorationTest, InMain) {
  const auto& tc = GetParam();
  const std::string execution_model(tc.execution_model);
  const std::string built_in(tc.built_in);

  const std::string text = std::string(R"(OpCapability RuntimeDescriptorArray
OpCapability RayTracingKHR
OpCapability SubgroupBallotKHR
OpExtension "SPV_EXT_descriptor_indexing"
OpExtension "SPV_KHR_ray_tracing"
OpExtension "SPV_KHR_shader_ballot"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint )") + execution_model +
                           std::string(R"( %main "main" %var
OpSource GLSL 460
OpSourceExtension "GL_EXT_nonuniform_qualifier"
OpSourceExtension "GL_KHR_ray_tracing"
OpName %main "main"
OpName %StorageBuffer "StorageBuffer"
OpMemberName %StorageBuffer 0 "index"
OpMemberName %StorageBuffer 1 "red"
OpName %sbo "sbo"
OpName %images "images"
OpMemberDecorate %StorageBuffer 0 Offset 0
OpMemberDecorate %StorageBuffer 1 Offset 4
OpDecorate %StorageBuffer BufferBlock
OpDecorate %sbo DescriptorSet 0
OpDecorate %sbo Binding 0
OpDecorate %images DescriptorSet 0
OpDecorate %images Binding 1
OpDecorate %images NonWritable
)") + std::string(R"(
; CHECK: OpDecorate [[var:%\w+]] BuiltIn )") +
                           built_in + std::string(R"(
; CHECK: OpDecorate [[var]] Volatile
OpDecorate %var BuiltIn )") + built_in +
                           std::string(R"(
%void = OpTypeVoid
%3 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%StorageBuffer = OpTypeStruct %uint %float
%_ptr_Uniform_StorageBuffer = OpTypePointer Uniform %StorageBuffer
%sbo = OpVariable %_ptr_Uniform_StorageBuffer Uniform
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%13 = OpTypeImage %float 2D 0 0 0 2 Rgba32f
%_runtimearr_13 = OpTypeRuntimeArray %13
%_ptr_UniformConstant__runtimearr_13 = OpTypePointer UniformConstant %_runtimearr_13
%images = OpVariable %_ptr_UniformConstant__runtimearr_13 UniformConstant
%_ptr_Input_uint = OpTypePointer Input %uint
%var = OpVariable %_ptr_Input_uint Input
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_UniformConstant_13 = OpTypePointer UniformConstant %13
%v2int = OpTypeVector %int 2
%25 = OpConstantComposite %v2int %int_0 %int_0
%v4float = OpTypeVector %float 4
%uint_0 = OpConstant %uint 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%main = OpFunction %void None %3
%5 = OpLabel
%19 = OpAccessChain %_ptr_Uniform_uint %sbo %int_0
%20 = OpLoad %uint %19
%22 = OpAccessChain %_ptr_UniformConstant_13 %images %20
%23 = OpLoad %13 %22
%27 = OpImageRead %v4float %23 %25
%29 = OpCompositeExtract %float %27 0
%31 = OpAccessChain %_ptr_Uniform_float %sbo %int_1
OpStore %31 %29
OpReturn
OpFunctionEnd
)");

  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

INSTANTIATE_TEST_SUITE_P(
    AddVolatileDecoration, AddVolatileDecorationTest,
    ::testing::ValuesIn(std::vector<ExecutionModelAndBuiltIn>{
        {"RayGenerationKHR", "SubgroupSize"},
        {"RayGenerationKHR", "SubgroupLocalInvocationId"},
        {"RayGenerationKHR", "SubgroupEqMask"},
        {"ClosestHitKHR", "SubgroupLocalInvocationId"},
        {"IntersectionKHR", "SubgroupEqMask"},
        {"MissKHR", "SubgroupGeMask"},
        {"CallableKHR", "SubgroupGtMask"},
        {"RayGenerationKHR", "SubgroupLeMask"},
    }));

using SetLoadVolatileTest =
    PassTest<::testing::TestWithParam<ExecutionModelAndBuiltIn>>;

TEST_P(SetLoadVolatileTest, InMain) {
  const auto& tc = GetParam();
  const std::string execution_model(tc.execution_model);
  const std::string built_in(tc.built_in);

  const std::string text = std::string(R"(OpCapability RuntimeDescriptorArray
OpCapability RayTracingKHR
OpCapability SubgroupBallotKHR
OpCapability VulkanMemoryModel
OpExtension "SPV_KHR_vulkan_memory_model"
OpExtension "SPV_EXT_descriptor_indexing"
OpExtension "SPV_KHR_ray_tracing"
OpExtension "SPV_KHR_shader_ballot"
OpMemoryModel Logical Vulkan
OpEntryPoint )") + execution_model +
                           std::string(R"( %main "main" %var
OpName %main "main"
OpName %StorageBuffer "StorageBuffer"
OpMemberName %StorageBuffer 0 "index"
OpMemberName %StorageBuffer 1 "red"
OpName %sbo "sbo"
OpName %images "images"
OpMemberDecorate %StorageBuffer 0 Offset 0
OpMemberDecorate %StorageBuffer 1 Offset 4
OpDecorate %StorageBuffer BufferBlock
OpDecorate %sbo DescriptorSet 0
OpDecorate %sbo Binding 0
OpDecorate %images DescriptorSet 0
OpDecorate %images Binding 1
OpDecorate %images NonWritable
)") + std::string(R"(
; CHECK: OpDecorate [[var:%\w+]] BuiltIn )") +
                           built_in + std::string(R"(
; CHECK: OpLoad {{%\w+}} [[var]] Volatile
OpDecorate %var BuiltIn )") + built_in +
                           std::string(R"(
%void = OpTypeVoid
%3 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%StorageBuffer = OpTypeStruct %uint %float
%_ptr_Uniform_StorageBuffer = OpTypePointer Uniform %StorageBuffer
%sbo = OpVariable %_ptr_Uniform_StorageBuffer Uniform
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%13 = OpTypeImage %float 2D 0 0 0 2 Rgba32f
%_runtimearr_13 = OpTypeRuntimeArray %13
%_ptr_UniformConstant__runtimearr_13 = OpTypePointer UniformConstant %_runtimearr_13
%images = OpVariable %_ptr_UniformConstant__runtimearr_13 UniformConstant
%_ptr_Input_uint = OpTypePointer Input %uint
%var = OpVariable %_ptr_Input_uint Input
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_UniformConstant_13 = OpTypePointer UniformConstant %13
%v2int = OpTypeVector %int 2
%25 = OpConstantComposite %v2int %int_0 %int_0
%v4float = OpTypeVector %float 4
%uint_0 = OpConstant %uint 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
%main = OpFunction %void None %3
%5 = OpLabel
%19 = OpAccessChain %_ptr_Uniform_uint %sbo %int_0
%20 = OpLoad %uint %19
%var_value = OpLoad %uint %var
%test = OpIAdd %uint %var_value %20
%22 = OpAccessChain %_ptr_UniformConstant_13 %images %test
%23 = OpLoad %13 %22
%27 = OpImageRead %v4float %23 %25
%29 = OpCompositeExtract %float %27 0
%31 = OpAccessChain %_ptr_Uniform_float %sbo %int_1
OpStore %31 %29
OpReturn
OpFunctionEnd
)");

  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

INSTANTIATE_TEST_SUITE_P(
    SetLoadVolatile, SetLoadVolatileTest,
    ::testing::ValuesIn(std::vector<ExecutionModelAndBuiltIn>{
        {"RayGenerationKHR", "SubgroupSize"},
        {"RayGenerationKHR", "SubgroupLocalInvocationId"},
        {"RayGenerationKHR", "SubgroupEqMask"},
        {"ClosestHitKHR", "SubgroupLocalInvocationId"},
        {"IntersectionKHR", "SubgroupEqMask"},
        {"MissKHR", "SubgroupGeMask"},
        {"CallableKHR", "SubgroupGtMask"},
        {"RayGenerationKHR", "SubgroupLeMask"},
    }));

}  // namespace
}  // namespace opt
}  // namespace spvtools
