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

using VolatileSpreadTest = PassTest<::testing::Test>;

TEST_F(VolatileSpreadTest, SpreadVolatileForHelperInvocation) {
  const std::string text =
      R"(
OpCapability Shader
OpCapability DemoteToHelperInvocation
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %var
OpExecutionMode %main OriginUpperLeft

; CHECK: OpDecorate [[var:%\w+]] BuiltIn HelperInvocation
; CHECK: OpDecorate [[var]] Volatile
OpDecorate %var BuiltIn HelperInvocation

%bool = OpTypeBool
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%_ptr_Input_bool = OpTypePointer Input %bool
%var = OpVariable %_ptr_Input_bool Input
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpDemoteToHelperInvocation
OpReturn
OpFunctionEnd
)";

  SetTargetEnv(SPV_ENV_UNIVERSAL_1_6);
  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

TEST_F(VolatileSpreadTest, MultipleExecutionModel) {
  const std::string text =
      R"(
OpCapability RuntimeDescriptorArray
OpCapability RayTracingKHR
OpCapability SubgroupBallotKHR
OpExtension "SPV_EXT_descriptor_indexing"
OpExtension "SPV_KHR_ray_tracing"
OpExtension "SPV_KHR_shader_ballot"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint RayGenerationKHR %RayGeneration "RayGeneration" %var
OpEntryPoint GLCompute %compute "Compute" %gl_LocalInvocationIndex
OpExecutionMode %compute LocalSize 16 16 1
OpSource GLSL 460
OpSourceExtension "GL_EXT_nonuniform_qualifier"
OpSourceExtension "GL_KHR_ray_tracing"
OpName %RayGeneration "RayGeneration"
OpName %StorageBuffer "StorageBuffer"
OpMemberName %StorageBuffer 0 "index"
OpMemberName %StorageBuffer 1 "red"
OpName %sbo "sbo"
OpName %images "images"
OpMemberDecorate %StorageBuffer 0 Offset 0
OpMemberDecorate %StorageBuffer 1 Offset 4
OpDecorate %gl_LocalInvocationIndex BuiltIn LocalInvocationIndex
OpDecorate %StorageBuffer BufferBlock
OpDecorate %sbo DescriptorSet 0
OpDecorate %sbo Binding 0
OpDecorate %images DescriptorSet 0
OpDecorate %images Binding 1
OpDecorate %images NonWritable

; CHECK:     OpEntryPoint RayGenerationNV {{%\w+}} "RayGeneration" [[var:%\w+]]
; CHECK:     OpDecorate [[var]] BuiltIn SubgroupSize
; CHECK:     OpDecorate [[var]] Volatile
; CHECK-NOT: OpDecorate {{%\w+}} Volatile
OpDecorate %var BuiltIn SubgroupSize

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
%uint_1 = OpConstant %uint 1
%_ptr_Uniform_float = OpTypePointer Uniform %float
%gl_LocalInvocationIndex = OpVariable %_ptr_Input_uint Input
%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%shared = OpVariable %_ptr_Workgroup_uint Workgroup

%RayGeneration = OpFunction %void None %3
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

%compute = OpFunction %void None %3
%66 = OpLabel
%62 = OpLoad %uint %gl_LocalInvocationIndex
%61 = OpAtomicIAdd %uint %shared %uint_1 %uint_0 %62
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

TEST_F(VolatileSpreadTest, VarUsedInMultipleEntryPoints) {
  const std::string text =
      R"(
OpCapability RuntimeDescriptorArray
OpCapability RayTracingKHR
OpCapability SubgroupBallotKHR
OpExtension "SPV_EXT_descriptor_indexing"
OpExtension "SPV_KHR_ray_tracing"
OpExtension "SPV_KHR_shader_ballot"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint RayGenerationKHR %RayGeneration "RayGeneration" %var
OpEntryPoint ClosestHitKHR %ClosestHit "ClosestHit" %var
OpSource GLSL 460
OpSourceExtension "GL_EXT_nonuniform_qualifier"
OpSourceExtension "GL_KHR_ray_tracing"
OpName %RayGeneration "RayGeneration"
OpName %ClosestHit "ClosestHit"
OpName %StorageBuffer "StorageBuffer"
OpMemberName %StorageBuffer 0 "index"
OpMemberName %StorageBuffer 1 "red"
OpName %sbo "sbo"
OpName %images "images"
OpMemberDecorate %StorageBuffer 0 Offset 0
OpMemberDecorate %StorageBuffer 1 Offset 4
OpDecorate %gl_LocalInvocationIndex BuiltIn LocalInvocationIndex
OpDecorate %StorageBuffer BufferBlock
OpDecorate %sbo DescriptorSet 0
OpDecorate %sbo Binding 0
OpDecorate %images DescriptorSet 0
OpDecorate %images Binding 1
OpDecorate %images NonWritable

; CHECK:     OpEntryPoint RayGenerationNV {{%\w+}} "RayGeneration" [[var:%\w+]]
; CHECK:     OpEntryPoint ClosestHitNV {{%\w+}} "ClosestHit" [[var]]
; CHECK:     OpDecorate [[var]] BuiltIn SubgroupSize
; CHECK:     OpDecorate [[var]] Volatile
; CHECK-NOT: OpDecorate {{%\w+}} Volatile
OpDecorate %var BuiltIn SubgroupSize

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
%uint_1 = OpConstant %uint 1
%_ptr_Uniform_float = OpTypePointer Uniform %float
%gl_LocalInvocationIndex = OpVariable %_ptr_Input_uint Input
%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%shared = OpVariable %_ptr_Workgroup_uint Workgroup

%RayGeneration = OpFunction %void None %3
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

%ClosestHit = OpFunction %void None %3
%45 = OpLabel
%49 = OpAccessChain %_ptr_Uniform_uint %sbo %int_0
%40 = OpLoad %uint %49
%42 = OpAccessChain %_ptr_UniformConstant_13 %images %40
%43 = OpLoad %13 %42
%47 = OpImageRead %v4float %43 %25
%59 = OpCompositeExtract %float %47 0
%51 = OpAccessChain %_ptr_Uniform_float %sbo %int_1
OpStore %51 %59
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

class VolatileSpreadErrorTest : public ::testing::Test {
 public:
  VolatileSpreadErrorTest()
      : consumer_([this](spv_message_level_t level, const char*,
                         const spv_position_t& position, const char* message) {
          if (!error_message_.empty()) error_message_ += "\n";
          switch (level) {
            case SPV_MSG_FATAL:
            case SPV_MSG_INTERNAL_ERROR:
            case SPV_MSG_ERROR:
              error_message_ += "ERROR";
              break;
            case SPV_MSG_WARNING:
              error_message_ += "WARNING";
              break;
            case SPV_MSG_INFO:
              error_message_ += "INFO";
              break;
            case SPV_MSG_DEBUG:
              error_message_ += "DEBUG";
              break;
          }
          error_message_ +=
              ": " + std::to_string(position.index) + ": " + message;
        }) {}

  Pass::Status RunPass(const std::string& text) {
    std::unique_ptr<IRContext> context_ =
        spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_2, consumer_, text);
    if (!context_.get()) return Pass::Status::Failure;

    PassManager manager;
    manager.SetMessageConsumer(consumer_);
    manager.AddPass<SpreadVolatileSemantics>();

    return manager.Run(context_.get());
  }

  std::string GetErrorMessage() const { return error_message_; }

  void TearDown() override { error_message_.clear(); }

 private:
  spvtools::MessageConsumer consumer_;
  std::string error_message_;
};

TEST_F(VolatileSpreadErrorTest, VarUsedInMultipleExecutionModelError) {
  const std::string text =
      R"(
OpCapability RuntimeDescriptorArray
OpCapability RayTracingKHR
OpCapability SubgroupBallotKHR
OpExtension "SPV_EXT_descriptor_indexing"
OpExtension "SPV_KHR_ray_tracing"
OpExtension "SPV_KHR_shader_ballot"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint RayGenerationKHR %RayGeneration "RayGeneration" %var
OpEntryPoint GLCompute %compute "Compute" %gl_LocalInvocationIndex %var
OpExecutionMode %compute LocalSize 16 16 1
OpSource GLSL 460
OpSourceExtension "GL_EXT_nonuniform_qualifier"
OpSourceExtension "GL_KHR_ray_tracing"
OpName %RayGeneration "RayGeneration"
OpName %StorageBuffer "StorageBuffer"
OpMemberName %StorageBuffer 0 "index"
OpMemberName %StorageBuffer 1 "red"
OpName %sbo "sbo"
OpName %images "images"
OpMemberDecorate %StorageBuffer 0 Offset 0
OpMemberDecorate %StorageBuffer 1 Offset 4
OpDecorate %gl_LocalInvocationIndex BuiltIn LocalInvocationIndex
OpDecorate %StorageBuffer BufferBlock
OpDecorate %sbo DescriptorSet 0
OpDecorate %sbo Binding 0
OpDecorate %images DescriptorSet 0
OpDecorate %images Binding 1
OpDecorate %images NonWritable
OpDecorate %var BuiltIn SubgroupSize
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
%uint_1 = OpConstant %uint 1
%_ptr_Uniform_float = OpTypePointer Uniform %float
%gl_LocalInvocationIndex = OpVariable %_ptr_Input_uint Input
%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%shared = OpVariable %_ptr_Workgroup_uint Workgroup

%RayGeneration = OpFunction %void None %3
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

%compute = OpFunction %void None %3
%66 = OpLabel
%62 = OpLoad %uint %gl_LocalInvocationIndex
%63 = OpLoad %uint %var
%64 = OpIAdd %uint %62 %63
%61 = OpAtomicIAdd %uint %shared %uint_1 %uint_0 %64
OpReturn
OpFunctionEnd
)";

  EXPECT_EQ(RunPass(text), Pass::Status::Failure);
  const char expected_error[] =
      "ERROR: 0: Variable is a target for Volatile semantics for an entry "
      "point, but it is not for another entry point";
  EXPECT_STREQ(GetErrorMessage().substr(0, sizeof(expected_error) - 1).c_str(),
               expected_error);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
