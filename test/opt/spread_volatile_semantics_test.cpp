// Copyright (c) 2022 Google LLC
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
  const bool use_v4uint;
};

using AddVolatileDecorationTest =
    PassTest<::testing::TestWithParam<ExecutionModelAndBuiltIn>>;

TEST_P(AddVolatileDecorationTest, InMain) {
  const auto& tc = GetParam();
  const std::string execution_model(tc.execution_model);
  const std::string built_in(tc.built_in);
  const std::string var_type =
      tc.use_v4uint ? "%_ptr_Input_v4uint" : "%_ptr_Input_uint";
  const std::string var_load_type = tc.use_v4uint ? "%v4uint" : "%uint";

  const std::string text =
      std::string(R"(OpCapability RuntimeDescriptorArray
OpCapability RayTracingKHR
OpCapability SubgroupBallotKHR
OpExtension "SPV_EXT_descriptor_indexing"
OpExtension "SPV_KHR_ray_tracing"
OpExtension "SPV_KHR_shader_ballot"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint )") +
      execution_model + std::string(R"( %main "main" %var
OpSource GLSL 460
OpSourceExtension "GL_EXT_nonuniform_qualifier"
OpSourceExtension "GL_KHR_ray_tracing"
OpName %main "main"
OpName %fn "fn"
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
OpDecorate %var BuiltIn )") + built_in + std::string(R"(
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
%v4uint = OpTypeVector %uint 4
%_ptr_Input_v4uint = OpTypePointer Input %v4uint
%var = OpVariable )") +
      var_type + std::string(R"( Input
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
%load = OpLoad )") + var_load_type + std::string(R"( %var
%22 = OpAccessChain %_ptr_UniformConstant_13 %images %20
%23 = OpLoad %13 %22
%27 = OpImageRead %v4float %23 %25
%29 = OpCompositeExtract %float %27 0
%31 = OpAccessChain %_ptr_Uniform_float %sbo %int_1
OpStore %31 %29
%32 = OpFunctionCall %void %fn
OpReturn
OpFunctionEnd
%fn = OpFunction %void None %3
%33 = OpLabel
OpReturn
OpFunctionEnd
)");

  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

INSTANTIATE_TEST_SUITE_P(
    AddVolatileDecoration, AddVolatileDecorationTest,
    ::testing::ValuesIn(std::vector<ExecutionModelAndBuiltIn>{
        {"RayGenerationKHR", "SubgroupSize", false},
        {"RayGenerationKHR", "SubgroupLocalInvocationId", false},
        {"RayGenerationKHR", "SubgroupEqMask", true},
        {"ClosestHitKHR", "SubgroupLocalInvocationId", true},
        {"IntersectionKHR", "SubgroupEqMask", true},
        {"MissKHR", "SubgroupGeMask", true},
        {"CallableKHR", "SubgroupGtMask", true},
        {"RayGenerationKHR", "SubgroupLeMask", true},
    }));

using SetLoadVolatileTest =
    PassTest<::testing::TestWithParam<ExecutionModelAndBuiltIn>>;

TEST_P(SetLoadVolatileTest, InMain) {
  const auto& tc = GetParam();
  const std::string execution_model(tc.execution_model);
  const std::string built_in(tc.built_in);

  const std::string var_type =
      tc.use_v4uint ? "%_ptr_Input_v4uint" : "%_ptr_Input_uint";
  const std::string var_value = tc.use_v4uint ? std::string(R"(
; CHECK: [[ptr:%\w+]] = OpAccessChain %_ptr_Input_uint [[var]] %int_0
; CHECK: OpLoad {{%\w+}} [[ptr]] Volatile
%ptr = OpAccessChain %_ptr_Input_uint %var %int_0
%var_value = OpLoad %uint %ptr)")
                                              : std::string(R"(
; CHECK: OpLoad {{%\w+}} [[var]] Volatile
%var_value = OpLoad %uint %var)");

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
%v4uint = OpTypeVector %uint 4
%_ptr_Input_v4uint = OpTypePointer Input %v4uint
%var = OpVariable )") + var_type +
                           std::string(R"( Input
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
)") + var_value + std::string(R"(
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
        {"RayGenerationKHR", "SubgroupSize", false},
        {"RayGenerationKHR", "SubgroupLocalInvocationId", false},
        {"RayGenerationKHR", "SubgroupEqMask", true},
        {"ClosestHitKHR", "SubgroupLocalInvocationId", true},
        {"IntersectionKHR", "SubgroupEqMask", true},
        {"MissKHR", "SubgroupGeMask", true},
        {"CallableKHR", "SubgroupGtMask", true},
        {"RayGenerationKHR", "SubgroupLeMask", true},
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
%load = OpLoad %bool %var
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
%20 = OpLoad %uint %var
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
%20 = OpLoad %uint %var
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
%40 = OpLoad %uint %var
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

class VolatileSpreadErrorTest : public PassTest<::testing::Test> {
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
%20 = OpLoad %uint %var
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

TEST_F(VolatileSpreadErrorTest,
       VarUsedInMultipleReverseOrderExecutionModelError) {
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
OpEntryPoint GLCompute %compute "Compute" %gl_LocalInvocationIndex %var
OpEntryPoint RayGenerationKHR %RayGeneration "RayGeneration" %var
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
%20 = OpLoad %uint %var
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

TEST_F(VolatileSpreadErrorTest, FunctionNotInlined) {
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

%NotInlined = OpFunction %void None %3
%32 = OpLabel
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

  EXPECT_EQ(RunPass(text), Pass::Status::SuccessWithoutChange);
}

TEST_F(VolatileSpreadErrorTest, VarNotUsedInEntryPointForVolatile) {
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
%63 = OpLoad %uint %var
%64 = OpIAdd %uint %62 %63
%61 = OpAtomicIAdd %uint %shared %uint_1 %uint_0 %64
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

TEST_F(VolatileSpreadTest, RecursivelySpreadVolatile) {
  const std::string text =
      R"(
OpCapability RuntimeDescriptorArray
OpCapability RayTracingKHR
OpCapability SubgroupBallotKHR
OpCapability VulkanMemoryModel
OpExtension "SPV_KHR_vulkan_memory_model"
OpExtension "SPV_EXT_descriptor_indexing"
OpExtension "SPV_KHR_ray_tracing"
OpExtension "SPV_KHR_shader_ballot"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical Vulkan
OpEntryPoint RayGenerationKHR %RayGeneration "RayGeneration" %var0 %var1
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
OpDecorate %StorageBuffer BufferBlock
OpDecorate %sbo DescriptorSet 0
OpDecorate %sbo Binding 0
OpDecorate %images DescriptorSet 0
OpDecorate %images Binding 1
OpDecorate %images NonWritable

; CHECK: OpDecorate [[var0:%\w+]] BuiltIn SubgroupEqMask
; CHECK: OpDecorate [[var1:%\w+]] BuiltIn SubgroupGeMask
OpDecorate %var0 BuiltIn SubgroupEqMask
OpDecorate %var1 BuiltIn SubgroupGeMask

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
%v4uint = OpTypeVector %uint 4
%_ptr_Input_v4uint = OpTypePointer Input %v4uint
%_ptr_Input_uint = OpTypePointer Input %uint
%var0 = OpVariable %_ptr_Input_v4uint Input
%var1 = OpVariable %_ptr_Input_v4uint Input
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_UniformConstant_13 = OpTypePointer UniformConstant %13
%v2int = OpTypeVector %int 2
%25 = OpConstantComposite %v2int %int_0 %int_0
%v4float = OpTypeVector %float 4
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%_ptr_Uniform_float = OpTypePointer Uniform %float

%RayGeneration = OpFunction %void None %3
%5 = OpLabel

; CHECK: [[ptr0:%\w+]] = OpAccessChain %_ptr_Input_uint [[var0]] %int_0
; CHECK: OpLoad {{%\w+}} [[ptr0]] Volatile
%19 = OpAccessChain %_ptr_Input_uint %var0 %int_0
%20 = OpLoad %uint %19

%22 = OpAccessChain %_ptr_UniformConstant_13 %images %20
%23 = OpLoad %13 %22
%27 = OpImageRead %v4float %23 %25
%29 = OpCompositeExtract %float %27 0
%31 = OpAccessChain %_ptr_Uniform_float %sbo %uint_1

; CHECK: OpLoad {{%\w+}} [[ptr0]] Volatile
%24 = OpLoad %uint %19

; CHECK: [[var2:%\w+]] = OpCopyObject %_ptr_Input_v4uint [[var0]]
; CHECK: [[ptr2:%\w+]] = OpAccessChain %_ptr_Input_uint [[var2]] %int_1
; CHECK: OpLoad {{%\w+}} [[ptr2]] Volatile
%18 = OpCopyObject %_ptr_Input_v4uint %var0
%21 = OpAccessChain %_ptr_Input_uint %18 %int_1
%26 = OpLoad %uint %21

%28 = OpIAdd %uint %24 %26
%30 = OpConvertUToF %float %28

; CHECK: [[ptr1:%\w+]] = OpAccessChain %_ptr_Input_uint [[var1]] %int_1
; CHECK: OpLoad {{%\w+}} [[ptr1]] Volatile
%32 = OpAccessChain %_ptr_Input_uint %var1 %int_1
%33 = OpLoad %uint %32

%34 = OpConvertUToF %float %33
%35 = OpFAdd %float %34 %30
%36 = OpFAdd %float %35 %29
OpStore %31 %36
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

TEST_F(VolatileSpreadTest, SpreadVolatileOnlyForTargetEntryPoints) {
  const std::string text =
      R"(
OpCapability RuntimeDescriptorArray
OpCapability RayTracingKHR
OpCapability SubgroupBallotKHR
OpCapability VulkanMemoryModel
OpCapability VulkanMemoryModelDeviceScopeKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpExtension "SPV_EXT_descriptor_indexing"
OpExtension "SPV_KHR_ray_tracing"
OpExtension "SPV_KHR_shader_ballot"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical Vulkan
OpEntryPoint RayGenerationKHR %RayGeneration "RayGeneration" %var0 %var1
OpEntryPoint GLCompute %compute "Compute" %var0 %var1
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
OpDecorate %StorageBuffer BufferBlock
OpDecorate %sbo DescriptorSet 0
OpDecorate %sbo Binding 0
OpDecorate %images DescriptorSet 0
OpDecorate %images Binding 1
OpDecorate %images NonWritable

; CHECK: OpDecorate [[var0:%\w+]] BuiltIn SubgroupEqMask
; CHECK: OpDecorate [[var1:%\w+]] BuiltIn SubgroupGeMask
OpDecorate %var0 BuiltIn SubgroupEqMask
OpDecorate %var1 BuiltIn SubgroupGeMask

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
%v4uint = OpTypeVector %uint 4
%_ptr_Input_v4uint = OpTypePointer Input %v4uint
%_ptr_Input_uint = OpTypePointer Input %uint
%var0 = OpVariable %_ptr_Input_v4uint Input
%var1 = OpVariable %_ptr_Input_v4uint Input
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%_ptr_UniformConstant_13 = OpTypePointer UniformConstant %13
%v2int = OpTypeVector %int 2
%25 = OpConstantComposite %v2int %int_0 %int_0
%v4float = OpTypeVector %float 4
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%shared = OpVariable %_ptr_Workgroup_uint Workgroup

%RayGeneration = OpFunction %void None %3
%5 = OpLabel

; CHECK: [[ptr0:%\w+]] = OpAccessChain %_ptr_Input_uint [[var0]] %int_0
; CHECK: OpLoad {{%\w+}} [[ptr0]] Volatile
%19 = OpAccessChain %_ptr_Input_uint %var0 %int_0
%20 = OpLoad %uint %19

%22 = OpAccessChain %_ptr_UniformConstant_13 %images %20
%23 = OpLoad %13 %22
%27 = OpImageRead %v4float %23 %25
%29 = OpCompositeExtract %float %27 0
%31 = OpAccessChain %_ptr_Uniform_float %sbo %uint_1

; CHECK: OpLoad {{%\w+}} [[ptr0]] Volatile
%24 = OpLoad %uint %19

; CHECK: [[var2:%\w+]] = OpCopyObject %_ptr_Input_v4uint [[var0]]
; CHECK: [[ptr2:%\w+]] = OpAccessChain %_ptr_Input_uint [[var2]] %int_1
; CHECK: OpLoad {{%\w+}} [[ptr2]] Volatile
%18 = OpCopyObject %_ptr_Input_v4uint %var0
%21 = OpAccessChain %_ptr_Input_uint %18 %int_1
%26 = OpLoad %uint %21

%28 = OpIAdd %uint %24 %26
%30 = OpConvertUToF %float %28

; CHECK: [[ptr1:%\w+]] = OpAccessChain %_ptr_Input_uint [[var1]] %int_1
; CHECK: OpLoad {{%\w+}} [[ptr1]] Volatile
%32 = OpAccessChain %_ptr_Input_uint %var1 %int_1
%33 = OpLoad %uint %32

%34 = OpConvertUToF %float %33
%35 = OpFAdd %float %34 %30
%36 = OpFAdd %float %35 %29
OpStore %31 %36
OpReturn
OpFunctionEnd

%compute = OpFunction %void None %3
%66 = OpLabel

; CHECK-NOT: OpLoad {{%\w+}} {{%\w+}} Volatile
%62 = OpLoad %v4uint %var0
%63 = OpLoad %v4uint %var1
%64 = OpIAdd %v4uint %62 %63
%65 = OpCompositeExtract %uint %64 0
%61 = OpAtomicIAdd %uint %shared %uint_1 %uint_0 %65
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

TEST_F(VolatileSpreadTest, SkipIfItHasNoExecutionModel) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%4 = OpFunction %2 None %3
%5 = OpLabel
OpReturn
OpFunctionEnd
)";

  Pass::Status status;
  std::tie(std::ignore, status) =
      SinglePassRunToBinary<SpreadVolatileSemantics>(text,
                                                     /* skip_nop = */ false);
  EXPECT_EQ(status, Pass::Status::SuccessWithoutChange);
}

TEST_F(VolatileSpreadTest, NoInlinedfuncCalls) {
  const std::string text = R"(
OpCapability RayTracingNV
OpCapability VulkanMemoryModel
OpCapability GroupNonUniform
OpCapability RayQueryKHR
OpExtension "SPV_NV_ray_tracing"
OpExtension "SPV_KHR_vulkan_memory_model"
OpExtension "SPV_KHR_ray_query"
OpMemoryModel Logical Vulkan
OpEntryPoint RayGenerationNV %main "main" %2 %3 %SubgroupSize
OpSource HLSL 630
OpName %accelerationStructureNV "accelerationStructureNV"
OpName %rs "rs"
OpName %CallData "CallData"
OpMemberName %CallData 0 "data"
OpName %myCallData "myCallData"
OpName %Payload "Payload"
OpMemberName %Payload 0 "color"
OpName %myPayload "myPayload"
OpName %main "main"
OpName %src_main "src.main"
OpName %bb_entry "bb.entry"
OpName %a "a"
OpName %b "b"
OpName %myCallData_0 "myCallData"
OpName %func0 "func0"
OpName %bb_entry_0 "bb.entry"
OpName %func2 "func2"
OpName %bb_entry_1 "bb.entry"
OpName %myPayload_0 "myPayload"
OpName %RayDesc "RayDesc"
OpMemberName %RayDesc 0 "Origin"
OpMemberName %RayDesc 1 "TMin"
OpMemberName %RayDesc 2 "Direction"
OpMemberName %RayDesc 3 "TMax"
OpName %rayDesc "rayDesc"
OpName %func1 "func1"
OpName %bb_entry_2 "bb.entry"
OpDecorate %2 BuiltIn LaunchIdNV
OpDecorate %3 BuiltIn LaunchSizeNV
OpDecorate %myCallData Location 0
OpDecorate %myPayload Location 0
OpDecorate %SubgroupSize BuiltIn SubgroupSize
OpDecorate %rs DescriptorSet 0
OpDecorate %rs Binding 0
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%v3float = OpTypeVector %float 3
%13 = OpConstantComposite %v3float %float_0 %float_0 %float_0
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float_n1 = OpConstant %float -1
%17 = OpConstantComposite %v3float %float_0 %float_0 %float_n1
%int_2 = OpConstant %int 2
%int_1 = OpConstant %int 1
%float_1000 = OpConstant %float 1000
%int_3 = OpConstant %int 3
%uint_1 = OpConstant %uint 1
%accelerationStructureNV = OpTypeAccelerationStructureKHR
%_ptr_UniformConstant_accelerationStructureNV = OpTypePointer UniformConstant %accelerationStructureNV
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%v4float = OpTypeVector %float 4
%CallData = OpTypeStruct %v4float
%_ptr_CallableDataNV_CallData = OpTypePointer CallableDataNV %CallData
%Payload = OpTypeStruct %v4float
%_ptr_RayPayloadNV_Payload = OpTypePointer RayPayloadNV %Payload
%_ptr_Input_uint = OpTypePointer Input %uint
%void = OpTypeVoid
%34 = OpTypeFunction %void
%_ptr_Function_v3uint = OpTypePointer Function %v3uint
%_ptr_Function_CallData = OpTypePointer Function %CallData
%_ptr_Function_Payload = OpTypePointer Function %Payload
%RayDesc = OpTypeStruct %v3float %float %v3float %float
%_ptr_Function_RayDesc = OpTypePointer Function %RayDesc
%_ptr_Function_v3float = OpTypePointer Function %v3float
%_ptr_Function_float = OpTypePointer Function %float
%rs = OpVariable %_ptr_UniformConstant_accelerationStructureNV UniformConstant
%2 = OpVariable %_ptr_Input_v3uint Input
%3 = OpVariable %_ptr_Input_v3uint Input
%myCallData = OpVariable %_ptr_CallableDataNV_CallData CallableDataNV
%myPayload = OpVariable %_ptr_RayPayloadNV_Payload RayPayloadNV
%SubgroupSize = OpVariable %_ptr_Input_uint Input
%main = OpFunction %void None %34
%35 = OpLabel
%36 = OpFunctionCall %void %src_main
OpReturn
OpFunctionEnd
%src_main = OpFunction %void None %34
%bb_entry = OpLabel
%a = OpVariable %_ptr_Function_v3uint Function
%b = OpVariable %_ptr_Function_v3uint Function
%myCallData_0 = OpVariable %_ptr_Function_CallData Function
%44 = OpLoad %v3uint %2
OpStore %a %44
%45 = OpLoad %v3uint %3
OpStore %b %45
%46 = OpFunctionCall %void %func0
%48 = OpCompositeConstruct %v4float %float_0 %float_0 %float_0 %float_0
%49 = OpCompositeConstruct %CallData %48
OpStore %myCallData_0 %49
%50 = OpLoad %CallData %myCallData_0
OpStore %myCallData %50
OpExecuteCallableNV %uint_0 %uint_0
%52 = OpLoad %CallData %myCallData
OpStore %myCallData_0 %52
OpReturn
OpFunctionEnd
%func0 = OpFunction %void DontInline %34
%bb_entry_0 = OpLabel
%54 = OpFunctionCall %void %func2
%56 = OpFunctionCall %void %func1
OpReturn
OpFunctionEnd
%func2 = OpFunction %void DontInline %34
%bb_entry_1 = OpLabel
%myPayload_0 = OpVariable %_ptr_Function_Payload Function
%rayDesc = OpVariable %_ptr_Function_RayDesc Function
%64 = OpCompositeConstruct %v4float %float_0 %float_0 %float_0 %float_0
%65 = OpCompositeConstruct %Payload %64
OpStore %myPayload_0 %65
%67 = OpAccessChain %_ptr_Function_v3float %rayDesc %int_0
%68 = OpAccessChain %_ptr_Function_v3float %rayDesc %int_2
OpStore %68 %17
%70 = OpAccessChain %_ptr_Function_float %rayDesc %int_1
OpStore %70 %float_0
%71 = OpAccessChain %_ptr_Function_float %rayDesc %int_3
OpStore %71 %float_1000
%72 = OpLoad %RayDesc %rayDesc
%73 = OpCompositeExtract %v3float %72 0
%74 = OpCompositeExtract %float %72 1
%75 = OpCompositeExtract %v3float %72 2
%76 = OpCompositeExtract %float %72 3
%77 = OpLoad %Payload %myPayload_0
OpStore %myPayload %77
%78 = OpLoad %accelerationStructureNV %rs
; CHECK: {{%\w+}} = OpLoad %uint %SubgroupSize Volatile
%79 = OpLoad %uint %SubgroupSize
OpTraceNV %78 %uint_0 %79 %uint_0 %uint_1 %uint_0 %73 %74 %75 %76 %uint_0
%81 = OpLoad %Payload %myPayload
OpStore %myPayload_0 %81
OpReturn
OpFunctionEnd
%func1 = OpFunction %void DontInline %34
%bb_entry_2 = OpLabel
OpReturn
OpFunctionEnd
)";
  SinglePassRunAndMatch<SpreadVolatileSemantics>(text, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
