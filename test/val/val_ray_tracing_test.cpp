// Copyright (c) 2022 The Khronos Group Inc.
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

// Tests ray tracing instructions from SPV_KHR_ray_tracing.

#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Values;

using ValidateRayTracing = spvtest::ValidateBase<bool>;

TEST_F(ValidateRayTracing, IgnoreIntersectionSuccess) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint AnyHitKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%main = OpFunction %void None %func
%label = OpLabel
OpIgnoreIntersectionKHR
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateRayTracing, IgnoreIntersectionExecutionModel) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint CallableKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%main = OpFunction %void None %func
%label = OpLabel
OpIgnoreIntersectionKHR
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpIgnoreIntersectionKHR requires AnyHitKHR execution model"));
}

TEST_F(ValidateRayTracing, TerminateRaySuccess) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint AnyHitKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%main = OpFunction %void None %func
%label = OpLabel
OpIgnoreIntersectionKHR
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateRayTracing, TerminateRayExecutionModel) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint MissKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%main = OpFunction %void None %func
%label = OpLabel
OpTerminateRayKHR
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpTerminateRayKHR requires AnyHitKHR execution model"));
}

TEST_F(ValidateRayTracing, ReportIntersectionRaySuccess) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint IntersectionKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%float = OpTypeFloat 32
%float_1 = OpConstant %float 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%bool = OpTypeBool
%main = OpFunction %void None %func
%label = OpLabel
%report = OpReportIntersectionKHR %bool %float_1 %uint_1
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateRayTracing, ReportIntersectionExecutionModel) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint MissKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%float = OpTypeFloat 32
%float_1 = OpConstant %float 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%bool = OpTypeBool
%main = OpFunction %void None %func
%label = OpLabel
%report = OpReportIntersectionKHR %bool %float_1 %uint_1
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpReportIntersectionKHR requires IntersectionKHR execution model"));
}

TEST_F(ValidateRayTracing, ReportIntersectionReturnType) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint IntersectionKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%float = OpTypeFloat 32
%float_1 = OpConstant %float 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%main = OpFunction %void None %func
%label = OpLabel
%report = OpReportIntersectionKHR %uint %float_1 %uint_1
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("expected Result Type to be bool scalar type"));
}

TEST_F(ValidateRayTracing, ReportIntersectionHit) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpCapability Float64
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint IntersectionKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%float64 = OpTypeFloat 64
%float64_1 = OpConstant %float64 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%bool = OpTypeBool
%main = OpFunction %void None %func
%label = OpLabel
%report = OpReportIntersectionKHR %bool %float64_1 %uint_1
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Hit must be a 32-bit int scalar"));
}

TEST_F(ValidateRayTracing, ReportIntersectionHitKind) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint IntersectionKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%float = OpTypeFloat 32
%float_1 = OpConstant %float 1
%sint = OpTypeInt 32 1
%sint_1 = OpConstant %sint 1
%bool = OpTypeBool
%main = OpFunction %void None %func
%label = OpLabel
%report = OpReportIntersectionKHR %bool %float_1 %sint_1
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Hit Kind must be a 32-bit unsigned int scalar"));
}

TEST_F(ValidateRayTracing, ExecuteCallableSuccess) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint CallableKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%data_ptr = OpTypePointer CallableDataKHR %int
%data = OpVariable %data_ptr CallableDataKHR
%inData_ptr = OpTypePointer IncomingCallableDataKHR %int
%inData = OpVariable %inData_ptr IncomingCallableDataKHR
%main = OpFunction %void None %func
%label = OpLabel
OpExecuteCallableKHR %uint_0 %data
OpExecuteCallableKHR %uint_0 %inData
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateRayTracing, ExecuteCallableExecutionModel) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint AnyHitKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%data_ptr = OpTypePointer CallableDataKHR %int
%data = OpVariable %data_ptr CallableDataKHR
%inData_ptr = OpTypePointer IncomingCallableDataKHR %int
%inData = OpVariable %inData_ptr IncomingCallableDataKHR
%main = OpFunction %void None %func
%label = OpLabel
OpExecuteCallableKHR %uint_0 %data
OpExecuteCallableKHR %uint_0 %inData
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("OpExecuteCallableKHR requires RayGenerationKHR, "
                "ClosestHitKHR, MissKHR and CallableKHR execution models"));
}

TEST_F(ValidateRayTracing, ExecuteCallableStorageClass) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint RayGenerationKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%data_ptr = OpTypePointer RayPayloadKHR %int
%data = OpVariable %data_ptr RayPayloadKHR
%main = OpFunction %void None %func
%label = OpLabel
OpExecuteCallableKHR %uint_0 %data
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Callable Data must have storage class CallableDataKHR "
                        "or IncomingCallableDataKHR"));
}

TEST_F(ValidateRayTracing, ExecuteCallableSbtIndex) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint CallableKHR %main "main"
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%int_1 = OpConstant %int 1
%data_ptr = OpTypePointer CallableDataKHR %int
%data = OpVariable %data_ptr CallableDataKHR
%main = OpFunction %void None %func
%label = OpLabel
OpExecuteCallableKHR %int_1 %data
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SBT Index must be a 32-bit unsigned int scalar"));
}

std::string GenerateRayTraceCode(
    const std::string& body,
    const std::string execution_model = "RayGenerationKHR",
    const std::string& declarations = "") {
  std::ostringstream ss;
  ss << R"(
OpCapability RayTracingKHR
OpCapability RayTraversalPrimitiveCullingKHR
OpCapability Float64
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint )"
     << execution_model << R"( %main "main"
OpDecorate %top_level_as DescriptorSet 0
OpDecorate %top_level_as Binding 0
%void = OpTypeVoid
%func = OpTypeFunction %void
%type_as = OpTypeAccelerationStructureKHR
%as_uc_ptr = OpTypePointer UniformConstant %type_as
%top_level_as = OpVariable %as_uc_ptr UniformConstant
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%float = OpTypeFloat 32
%float64 = OpTypeFloat 64
%f32vec3 = OpTypeVector %float 3
%f32vec4 = OpTypeVector %float 4
%float_0 = OpConstant %float 0
%float64_0 = OpConstant %float64 0
%v3composite = OpConstantComposite %f32vec3 %float_0 %float_0 %float_0
%v4composite = OpConstantComposite %f32vec4 %float_0 %float_0 %float_0 %float_0
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%payload_ptr = OpTypePointer RayPayloadKHR %int
%payload = OpVariable %payload_ptr RayPayloadKHR
%callable_ptr = OpTypePointer CallableDataKHR %int
%callable = OpVariable %callable_ptr CallableDataKHR
%ptr_uint = OpTypePointer Private %uint
%var_uint = OpVariable %ptr_uint Private
%ptr_float = OpTypePointer Private %float
%var_float = OpVariable %ptr_float Private
%ptr_f32vec3 = OpTypePointer Private %f32vec3
%var_f32vec3 = OpVariable %ptr_f32vec3 Private
)";

  ss << declarations;

  ss << R"(
%main = OpFunction %void None %func
%label = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";
  return ss.str();
}

TEST_F(ValidateRayTracing, TraceRaySuccess) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload

%_uint = OpLoad %uint %var_uint
%_float = OpLoad %float %var_float
%_f32vec3 = OpLoad %f32vec3 %var_f32vec3
OpTraceRayKHR %as %_uint %_uint %_uint %_uint %_uint %_f32vec3 %_float %_f32vec3 %_float %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateRayTracing, TraceRayExecutionModel) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body, "CallableKHR").c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpTraceRayKHR requires RayGenerationKHR, "
                        "ClosestHitKHR and MissKHR execution models"));
}

TEST_F(ValidateRayTracing, TraceRayAccelerationStructure) {
  const std::string body = R"(
%_uint = OpLoad %uint %var_uint
OpTraceRayKHR %_uint %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Expected Acceleration Structure to be of type "
                        "OpTypeAccelerationStructureKHR"));
}

TEST_F(ValidateRayTracing, TraceRayRayFlags) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %float_0 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Ray Flags must be a 32-bit int scalar"));
}

TEST_F(ValidateRayTracing, TraceRayCullMask) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %float_0 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Cull Mask must be a 32-bit int scalar"));
}

TEST_F(ValidateRayTracing, TraceRaySbtOffest) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %float_0 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SBT Offset must be a 32-bit int scalar"));
}

TEST_F(ValidateRayTracing, TraceRaySbtStride) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %float_0 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SBT Stride must be a 32-bit int scalar"));
}

TEST_F(ValidateRayTracing, TraceRayMissIndex) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %float_0 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Miss Index must be a 32-bit int scalar"));
}

TEST_F(ValidateRayTracing, TraceRayRayOrigin) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %float_0 %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Ray Origin must be a 32-bit float 3-component vector"));
}

TEST_F(ValidateRayTracing, TraceRayRayTMin) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %uint_1 %v3composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Ray TMin must be a 32-bit float scalar"));
}

TEST_F(ValidateRayTracing, TraceRayRayDirection) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v4composite %float_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Ray Direction must be a 32-bit float 3-component vector"));
}

TEST_F(ValidateRayTracing, TraceRayRayTMax) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float64_0 %payload
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Ray TMax must be a 32-bit float scalar"));
}

TEST_F(ValidateRayTracing, TraceRayPayload) {
  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %callable
)";

  CompileSuccessfully(GenerateRayTraceCode(body).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Payload must have storage class RayPayloadKHR or "
                        "IncomingRayPayloadKHR"));
}

TEST_F(ValidateRayTracing, InterfaceIncomingRayPayload) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint CallableKHR %main "main" %inData1 %inData2
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%int = OpTypeInt 32 1
%inData_ptr = OpTypePointer IncomingRayPayloadKHR %int
%inData1 = OpVariable %inData_ptr IncomingRayPayloadKHR
%inData2 = OpVariable %inData_ptr IncomingRayPayloadKHR
%main = OpFunction %void None %func
%label = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str(), SPV_ENV_VULKAN_1_2);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateAndRetrieveValidationState(SPV_ENV_VULKAN_1_2));
  EXPECT_THAT(getDiagnosticString(),
              AnyVUID("VUID-StandaloneSpirv-IncomingRayPayloadKHR-04700"));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Entry-point has more than one variable with the "
                "IncomingRayPayloadKHR storage class in the interface"));
}

TEST_F(ValidateRayTracing, InterfaceHitAttribute) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint CallableKHR %main "main" %inData1 %inData2
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%int = OpTypeInt 32 1
%inData_ptr = OpTypePointer HitAttributeKHR %int
%inData1 = OpVariable %inData_ptr HitAttributeKHR
%inData2 = OpVariable %inData_ptr HitAttributeKHR
%main = OpFunction %void None %func
%label = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str(), SPV_ENV_VULKAN_1_2);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateAndRetrieveValidationState(SPV_ENV_VULKAN_1_2));
  EXPECT_THAT(getDiagnosticString(),
              AnyVUID("VUID-StandaloneSpirv-HitAttributeKHR-04702"));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Entry-point has more than one variable with the "
                        "HitAttributeKHR storage class in the interface"));
}

TEST_F(ValidateRayTracing, InterfaceIncomingCallableData) {
  const std::string body = R"(
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint CallableKHR %main "main" %inData1 %inData2
OpName %main "main"
%void = OpTypeVoid
%func = OpTypeFunction %void
%int = OpTypeInt 32 1
%inData_ptr = OpTypePointer IncomingCallableDataKHR %int
%inData1 = OpVariable %inData_ptr IncomingCallableDataKHR
%inData2 = OpVariable %inData_ptr IncomingCallableDataKHR
%main = OpFunction %void None %func
%label = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(body.c_str(), SPV_ENV_VULKAN_1_2);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateAndRetrieveValidationState(SPV_ENV_VULKAN_1_2));
  EXPECT_THAT(getDiagnosticString(),
              AnyVUID("VUID-StandaloneSpirv-IncomingCallableDataKHR-04706"));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Entry-point has more than one variable with the "
                "IncomingCallableDataKHR storage class in the interface"));
}

TEST_F(ValidateRayTracing, TraceRayMinMoreThanMax) {
  const std::string declarations = R"(
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
)";

  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_2 %v3composite %float_1 %payload
)";

  CompileSuccessfully(
      GenerateRayTraceCode(body, "RayGenerationKHR", declarations).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Ray Tmin (2) is larger than Ray Tmax (1)"));
}

TEST_F(ValidateRayTracing, TraceRayMinNegative) {
  const std::string declarations = R"(
%float_n1 = OpConstant %float -1
)";

  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_n1 %v3composite %float_0 %payload
)";

  CompileSuccessfully(
      GenerateRayTraceCode(body, "RayGenerationKHR", declarations).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Ray Tmin is negative (-1)"));
}

TEST_F(ValidateRayTracing, TraceRayRayFlagsBothSkipPrimitiveCulling) {
  // SkipTrianglesKHR | SkipAABBsKHR
  const std::string declarations = R"(
%uint_768 = OpConstant %uint 768
)";

  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_768 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(
      GenerateRayTraceCode(body, "RayGenerationKHR", declarations).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Ray Flags contains both SkipTrianglesKHR and SkipAABBsKHR"));
}

TEST_F(ValidateRayTracing, TraceRayRayFlagsSkipAABBs) {
  // only SkipAABBsKHR
  const std::string declarations = R"(
%uint_512 = OpConstant %uint 512
)";

  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_512 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(
      GenerateRayTraceCode(body, "RayGenerationKHR", declarations).c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateRayTracing, TraceRayRayFlagsSkipTriangleCullBack) {
  // SkipTrianglesKHR and CullBackFacingTrianglesKHR
  const std::string declarations = R"(
%uint_272 = OpConstant %uint 272
)";

  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_272 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(
      GenerateRayTraceCode(body, "RayGenerationKHR", declarations).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Ray Flags contains more than one of SkipTrianglesKHR or "
                "CullFrontFacingTrianglesKHR or CullBackFacingTrianglesKHR"));
}

TEST_F(ValidateRayTracing, TraceRayRayFlagsSkipTriangleCullFrontAndBack) {
  // SkipTrianglesKHR and CullFrontFacingTrianglesKHR and
  // CullBackFacingTrianglesKHR
  const std::string declarations = R"(
%uint_304 = OpConstant %uint 304
)";

  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_304 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(
      GenerateRayTraceCode(body, "RayGenerationKHR", declarations).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Ray Flags contains more than one of SkipTrianglesKHR or "
                "CullFrontFacingTrianglesKHR or CullBackFacingTrianglesKHR"));
}

TEST_F(ValidateRayTracing, TraceRayRayFlagsSkipAABBCullBackward) {
  // SkipAABBsKHR and CullBackFacingTrianglesKHR (legal)
  const std::string declarations = R"(
%uint_528 = OpConstant %uint 528
)";

  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_528 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(
      GenerateRayTraceCode(body, "RayGenerationKHR", declarations).c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateRayTracing, TraceRayRayFlagsOpaqueAndCullNoOpaque) {
  // OpaqueKHR and CullNoOpaqueKHR
  const std::string declarations = R"(
%uint_129 = OpConstant %uint 129
)";

  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_129 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(
      GenerateRayTraceCode(body, "RayGenerationKHR", declarations).c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Ray Flags contains more than one of OpaqueKHR or "
                        "NoOpaqueKHR or CullOpaqueKHR or CullNoOpaqueKHR"));
}

TEST_F(ValidateRayTracing, TraceRayRayFlagsOpaqueAndCullBack) {
  // OpaqueKHR and CullBackFacingTrianglesKHR (legal)
  const std::string declarations = R"(
%uint_17 = OpConstant %uint 17
)";

  const std::string body = R"(
%as = OpLoad %type_as %top_level_as
OpTraceRayKHR %as %uint_17 %uint_1 %uint_1 %uint_1 %uint_1 %v3composite %float_0 %v3composite %float_0 %payload
)";

  CompileSuccessfully(
      GenerateRayTraceCode(body, "RayGenerationKHR", declarations).c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

}  // namespace
}  // namespace val
}  // namespace spvtools
