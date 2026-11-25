// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// Tests SPV_AMDX_multi_level_acceleration_structure.

#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "spirv-tools/libspirv.hpp"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Values;

const char* const kExtension =
    "OpExtension \"SPV_AMDX_multi_level_acceleration_structure\"\n";
const char* const kRayTracingCapability =
    "OpCapability RayTracingMultiLevelAccelerationStructureAMDX\n";
const char* const kRayQueryCapability =
    "OpCapability RayQueryMultiLevelAccelerationStructureAMDX\n";

struct MlasInstruction {
  const char* opcode;
  const char* result_type;
  bool ray_query;
  bool has_level;
};

using ValidateMlas = spvtest::ValidateBase<MlasInstruction>;

std::string GenerateShader(const std::string& body, bool ray_query,
                           const std::string& model = "ClosestHitKHR") {
  std::ostringstream ss;
  ss << "OpCapability Shader\nOpCapability Int64\nOpCapability Float64\n"
     << (ray_query ? kRayQueryCapability : kRayTracingCapability)
     << (ray_query ? "OpExtension \"SPV_KHR_ray_query\"\n"
                   : "OpExtension \"SPV_KHR_ray_tracing\"\n")
     << kExtension << "OpMemoryModel Logical GLSL450\n"
     << "OpEntryPoint " << model << " %main \"main\"\n";
  if (model == "GLCompute") {
    ss << "OpExecutionMode %main LocalSize 1 1 1\n";
  } else if (model == "Fragment") {
    ss << "OpExecutionMode %main OriginUpperLeft\n";
  }
  ss << R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%u64 = OpTypeInt 64 0
%f32 = OpTypeFloat 32
%f64 = OpTypeFloat 64
%u32vec2 = OpTypeVector %u32 2
%f32vec2 = OpTypeVector %f32 2
%f32vec3 = OpTypeVector %f32 3
%f32vec4 = OpTypeVector %f32 4
%f64vec3 = OpTypeVector %f64 3
%mat4x3 = OpTypeMatrix %f32vec3 4
%mat3x3 = OpTypeMatrix %f32vec3 3
%mat4x2 = OpTypeMatrix %f32vec2 4
%mat4x4 = OpTypeMatrix %f32vec4 4
%mat4x3f64 = OpTypeMatrix %f64vec3 4
%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%s32_0 = OpConstant %s32 0
%s32_1 = OpConstant %s32 1
%u64_0 = OpConstant %u64 0
%f32_0 = OpConstant %f32 0
%bool_false = OpConstantFalse %bool
%u32vec2_0 = OpConstantNull %u32vec2
%ptr_u32 = OpTypePointer Function %u32
)";
  if (ray_query) {
    ss << "%type_rq = OpTypeRayQueryKHR\n"
          "%ptr_rq = OpTypePointer Function %type_rq\n";
  }
  ss << "%main = OpFunction %void None %func\n%entry = OpLabel\n"
        "%not_ray_query = OpVariable %ptr_u32 Function\n";
  if (ray_query) {
    ss << "%ray_query = OpVariable %ptr_rq Function\n";
  }
  ss << "%dynamic = OpIAdd %u32 %u32_0 %u32_1\n"
     << body << "\nOpReturn\nOpFunctionEnd\n";
  return ss.str();
}

std::string MakeAccessor(const MlasInstruction& inst,
                         const std::string& result_type = "",
                         const std::string& level = "%u32_0",
                         const std::string& intersection = "%u32_0",
                         const std::string& query = "%ray_query") {
  std::ostringstream ss;
  ss << "%result = " << inst.opcode << " "
     << (result_type.empty() ? inst.result_type : result_type);
  if (inst.ray_query) ss << " " << query << " " << intersection;
  if (inst.has_level) ss << " " << level;
  return ss.str();
}

std::string ShaderForAccessor(const MlasInstruction& inst,
                              const std::string& body) {
  return GenerateShader(body, inst.ray_query,
                        inst.ray_query ? "GLCompute" : "ClosestHitKHR");
}

TEST_P(ValidateMlas, Success) {
  const auto& inst = GetParam();
  for (auto env : {SPV_ENV_UNIVERSAL_1_4, SPV_ENV_VULKAN_1_2}) {
    for (const auto* level : {"%u32_0", "%s32_0", "%dynamic"}) {
      for (const auto* intersection :
           {"%u32_0", "%u32_1", "%s32_0", "%s32_1"}) {
        SCOPED_TRACE(level);
        SCOPED_TRACE(intersection);
        CompileSuccessfully(
            ShaderForAccessor(inst,
                              MakeAccessor(inst, "", level, intersection)),
            env);
        ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(env))
            << getDiagnosticString();
      }
    }
  }
}

TEST_P(ValidateMlas, InvalidResultType) {
  const auto& inst = GetParam();
  for (const auto* type : {"%f32", "%u64", "%u32vec2", "%mat3x3", "%mat4x2",
                           "%mat4x4", "%mat4x3f64"}) {
    SCOPED_TRACE(type);
    CompileSuccessfully(ShaderForAccessor(inst, MakeAccessor(inst, type)),
                        SPV_ENV_VULKAN_1_2);
    EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_2));
    EXPECT_THAT(getDiagnosticString(), HasSubstr("Result Type"));
  }
}

TEST_P(ValidateMlas, InvalidLevelType) {
  const auto& inst = GetParam();
  if (!inst.has_level) return;
  for (const auto* level : {"%f32_0", "%u64_0", "%u32vec2_0", "%bool_false"}) {
    SCOPED_TRACE(level);
    CompileSuccessfully(ShaderForAccessor(inst, MakeAccessor(inst, "", level)),
                        SPV_ENV_VULKAN_1_2);
    EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_2));
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Level must be a 32-bit int scalar"));
  }
}

TEST_P(ValidateMlas, RayTracingExecutionModel) {
  const auto& inst = GetParam();
  if (inst.ray_query) return;
  for (const auto* model :
       {"IntersectionKHR", "AnyHitKHR", "ClosestHitKHR", "RayGenerationKHR",
        "MissKHR", "CallableKHR", "Fragment", "GLCompute"}) {
    SCOPED_TRACE(model);
    CompileSuccessfully(GenerateShader(MakeAccessor(inst), false, model),
                        SPV_ENV_VULKAN_1_2);
    if (std::string(model) == "IntersectionKHR" ||
        std::string(model) == "AnyHitKHR" ||
        std::string(model) == "ClosestHitKHR") {
      EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_2))
          << getDiagnosticString();
    } else {
      EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_2));
      EXPECT_THAT(getDiagnosticString(),
                  HasSubstr("Per-level instance instructions require "
                            "IntersectionKHR, AnyHitKHR or ClosestHitKHR"));
    }
  }
}

TEST_P(ValidateMlas, InvalidRayQueryPointer) {
  const auto& inst = GetParam();
  if (!inst.ray_query) return;
  CompileSuccessfully(
      ShaderForAccessor(
          inst, MakeAccessor(inst, "", "%u32_0", "%u32_0", "%not_ray_query")),
      SPV_ENV_VULKAN_1_2);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_2));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Ray Query must be a pointer to OpTypeRayQueryKHR"));
}

TEST_P(ValidateMlas, InvalidIntersection) {
  const auto& inst = GetParam();
  if (!inst.ray_query) return;
  for (const auto* intersection :
       {"%f32_0", "%u64_0", "%u32vec2_0", "%dynamic"}) {
    SCOPED_TRACE(intersection);
    CompileSuccessfully(
        ShaderForAccessor(inst, MakeAccessor(inst, "", "%u32_0", intersection)),
        SPV_ENV_VULKAN_1_2);
    EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_2));
    EXPECT_THAT(
        getDiagnosticString(),
        HasSubstr("Intersection ID to be a constant 32-bit int scalar"));
  }
}

TEST_P(ValidateMlas, MissingCapability) {
  const auto& inst = GetParam();
  std::string shader = ShaderForAccessor(inst, MakeAccessor(inst));
  const char* capability =
      inst.ray_query ? kRayQueryCapability : kRayTracingCapability;
  shader.replace(shader.find(capability), std::string(capability).size(),
                 inst.ray_query ? "OpCapability RayQueryKHR\n"
                                : "OpCapability RayTracingKHR\n");
  CompileSuccessfully(shader, SPV_ENV_VULKAN_1_2);
  EXPECT_EQ(SPV_ERROR_INVALID_CAPABILITY,
            ValidateInstructions(SPV_ENV_VULKAN_1_2));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr(inst.ray_query
                            ? "RayQueryMultiLevelAccelerationStructureAMDX"
                            : "RayTracingMultiLevelAccelerationStructureAMDX"));
}

TEST_P(ValidateMlas, MissingExtension) {
  const auto& inst = GetParam();
  std::string shader = ShaderForAccessor(inst, MakeAccessor(inst));
  shader.erase(shader.find(kExtension), std::string(kExtension).size());
  CompileSuccessfully(shader, SPV_ENV_UNIVERSAL_1_4);
  EXPECT_EQ(SPV_ERROR_MISSING_EXTENSION,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_4));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("SPV_AMDX_multi_level_acceleration_structure"));
}

TEST_P(ValidateMlas, RequiresSpirv14) {
  const auto& inst = GetParam();
  CompileSuccessfully(ShaderForAccessor(inst, MakeAccessor(inst)),
                      SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_WRONG_VERSION,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("requires SPIR-V version 1.4"));
}

TEST_P(ValidateMlas, AssemblyRoundTrip) {
  const auto& inst = GetParam();
  CompileSuccessfully(ShaderForAccessor(inst, MakeAccessor(inst)),
                      SPV_ENV_UNIVERSAL_1_4);
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_4);
  std::string text;
  ASSERT_TRUE(tools.Disassemble(binary_->code, binary_->wordCount, &text,
                                SPV_BINARY_TO_TEXT_OPTION_NO_HEADER));
  EXPECT_THAT(text, HasSubstr(inst.opcode));
  std::vector<uint32_t> binary;
  ASSERT_TRUE(tools.Assemble(text, &binary,
                             SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS));
  EXPECT_EQ(binary, std::vector<uint32_t>(binary_->code,
                                          binary_->code + binary_->wordCount));
}

INSTANTIATE_TEST_SUITE_P(
    MultiLevelAccelerationStructure, ValidateMlas,
    Values(MlasInstruction{"OpGetInstanceCustomIndexAtLevelAMDX", "%u32", false,
                           true},
           MlasInstruction{"OpGetInstanceIdAtLevelAMDX", "%s32", false, true},
           MlasInstruction{"OpGetObjectToWorldAtLevelAMDX", "%mat4x3", false,
                           true},
           MlasInstruction{"OpGetWorldToObjectAtLevelAMDX", "%mat4x3", false,
                           true},
           MlasInstruction{"OpRayQueryGetIntersectionInstanceLevelAMDX", "%u32",
                           true, false},
           MlasInstruction{
               "OpRayQueryGetIntersectionInstanceCustomIndexAtLevelAMDX",
               "%u32", true, true},
           MlasInstruction{"OpRayQueryGetIntersectionInstanceIdAtLevelAMDX",
                           "%s32", true, true},
           MlasInstruction{"OpRayQueryGetIntersectionObjectToWorldAtLevelAMDX",
                           "%mat4x3", true, true},
           MlasInstruction{"OpRayQueryGetIntersectionWorldToObjectAtLevelAMDX",
                           "%mat4x3", true, true}));

}  // namespace
}  // namespace val
}  // namespace spvtools
