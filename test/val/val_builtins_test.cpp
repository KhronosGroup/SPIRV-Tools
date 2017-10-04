// Copyright (c) 2017 Google Inc.
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

// Common validation fixtures for unit tests

#include "gmock/gmock.h"
#include "unit_spirv.h"
#include "val_fixtures.h"
#include "source/val/decoration.h"

namespace {

using std::string;
using std::vector;
using std::tie;
using std::tuple;
using std::make_tuple;
using ::testing::HasSubstr;
using ::testing::Eq;
using libspirv::Decoration;

using tuple4string = tuple<string,string,string,string>;

using ValidateBuiltInExecutionModel = spvtest::ValidateBase<tuple4string>;
using ValidateBuiltInStorage = spvtest::ValidateBase<tuple4string>;
using ValidateBuiltInType = spvtest::ValidateBase<tuple4string>;
using ValidateBuiltIn = spvtest::ValidateBase<tuple4string>;


string GenerateShaderCode(const tuple4string& tuple)
{
  string bltin, type, storage, exec;
  tie(bltin,type,storage,exec) = tuple;
  
  string capabilities = R"(
    OpCapability Shader
    OpCapability Tessellation
    OpCapability Geometry
    OpCapability DrawParameters
    OpCapability SampleRateShading
    OpCapability DeviceGroup
    OpCapability MultiView
    OpCapability MultiViewport
    OpCapability Linkage
  )";
  string extensions = R"(
    OpExtension "SPV_KHR_shader_draw_parameters"
    OpExtension "SPV_KHR_device_group"
    OpExtension "SPV_KHR_multiview"
  )";
  string memory = R"(
    OpMemoryModel Logical GLSL450
  )";
  string entries = R"(
    OpEntryPoint )" + exec + R"( %entry "entry" %bltin
  )";
  string modes = "";
  string decorations = R"(  
    OpDecorate %bltin BuiltIn )" + bltin + R"(
  )";
  string types = R"(
    %void    = OpTypeVoid
    %bool    = OpTypeBool
    %uint32  = OpTypeInt 32 0
    %float   = OpTypeFloat 32
    %vecu2   = OpTypeVector %uint32 2
    %vecu3   = OpTypeVector %uint32 3
    %vecu4   = OpTypeVector %uint32 4
    %vecf2   = OpTypeVector %float 2
    %vecf3   = OpTypeVector %float 3
    %vecf4   = OpTypeVector %float 4
    %const2  = OpConstant %uint32 2
    %const3  = OpConstant %uint32 3
    %const4  = OpConstant %uint32 4
    %arru2   = OpTypeArray %uint32 %const2
    %arru3   = OpTypeArray %uint32 %const3
    %arru4   = OpTypeArray %uint32 %const4
    %arrf2   = OpTypeArray %float %const2
    %arrf3   = OpTypeArray %float %const3
    %arrf4   = OpTypeArray %float %const4
    %voidfun = OpTypeFunction %void
  )";
  string definitions = R"(
    %ptr   = OpTypePointer )" + storage + " " + type + R"(
    %bltin = OpVariable %ptr )" + storage + R"(
    %entry = OpFunction %void None %voidfun
    %label = OpLabel
             OpReturn
             OpFunctionEnd
  )";

  return capabilities + extensions + memory + entries
         + modes + decorations + types + definitions;
}

// clang-format off
#define BUILTIN(builtin,type,storage,exec) \
  make_tuple(string(builtin),string(type),string(storage),string(exec))

TEST_P(ValidateBuiltIn, BuiltInValid) {
  CompileSuccessfully(GenerateShaderCode(GetParam()));
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

INSTANTIATE_TEST_CASE_P(BuiltInValid, ValidateBuiltIn,
  ::testing::Values(
       BUILTIN("Position","%vecf4","Output","Vertex"),
       BUILTIN("PointSize","%float","Output","Vertex"),
       BUILTIN("ClipDistance","%arrf2","Output","Vertex"),
       BUILTIN("CullDistance","%arrf3","Output","Vertex"),
       BUILTIN("VertexId","%uint32","Input","Vertex"),
       BUILTIN("InstanceId","%float","Input","Vertex"),
       BUILTIN("PrimitiveId","%uint32","Input","Fragment"),
       BUILTIN("InvocationId","%uint32","Input","Geometry"),
       BUILTIN("Layer","%uint32","Input","Fragment"),
       BUILTIN("ViewportIndex","%uint32","Output","Geometry"),
       BUILTIN("TessLevelOuter","%arrf4","Output","TessellationControl"),
       BUILTIN("TessLevelInner","%arrf2","Input","TessellationEvaluation"),
       BUILTIN("TessCoord","%vecf3","Input","TessellationEvaluation"),
       BUILTIN("PatchVertices","%uint32","Input","TessellationControl"),
       BUILTIN("FragCoord","%vecf4","Input","Fragment"),
       BUILTIN("PointCoord","%vecf2","Input","Fragment"),
       BUILTIN("FrontFacing","%bool","Input","Fragment"),
       BUILTIN("SampleId","%uint32","Input","Fragment"),
       BUILTIN("SamplePosition","%vecf2","Input","Fragment"),
       BUILTIN("SampleMask","%arru2","Input","Fragment"),
       BUILTIN("FragDepth","%float","Output","Fragment"),
       BUILTIN("HelperInvocation","%bool","Input","Fragment"),
       BUILTIN("NumWorkgroups","%vecu3","Input","GLCompute"),
       //BUILTIN("WorkgroupSize","%vecu3","Uniform","GLCompute"),
       BUILTIN("WorkgroupId","%vecu3","Input","GLCompute"),
       BUILTIN("LocalInvocationId","%vecu3","Input","GLCompute"),
       BUILTIN("GlobalInvocationId","%vecu3","Input","GLCompute"),
       BUILTIN("LocalInvocationIndex","%uint32","Input","GLCompute"),
       // WorkDim, GLobalSize, EnqueueWorkgroupSize, GLobalOffset
       // GLobalLinearId, SubgroupSize, SubgroupMaxSize, NumSubgroups
       // NumEnqueuedSubgroups, SubgroupId, SubroupLocalInvocationId,
       BUILTIN("VertexIndex","%uint32","Input","Vertex"),
       BUILTIN("InstanceIndex","%uint32","Input","Vertex")
       // BaseVertex, BaseInstance, DrawIndex, DeviceIndex, ViewIndex
  ), );


TEST_P(ValidateBuiltInExecutionModel, BuiltInExecutionModel) {
  CompileSuccessfully(GenerateShaderCode(GetParam()));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Built-in variables are restricted to only certain"
                        "EXECUTION MODELS (see e.g. Vulkan specification)"));
}

INSTANTIATE_TEST_CASE_P(BuiltInExecutionModel, ValidateBuiltInExecutionModel,
  ::testing::Values(
       BUILTIN("Position","%vecf4","Output","Fragment"),
       BUILTIN("PointSize","%float","Output","GLCompute"),
       BUILTIN("ClipDistance","%arrf2","Output","GLCompute"),
       BUILTIN("CullDistance","%arrf3","Output","GLCompute"),
       BUILTIN("VertexId","%uint32","Input","TessellationControl"),
       BUILTIN("InstanceId","%float","Input","TessellationEvaluation"),
       BUILTIN("PrimitiveId","%uint32","Input","Vertex"),
       BUILTIN("InvocationId","%uint32","Input","Fragment"),
       BUILTIN("Layer","%uint32","Input","GLCompute"),
       BUILTIN("ViewportIndex","%uint32","Output","Vertex"),
       BUILTIN("TessLevelOuter","%arrf4","Output","Geometry"),
       BUILTIN("TessLevelInner","%arrf2","Input","Fragment"),
       BUILTIN("TessCoord","%vecf3","Input","TessellationControl"),
       BUILTIN("PatchVertices","%uint32","Input","Vertex"),
       BUILTIN("FragCoord","%vecf4","Input","Geometry"),
       BUILTIN("PointCoord","%vecf2","Input","GLCompute"),
       BUILTIN("FrontFacing","%bool","Input","Vertex"),
       BUILTIN("SampleId","%uint32","Input","GLCompute"),
       BUILTIN("SamplePosition","%vecf2","Input","Geometry"),
       BUILTIN("SampleMask","%arru2","Input","Vertex"),
       BUILTIN("FragDepth","%float","Output","GLCompute"),
       BUILTIN("HelperInvocation","%bool","Input","GLCompute"),
       BUILTIN("NumWorkgroups","%vecu3","Input","Vertex"),
       //BUILTIN("WorkgroupSize","%vecu3","Uniform","Fragment"),
       BUILTIN("WorkgroupId","%vecu3","Input","TessellationControl"),
       BUILTIN("LocalInvocationId","%vecu3","Input","TessellationEvaluation"),
       BUILTIN("GlobalInvocationId","%vecu3","Input","Geometry"),
       BUILTIN("LocalInvocationIndex","%uint32","Input","Vertex"),
       // WorkDim, GLobalSize, EnqueueWorkgroupSize, GLobalOffset
       // GLobalLinearId, SubgroupSize, SubgroupMaxSize, NumSubgroups
       // NumEnqueuedSubgroups, SubgroupId, SubroupLocalInvocationId,
       BUILTIN("VertexIndex","%uint32","Input","Fragment"),
       BUILTIN("InstanceIndex","%uint32","Input","Geometry")
       // BaseVertex, BaseInstance, DrawIndex, DeviceIndex, ViewIndex
  ), );


TEST_P(ValidateBuiltInStorage, BuiltInStorage) {
  CompileSuccessfully(GenerateShaderCode(GetParam()));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Built-in variables must match the STORAGE CLASS "
                         "of the built-in (see e.g. Vulkan specification)"));
}

INSTANTIATE_TEST_CASE_P(BuiltInStorage, ValidateBuiltInStorage,
  ::testing::Values(
       BUILTIN("Position","%vecf4","Input","Vertex"),
       BUILTIN("PointSize","%float","Input","Vertex"),
       BUILTIN("ClipDistance","%arrf2","Input","Vertex"),
       BUILTIN("CullDistance","%arrf3","Input","Vertex"),
       BUILTIN("VertexId","%uint32","Output","Vertex"),
       BUILTIN("InstanceId","%float","Output","Vertex"),
       BUILTIN("PrimitiveId","%uint32","Output","Fragment"),
       BUILTIN("InvocationId","%uint32","Output","Geometry"),
       BUILTIN("Layer","%uint32","Output","Fragment"),
       BUILTIN("ViewportIndex","%uint32","Input","Geometry"),
       BUILTIN("TessLevelOuter","%arrf4","Input","TessellationControl"),
       BUILTIN("TessLevelInner","%arrf2","Output","TessellationEvaluation"),
       BUILTIN("TessCoord","%vecf3","Output","TessellationEvaluation"),
       BUILTIN("PatchVertices","%uint32","Output","TessellationControl"),
       BUILTIN("FragCoord","%vecf4","Output","Fragment"),
       BUILTIN("PointCoord","%vecf2","Output","Fragment"),
       BUILTIN("FrontFacing","%bool","Output","Fragment"),
       BUILTIN("SampleId","%uint32","Output","Fragment"),
       BUILTIN("SamplePosition","%vecf2","Output","Fragment"),
       //BUILTIN("SampleMask","%arru2","Uniform","Fragment"),
       BUILTIN("FragDepth","%float","Input","Fragment"),
       BUILTIN("HelperInvocation","%bool","Output","Fragment"),
       BUILTIN("NumWorkgroups","%vecu3","Output","GLCompute"),
       //BUILTIN("WorkgroupSize","%vecu3","Uniform","GLCompute"),
       BUILTIN("WorkgroupId","%vecu3","Output","GLCompute"),
       BUILTIN("LocalInvocationId","%vecu3","Output","GLCompute"),
       BUILTIN("GlobalInvocationId","%vecu3","Output","GLCompute"),
       BUILTIN("LocalInvocationIndex","%uint32","Output","GLCompute"),
       // WorkDim, GLobalSize, EnqueueWorkgroupSize, GLobalOffset
       // GLobalLinearId, SubgroupSize, SubgroupMaxSize, NumSubgroups
       // NumEnqueuedSubgroups, SubgroupId, SubroupLocalInvocationId,
       BUILTIN("VertexIndex","%uint32","Output","Vertex"),
       BUILTIN("InstanceIndex","%uint32","Output","Vertex")
       // BaseVertex, BaseInstance, DrawIndex, DeviceIndex, ViewIndex
  ), );


TEST_P(ValidateBuiltInType, BuiltInType) {
  CompileSuccessfully(GenerateShaderCode(GetParam()));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Built-in variables must match the DATA TYPE "
                        "of the built-in (see e.g. Vulkan specification)"));
}

INSTANTIATE_TEST_CASE_P(BuiltInType, ValidateBuiltInType,
  ::testing::Values(
       BUILTIN("Position","%uint32","Output","Vertex"),
       BUILTIN("PointSize","%uint32","Output","Vertex"),
       BUILTIN("ClipDistance","%vecf2","Output","Vertex"),
       BUILTIN("CullDistance","%float","Output","Vertex"),
       BUILTIN("VertexId","%arrf3","Input","Vertex"),
       BUILTIN("InstanceId","%uint32","Input","Vertex"),
       BUILTIN("PrimitiveId","%vecf4","Input","Fragment"),
       BUILTIN("InvocationId","%float","Input","Geometry"),
       BUILTIN("Layer","%arru3","Output","Geometry"),
       BUILTIN("ViewportIndex","%vecu2","Input","Fragment"),
       BUILTIN("TessLevelOuter","%uint32","Input","TessellationEvaluation"),
       BUILTIN("TessLevelInner","%arrf4","Output","TessellationControl"),
       BUILTIN("TessCoord","%vecu3","Input","TessellationEvaluation"),
       BUILTIN("PatchVertices","%float","Input","TessellationControl"),
       BUILTIN("FragCoord","%vecf2","Input","Fragment"),
       BUILTIN("PointCoord","%vecf4","Input","Fragment"),
       BUILTIN("FrontFacing","%void","Input","Fragment"),
       BUILTIN("SampleId","%bool","Input","Fragment"),
       BUILTIN("SamplePosition","%vecu2","Input","Fragment"),
       BUILTIN("SampleMask","%arrf3","Input","Fragment"),
       BUILTIN("FragDepth","%void","Output","Fragment"),
       BUILTIN("HelperInvocation","%void","Input","Fragment"),
       BUILTIN("NumWorkgroups","%arrf2","Input","GLCompute"),
       //BUILTIN("WorkgroupSize","%vecu3","Uniform","GLCompute"),
       BUILTIN("WorkgroupId","%vecf3","Input","GLCompute"),
       BUILTIN("LocalInvocationId","%uint32","Input","GLCompute"),
       BUILTIN("GlobalInvocationId","%float","Input","GLCompute"),
       BUILTIN("LocalInvocationIndex","%vecu3","Input","GLCompute"),
       // WorkDim, GLobalSize, EnqueueWorkgroupSize, GLobalOffset
       // GLobalLinearId, SubgroupSize, SubgroupMaxSize, NumSubgroups
       // NumEnqueuedSubgroups, SubgroupId, SubroupLocalInvocationId,
       BUILTIN("VertexIndex","%vecf2","Input","Vertex"),
       BUILTIN("InstanceIndex","%arru4","Input","Vertex")
       // BaseVertex, BaseInstance, DrawIndex, DeviceIndex, ViewIndex
       
  ), );

// clang-format on
#undef BUILTIN

TEST_F(ValidateBuiltIn, PrimitiveIdAsFragmentInput) {
  string spirv = R"(
    OpCapability Shader
    OpCapability Linkage
    OpMemoryModel Logical GLSL450
    OpEntryPoint Fragment %entry "entry" %bltin
    OpDecorate %bltin BuiltIn PrimitiveId
    %void    = OpTypeVoid
    %uint32  = OpTypeInt 32 0
    %voidfun = OpTypeFunction %void
    %ptr   = OpTypePointer Input %uint32
    %bltin = OpVariable %ptr Input
    %entry = OpFunction %void None %voidfun
    %label = OpLabel
             OpReturn
             OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_CAPABILITY,
            ValidateAndRetrieveValidationState());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Decorate requires one of these capabilities: "
                        "Geometry Tessellation"));
}

}  // anonymous namespace
