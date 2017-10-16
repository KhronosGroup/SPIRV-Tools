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

namespace spvtest {
  // TODO(jcaraban): what is a better place for this struct? val_fixtures.cpp?
  struct BuiltInCase {
    std::string name; // the builtin's name
    std::string type; // its data type
    std::string storage; // its storage class
    std::string stage; // the shader stage
  };
  template class spvtest::ValidateBase<BuiltInCase>;
}

namespace {

using std::string;
using ::testing::HasSubstr;
using ::testing::Eq;

using spvtest::BuiltInCase;
using ValidateBuiltInStage = spvtest::ValidateBase<BuiltInCase>;
using ValidateBuiltInStorage = spvtest::ValidateBase<BuiltInCase>;
using ValidateBuiltInType = spvtest::ValidateBase<BuiltInCase>;
using ValidateBuiltIn = spvtest::ValidateBase<BuiltInCase>;

using ValidateBuiltInStageCL = spvtest::ValidateBase<BuiltInCase>;
using ValidateBuiltInStorageCL = spvtest::ValidateBase<BuiltInCase>;
using ValidateBuiltInTypeCL = spvtest::ValidateBase<BuiltInCase>;
using ValidateBuiltInCL = spvtest::ValidateBase<BuiltInCase>;


string GenerateShaderCode(const BuiltInCase& bic)
{ 
  string capabilities = R"(
    OpCapability Shader
    OpCapability Tessellation
    OpCapability Geometry
    OpCapability DrawParameters
    OpCapability SampleRateShading
    OpCapability DeviceGroup
    OpCapability MultiView
    OpCapability MultiViewport
    OpCapability SubgroupBallotKHR
    OpCapability StencilExportEXT
    OpCapability ShaderViewportMaskNV
    OpCapability ShaderStereoViewNV
    OpCapability PerViewAttributesNV
  )";
  string extensions = R"(
    OpExtension "SPV_KHR_shader_draw_parameters"
    OpExtension "SPV_KHR_device_group"
    OpExtension "SPV_KHR_multiview"
    OpExtension "SPV_KHR_shader_ballot"
    OpExtension "SPV_AMD_shader_explicit_vertex_parameter"
    OpExtension "SPV_EXT_shader_stencil_export"
    OpExtension "SPV_NV_viewport_array2"
    OpExtension "SPV_NV_stereo_view_rendering"
    OpExtension "SPV_NVX_multiview_per_view_attributes"
  )";
  string memory = R"(
    OpMemoryModel Logical GLSL450
  )";
  string entries = R"(
    OpEntryPoint )" + bic.stage + R"( %entry "entry" %bltin
  )";
  string modes = "";
  string decorations = R"(  
    OpDecorate %bltin BuiltIn )" + bic.name + R"(
  )";
  string types = R"(
    %void    = OpTypeVoid
    %bool    = OpTypeBool
    %int32   = OpTypeInt 32 1
    %float   = OpTypeFloat 32
    %veci2   = OpTypeVector %int32 2
    %veci3   = OpTypeVector %int32 3
    %veci4   = OpTypeVector %int32 4
    %vecf2   = OpTypeVector %float 2
    %vecf3   = OpTypeVector %float 3
    %vecf4   = OpTypeVector %float 4
    %const2  = OpConstant %int32 2
    %const3  = OpConstant %int32 3
    %const4  = OpConstant %int32 4
    %arri2   = OpTypeArray %int32 %const2
    %arri3   = OpTypeArray %int32 %const3
    %arri4   = OpTypeArray %int32 %const4
    %arrf2   = OpTypeArray %float %const2
    %arrf3   = OpTypeArray %float %const3
    %arrf4   = OpTypeArray %float %const4
    %voidfun = OpTypeFunction %void
  )";
  string definitions = R"(
    %ptr   = OpTypePointer )" + bic.storage + " " + bic.type + R"(
    %bltin = OpVariable %ptr )" + bic.storage + R"(
    %entry = OpFunction %void None %voidfun
    %label = OpLabel
             OpReturn
             OpFunctionEnd
  )";

  return capabilities + extensions + memory + entries
         + modes + decorations + types + definitions;
}

string GenerateKernelCode(const BuiltInCase& bic)
{ 
  string capabilities = R"(
    OpCapability Kernel
    OpCapability Addresses
    OpCapability Int64
    OpCapability Shader
    OpCapability Tessellation
    OpCapability Geometry
  )";
  string extensions = "";
  string memory = R"(
    OpMemoryModel Physical64 OpenCL
  )";
  string entries = R"(
    OpEntryPoint )" + bic.stage + R"( %entry "entry" %bltin
  )";
  string modes = "";
  string decorations = R"(  
    OpDecorate %bltin BuiltIn )" + bic.name + R"(
  )";
  string types = R"(
    %void    = OpTypeVoid
    %float   = OpTypeFloat 32
    %uint32  = OpTypeInt 32 0
    %uint64  = OpTypeInt 64 0
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
    %ptr   = OpTypePointer )" + bic.storage + " " + bic.type + R"(
    %bltin = OpVariable %ptr )" + bic.storage + R"(
    %entry = OpFunction %void None %voidfun
    %label = OpLabel
             OpReturn
             OpFunctionEnd
  )";

  return capabilities + extensions + memory + entries
         + modes + decorations + types + definitions;
}

//
// PARAMETERIZED TESTS
//

TEST_P(ValidateBuiltIn, BuiltInValid) {
  CompileSuccessfully(GenerateShaderCode(GetParam()),SPV_ENV_VULKAN_1_0);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

// clang-format off
INSTANTIATE_TEST_CASE_P(BasicBuiltIn, ValidateBuiltIn,
  ::testing::Values(
       BuiltInCase{"Position","%vecf4","Output","Vertex"},
       BuiltInCase{"PointSize","%float","Output","Vertex"},
       BuiltInCase{"ClipDistance","%arrf2","Output","Vertex"},
       BuiltInCase{"CullDistance","%arrf3","Output","Vertex"},
       BuiltInCase{"VertexId","%int32","Input","Vertex"},
       BuiltInCase{"InstanceId","%int32","Input","Vertex"},
       BuiltInCase{"PrimitiveId","%int32","Input","Fragment"},
       BuiltInCase{"InvocationId","%int32","Input","Geometry"},
       BuiltInCase{"Layer","%int32","Input","Fragment"},
       BuiltInCase{"ViewportIndex","%int32","Output","Geometry"},
       BuiltInCase{"TessLevelOuter","%arrf4","Output","TessellationControl"},
       BuiltInCase{"TessLevelInner","%arrf2","Input","TessellationEvaluation"},
       BuiltInCase{"TessCoord","%vecf3","Input","TessellationEvaluation"},
       BuiltInCase{"PatchVertices","%int32","Input","TessellationControl"},
       BuiltInCase{"FragCoord","%vecf4","Input","Fragment"},
       BuiltInCase{"PointCoord","%vecf2","Input","Fragment"},
       BuiltInCase{"FrontFacing","%bool","Input","Fragment"},
       BuiltInCase{"SampleId","%int32","Input","Fragment"},
       BuiltInCase{"SamplePosition","%vecf2","Input","Fragment"},
       BuiltInCase{"SampleMask","%arri2","Input","Fragment"},
       BuiltInCase{"FragDepth","%float","Output","Fragment"},
       BuiltInCase{"HelperInvocation","%bool","Input","Fragment"},
       BuiltInCase{"NumWorkgroups","%veci3","Input","GLCompute"},
       // WorkgroupSize is tested individually
       BuiltInCase{"WorkgroupId","%veci3","Input","GLCompute"},
       BuiltInCase{"LocalInvocationId","%veci3","Input","GLCompute"},
       BuiltInCase{"GlobalInvocationId","%veci3","Input","GLCompute"},
       BuiltInCase{"LocalInvocationIndex","%int32","Input","GLCompute"},
       BuiltInCase{"VertexIndex","%int32","Input","Vertex"},
       BuiltInCase{"InstanceIndex","%int32","Input","Vertex"}
  ));

INSTANTIATE_TEST_CASE_P(ExtendedBuiltIn, ValidateBuiltIn,
  ::testing::Values(
       BuiltInCase{"SubgroupEqMaskKHR","%veci4","Input","Vertex"},
       BuiltInCase{"SubgroupGeMaskKHR","%veci4","Input","Geometry"},
       BuiltInCase{"SubgroupGtMaskKHR","%veci4","Input","TessellationControl"},
       BuiltInCase{"SubgroupLeMaskKHR","%veci4","Input","TessellationEvaluation"},
       BuiltInCase{"SubgroupLtMaskKHR","%veci4","Input","Fragment"},

       BuiltInCase{"BaseVertex","%int32","Input","Vertex"},
       BuiltInCase{"BaseInstance","%int32","Input","Vertex"},
       BuiltInCase{"DrawIndex","%int32","Input","Vertex"},
       BuiltInCase{"DeviceIndex","%int32","Input","GLCompute"},
       BuiltInCase{"ViewIndex","%int32","Input","Fragment"},

       BuiltInCase{"BaryCoordNoPerspAMD","%vecf2","Input","Fragment"},
       BuiltInCase{"BaryCoordNoPerspCentroidAMD","%vecf2","Input","Fragment"},
       BuiltInCase{"BaryCoordNoPerspSampleAMD","%vecf2","Input","Fragment"},
       BuiltInCase{"BaryCoordSmoothAMD","%vecf2","Input","Fragment"},
       BuiltInCase{"BaryCoordSmoothCentroidAMD","%vecf2","Input","Fragment"},
       BuiltInCase{"BaryCoordSmoothSampleAMD","%vecf2","Input","Fragment"},
       BuiltInCase{"BaryCoordPullModelAMD","%vecf3","Input","Fragment"},

       BuiltInCase{"FragStencilRefEXT","%int32","Output","Fragment"},
       BuiltInCase{"ViewportMaskNV","%int32","Output","Vertex"},
       BuiltInCase{"SecondaryPositionNV","%vecf4","Input","Geometry"},
       BuiltInCase{"SecondaryViewportMaskNV","%arri3","Output","Vertex"},
       BuiltInCase{"PositionPerViewNV","%vecf4","Input","TessellationControl"},
       BuiltInCase{"ViewportMaskPerViewNV","%arri4","Output","Geometry"}
  ));
// clang-format on

TEST_P(ValidateBuiltInCL, BuiltInCLOK) {
  CompileSuccessfully(GenerateKernelCode(GetParam()),SPV_ENV_OPENCL_2_2);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

// clang-format off
INSTANTIATE_TEST_CASE_P(BuiltInCLOK, ValidateBuiltInCL,
  ::testing::Values(
       BuiltInCase{"NumWorkgroups","%vecu3","Input","Kernel"},
       // WorkgroupSize is tested individually
       BuiltInCase{"WorkgroupId","%vecu3","Input","Kernel"},
       BuiltInCase{"LocalInvocationId","%vecu3","Input","Kernel"},
       BuiltInCase{"GlobalInvocationId","%vecu3","Input","Kernel"},
       BuiltInCase{"LocalInvocationIndex","%uint32","Input","Kernel"},

       BuiltInCase{"WorkDim","%uint32","Input","Kernel"},
       BuiltInCase{"GlobalSize","%vecu3","Input","Kernel"},
       BuiltInCase{"EnqueuedWorkgroupSize","%vecu3","Input","Kernel"},
       BuiltInCase{"GlobalOffset","%vecu3","Input","Kernel"},
       BuiltInCase{"GlobalLinearId","%uint32","Input","Kernel"},
       BuiltInCase{"SubgroupSize","%uint32","Input","Kernel"},
       BuiltInCase{"SubgroupMaxSize","%uint32","Input","Kernel"},
       BuiltInCase{"NumSubgroups","%uint32","Input","Kernel"},
       BuiltInCase{"NumEnqueuedSubgroups","%uint32","Input","Kernel"},
       BuiltInCase{"SubgroupId","%uint32","Input","Kernel"},
       BuiltInCase{"SubgroupLocalInvocationId","%uint32","Input","Kernel"}
  ));
// clang-format on

TEST_P(ValidateBuiltInStage, BuiltInStage) {
  CompileSuccessfully(GenerateShaderCode(GetParam()),SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr( GetParam().name +
                " built-in is restricted to certain EXECUTION MODELS "
                "(see SPIR-V, Vulkan, OpenGL, OpenCL specifications)"));
}

// clang-format off
INSTANTIATE_TEST_CASE_P(BasicBuiltIn, ValidateBuiltInStage,
  ::testing::Values(
       BuiltInCase{"Position","%vecf4","Output","Fragment"},
       BuiltInCase{"PointSize","%float","Output","GLCompute"},
       BuiltInCase{"ClipDistance","%arrf2","Output","GLCompute"},
       BuiltInCase{"CullDistance","%arrf3","Output","GLCompute"},
       BuiltInCase{"VertexId","%int32","Input","TessellationControl"},
       BuiltInCase{"InstanceId","%int32","Input","TessellationEvaluation"},
       BuiltInCase{"PrimitiveId","%int32","Input","Vertex"},
       BuiltInCase{"InvocationId","%int32","Input","Fragment"},
       BuiltInCase{"Layer","%int32","Input","GLCompute"},
       BuiltInCase{"ViewportIndex","%int32","Output","GLCompute"},
       BuiltInCase{"TessLevelOuter","%arrf4","Output","Geometry"},
       BuiltInCase{"TessLevelInner","%arrf2","Input","Fragment"},
       BuiltInCase{"TessCoord","%vecf3","Input","TessellationControl"},
       BuiltInCase{"PatchVertices","%int32","Input","Vertex"},
       BuiltInCase{"FragCoord","%vecf4","Input","Geometry"},
       BuiltInCase{"PointCoord","%vecf2","Input","GLCompute"},
       BuiltInCase{"FrontFacing","%bool","Input","Vertex"},
       BuiltInCase{"SampleId","%int32","Input","GLCompute"},
       BuiltInCase{"SamplePosition","%vecf2","Input","Geometry"},
       BuiltInCase{"SampleMask","%arri2","Input","Vertex"},
       BuiltInCase{"FragDepth","%float","Output","GLCompute"},
       BuiltInCase{"HelperInvocation","%bool","Input","GLCompute"},
       BuiltInCase{"NumWorkgroups","%veci3","Input","Vertex"},
       // 25 = WorkgroupSize is tested individually
       BuiltInCase{"WorkgroupId","%veci3","Input","TessellationControl"},
       BuiltInCase{"LocalInvocationId","%veci3","Input","TessellationEvaluation"},
       BuiltInCase{"GlobalInvocationId","%veci3","Input","Geometry"},
       BuiltInCase{"LocalInvocationIndex","%int32","Input","Vertex"},
       BuiltInCase{"VertexIndex","%int32","Input","Fragment"},
       BuiltInCase{"InstanceIndex","%int32","Input","Geometry"}
  ));

INSTANTIATE_TEST_CASE_P(ExtendedBuiltIn, ValidateBuiltInStage,
  ::testing::Values(
       BuiltInCase{"SubgroupEqMaskKHR","%veci3","Input","GLCompute"},
       BuiltInCase{"SubgroupGeMaskKHR","%veci3","Input","GLCompute"},
       BuiltInCase{"SubgroupGtMaskKHR","%veci3","Input","GLCompute"},
       BuiltInCase{"SubgroupLeMaskKHR","%veci3","Input","GLCompute"},
       BuiltInCase{"SubgroupLtMaskKHR","%veci3","Input","GLCompute"},

       BuiltInCase{"BaseVertex","%int32","Input","Fragment"},
       BuiltInCase{"BaseInstance","%int32","Input","Geometry"},
       BuiltInCase{"DrawIndex","%int32","Input","TessellationControl"},
       // 4438 = DeviceIndex supports all shader stages, thus cannot fail
       BuiltInCase{"ViewIndex","%int32","Input","GLCompute"},

       BuiltInCase{"BaryCoordNoPerspAMD","%vecf2","Input","TessellationControl"},
       BuiltInCase{"BaryCoordNoPerspCentroidAMD","%vecf2","Input","Geometry"},
       BuiltInCase{"BaryCoordNoPerspSampleAMD","%vecf2","Input","GLCompute"},
       BuiltInCase{"BaryCoordSmoothAMD","%vecf2","Input","TessellationEvaluation"},
       BuiltInCase{"BaryCoordSmoothCentroidAMD","%vecf2","Input","Vertex"},
       BuiltInCase{"BaryCoordSmoothSampleAMD","%vecf2","Input","GLCompute"},
       BuiltInCase{"BaryCoordPullModelAMD","%vecf3","Input","TessellationControl"},

       BuiltInCase{"FragStencilRefEXT","%int32","Output","TessellationEvaluation"},
       BuiltInCase{"ViewportMaskNV","%int32","Output","Fragment"},
       BuiltInCase{"SecondaryPositionNV","%vecf4","Input","GLCompute"},
       BuiltInCase{"SecondaryViewportMaskNV","%arri3","Output","Fragment"},
       BuiltInCase{"PositionPerViewNV","%vecf4","Input","Fragment"},
       BuiltInCase{"ViewportMaskPerViewNV","%arri4","Output","GLCompute"}
  ));
// clang-format on

TEST_P(ValidateBuiltInStageCL, BuiltInStageCL) {
  CompileSuccessfully(GenerateKernelCode(GetParam()),SPV_ENV_OPENCL_2_2);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr( GetParam().name +
                " built-in is restricted to certain EXECUTION MODELS "
                "(see SPIR-V, Vulkan, OpenGL, OpenCL specifications)"));
}

// clang-format off
INSTANTIATE_TEST_CASE_P(BuiltInStageCL, ValidateBuiltInStageCL,
  ::testing::Values(
       BuiltInCase{"NumWorkgroups","%vecu3","Input","Vertex"},
       // 25 = WorkgroupSize is tested individually
       BuiltInCase{"WorkgroupId","%vecu3","Input","TessellationControl"},
       BuiltInCase{"LocalInvocationId","%vecu3","Input","TessellationEvaluation"},
       BuiltInCase{"GlobalInvocationId","%vecu3","Input","Geometry"},
       BuiltInCase{"LocalInvocationIndex","%uint32","Input","Vertex"},
       
       BuiltInCase{"WorkDim","%uint32","Input","Vertex"},
       BuiltInCase{"GlobalSize","%vecu3","Input","Fragment"},
       BuiltInCase{"EnqueuedWorkgroupSize","%vecu3","Input","Geometry"},
       BuiltInCase{"GlobalOffset","%vecu3","Input","TessellationEvaluation"},
       BuiltInCase{"GlobalLinearId","%uint32","Input","TessellationControl"},
       BuiltInCase{"SubgroupSize","%uint32","Input","GLCompute"},
       BuiltInCase{"SubgroupMaxSize","%uint32","Input","Vertex"},
       BuiltInCase{"NumSubgroups","%uint32","Input","Fragment"},
       BuiltInCase{"NumEnqueuedSubgroups","%uint32","Input","Geometry"},
       BuiltInCase{"SubgroupId","%uint32","Input","TessellationEvaluation"},
       BuiltInCase{"SubgroupLocalInvocationId","%uint32","Input","GLCompute"}
  ));
// clang-format on

TEST_P(ValidateBuiltInStorage, BuiltInStorage) {
  CompileSuccessfully(GenerateShaderCode(GetParam()),SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr( GetParam().name +
                " built-in is restricted to certain STORAGE CLASSES,"
                " depending on the execution model (see SPIR-V spec)"));
}

// clang-format off
INSTANTIATE_TEST_CASE_P(BasicBuiltIn, ValidateBuiltInStorage,
  ::testing::Values(
       BuiltInCase{"Position","%vecf4","Input","Vertex"},
       BuiltInCase{"PointSize","%float","Input","Vertex"},
       BuiltInCase{"ClipDistance","%arrf2","Input","Vertex"},
       BuiltInCase{"CullDistance","%arrf3","Input","Vertex"},
       BuiltInCase{"VertexId","%int32","Output","Vertex"},
       BuiltInCase{"InstanceId","%int32","Output","Vertex"},
       BuiltInCase{"PrimitiveId","%int32","Output","Fragment"},
       BuiltInCase{"InvocationId","%int32","Output","Geometry"},
       BuiltInCase{"Layer","%int32","Output","Fragment"},
       BuiltInCase{"ViewportIndex","%int32","Input","Geometry"},
       BuiltInCase{"TessLevelOuter","%arrf4","Input","TessellationControl"},
       BuiltInCase{"TessLevelInner","%arrf2","Output","TessellationEvaluation"},
       BuiltInCase{"TessCoord","%vecf3","Output","TessellationEvaluation"},
       BuiltInCase{"PatchVertices","%int32","Output","TessellationControl"},
       BuiltInCase{"FragCoord","%vecf4","Output","Fragment"},
       BuiltInCase{"PointCoord","%vecf2","Output","Fragment"},
       BuiltInCase{"FrontFacing","%bool","Output","Fragment"},
       BuiltInCase{"SampleId","%int32","Output","Fragment"},
       BuiltInCase{"SamplePosition","%vecf2","Output","Fragment"},
       // 20 = SampleMask allows Input/Output and cannot fail this test
       BuiltInCase{"FragDepth","%float","Input","Fragment"},
       BuiltInCase{"HelperInvocation","%bool","Output","Fragment"},
       BuiltInCase{"NumWorkgroups","%veci3","Output","GLCompute"},
       // 25 = WorkgroupSize is tested individually
       BuiltInCase{"WorkgroupId","%veci3","Output","GLCompute"},
       BuiltInCase{"LocalInvocationId","%veci3","Output","GLCompute"},
       BuiltInCase{"GlobalInvocationId","%veci3","Output","GLCompute"},
       BuiltInCase{"LocalInvocationIndex","%int32","Output","GLCompute"},
       BuiltInCase{"VertexIndex","%int32","Output","Vertex"},
       BuiltInCase{"InstanceIndex","%int32","Output","Vertex"}
  ));

INSTANTIATE_TEST_CASE_P(ExtendedBuiltIn, ValidateBuiltInStorage,
  ::testing::Values(
       BuiltInCase{"SubgroupEqMaskKHR","%veci3","Output","Vertex"},
       BuiltInCase{"SubgroupGeMaskKHR","%veci3","Output","Geometry"},
       BuiltInCase{"SubgroupGtMaskKHR","%veci3","Output","TessellationControl"},
       BuiltInCase{"SubgroupLeMaskKHR","%veci3","Output","TessellationEvaluation"},
       BuiltInCase{"SubgroupLtMaskKHR","%veci3","Output","Fragment"},

       BuiltInCase{"BaseVertex","%int32","Output","Vertex"},
       BuiltInCase{"BaseInstance","%int32","Output","Vertex"},
       BuiltInCase{"DrawIndex","%int32","Output","Vertex"},
       BuiltInCase{"DeviceIndex","%int32","Output","GLCompute"},
       BuiltInCase{"ViewIndex","%int32","Output","Fragment"},

       BuiltInCase{"BaryCoordNoPerspAMD","%vecf2","Output","Fragment"},
       BuiltInCase{"BaryCoordNoPerspCentroidAMD","%vecf2","Output","Fragment"},
       BuiltInCase{"BaryCoordNoPerspSampleAMD","%vecf2","Output","Fragment"},
       BuiltInCase{"BaryCoordSmoothAMD","%vecf2","Output","Fragment"},
       BuiltInCase{"BaryCoordSmoothCentroidAMD","%vecf2","Output","Fragment"},
       BuiltInCase{"BaryCoordSmoothSampleAMD","%vecf2","Output","Fragment"},
       BuiltInCase{"BaryCoordPullModelAMD","%vecf3","Output","Fragment"},

       BuiltInCase{"FragStencilRefEXT","%int32","Input","Fragment"},
       BuiltInCase{"ViewportMaskNV","%int32","Input","Vertex"},
       BuiltInCase{"SecondaryPositionNV","%vecf4","Input","Vertex"},
       BuiltInCase{"SecondaryViewportMaskNV","%arri3","Input","Geometry"},
       BuiltInCase{"PositionPerViewNV","%vecf4","Input","Vertex"},
       BuiltInCase{"ViewportMaskPerViewNV","%arri4","Input","Geometry"}
  ));
// clang-format on

TEST_P(ValidateBuiltInStorageCL, BuiltInStorageCL) {
  CompileSuccessfully(GenerateKernelCode(GetParam()),SPV_ENV_OPENCL_2_2);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr( GetParam().name +
                " built-in is restricted to certain STORAGE CLASSES,"
                " depending on the execution model (see SPIR-V spec)"));
}

// clang-format off
INSTANTIATE_TEST_CASE_P(BuiltInStorageCL, ValidateBuiltInStorageCL,
  ::testing::Values(
       BuiltInCase{"NumWorkgroups","%vecu3","Output","Kernel"},
       // 25 = WorkgroupSize is tested individually
       BuiltInCase{"WorkgroupId","%vecu3","Output","Kernel"},
       BuiltInCase{"LocalInvocationId","%vecu3","Output","Kernel"},
       BuiltInCase{"GlobalInvocationId","%vecu3","Output","Kernel"},
       BuiltInCase{"LocalInvocationIndex","%uint32","Output","Kernel"},
       BuiltInCase{"WorkDim","%uint32","Output","Kernel"},
       BuiltInCase{"GlobalSize","%vecu3","Output","Kernel"},
       BuiltInCase{"EnqueuedWorkgroupSize","%vecu3","Output","Kernel"},
       BuiltInCase{"GlobalOffset","%vecu3","Output","Kernel"},
       BuiltInCase{"GlobalLinearId","%uint32","Output","Kernel"},
       BuiltInCase{"SubgroupSize","%uint32","Output","Kernel"},
       BuiltInCase{"SubgroupMaxSize","%uint32","Output","Kernel"},
       BuiltInCase{"NumSubgroups","%uint32","Output","Kernel"},
       BuiltInCase{"NumEnqueuedSubgroups","%uint32","Output","Kernel"},
       BuiltInCase{"SubgroupId","%uint32","Output","Kernel"},
       BuiltInCase{"SubgroupLocalInvocationId","%uint32","Output","Kernel"}
  ));
// clang-format on

TEST_P(ValidateBuiltInType, BuiltInType) {
  CompileSuccessfully(GenerateShaderCode(GetParam()),SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr( GetParam().name +
                " built-in must match the DATA TYPE defined"
                " in the specification (see SPIR-V specification)"));
}

// clang-format off
INSTANTIATE_TEST_CASE_P(BasicBuiltIn, ValidateBuiltInType,
  ::testing::Values(
       BuiltInCase{"Position","%int32","Output","Vertex"},
       BuiltInCase{"PointSize","%int32","Output","Vertex"},
       BuiltInCase{"ClipDistance","%vecf2","Output","Vertex"},
       BuiltInCase{"CullDistance","%float","Output","Vertex"},
       BuiltInCase{"VertexId","%arrf3","Input","Vertex"},
       BuiltInCase{"InstanceId","%float","Input","Vertex"},
       BuiltInCase{"PrimitiveId","%vecf4","Input","Fragment"},
       BuiltInCase{"InvocationId","%float","Input","Geometry"},
       BuiltInCase{"Layer","%arri3","Output","Geometry"},
       BuiltInCase{"ViewportIndex","%veci2","Input","Fragment"},
       BuiltInCase{"TessLevelOuter","%int32","Input","TessellationEvaluation"},
       BuiltInCase{"TessLevelInner","%arrf4","Output","TessellationControl"},
       BuiltInCase{"TessCoord","%veci3","Input","TessellationEvaluation"},
       BuiltInCase{"PatchVertices","%float","Input","TessellationControl"},
       BuiltInCase{"FragCoord","%vecf2","Input","Fragment"},
       BuiltInCase{"PointCoord","%vecf4","Input","Fragment"},
       BuiltInCase{"FrontFacing","%void","Input","Fragment"},
       BuiltInCase{"SampleId","%bool","Input","Fragment"},
       BuiltInCase{"SamplePosition","%veci2","Input","Fragment"},
       BuiltInCase{"SampleMask","%arrf3","Input","Fragment"},
       BuiltInCase{"FragDepth","%void","Output","Fragment"},
       BuiltInCase{"HelperInvocation","%void","Input","Fragment"},
       BuiltInCase{"NumWorkgroups","%arrf2","Input","GLCompute"},
       // 25 = WorkgroupSize is tested individually
       BuiltInCase{"WorkgroupId","%vecf3","Input","GLCompute"},
       BuiltInCase{"LocalInvocationId","%int32","Input","GLCompute"},
       BuiltInCase{"GlobalInvocationId","%float","Input","GLCompute"},
       BuiltInCase{"LocalInvocationIndex","%veci3","Input","GLCompute"},
       BuiltInCase{"VertexIndex","%vecf2","Input","Vertex"},
       BuiltInCase{"InstanceIndex","%arri4","Input","Vertex"}
  ));

INSTANTIATE_TEST_CASE_P(ExtendedBuiltIn, ValidateBuiltInType,
  ::testing::Values(
       BuiltInCase{"SubgroupEqMaskKHR","%int32","Input","Vertex"},
       BuiltInCase{"SubgroupGeMaskKHR","%bool","Input","Geometry"},
       BuiltInCase{"SubgroupGtMaskKHR","%float","Input","TessellationControl"},
       BuiltInCase{"SubgroupLeMaskKHR","%vecf3","Input","TessellationEvaluation"},
       BuiltInCase{"SubgroupLtMaskKHR","%arri3","Input","Fragment"},

       BuiltInCase{"BaseVertex","%float","Input","Vertex"},
       BuiltInCase{"BaseInstance","%veci3","Input","Vertex"},
       BuiltInCase{"DrawIndex","%arrf2","Input","Vertex"},
       BuiltInCase{"DeviceIndex","%void","Input","GLCompute"},
       BuiltInCase{"ViewIndex","%bool","Input","Fragment"},

       BuiltInCase{"BaryCoordNoPerspAMD","%vecf3","Input","Fragment"},
       BuiltInCase{"BaryCoordNoPerspCentroidAMD","%vecf4","Input","Fragment"},
       BuiltInCase{"BaryCoordNoPerspSampleAMD","%veci2","Input","Fragment"},
       BuiltInCase{"BaryCoordSmoothAMD","%veci3","Input","Fragment"},
       BuiltInCase{"BaryCoordSmoothCentroidAMD","%veci4","Input","Fragment"},
       BuiltInCase{"BaryCoordSmoothSampleAMD","%float","Input","Fragment"},
       BuiltInCase{"BaryCoordPullModelAMD","%arrf3","Input","Fragment"},

       BuiltInCase{"FragStencilRefEXT","%bool","Output","Fragment"},
       BuiltInCase{"ViewportMaskNV","%arrf4","Output","Vertex"},
       BuiltInCase{"PositionPerViewNV","%arrf4","Input","TessellationControl"},
       BuiltInCase{"ViewportMaskPerViewNV","%void","Output","Geometry"}
  ));
// clang-format on

TEST_P(ValidateBuiltInTypeCL, BuiltInTypeCL) {
  CompileSuccessfully(GenerateKernelCode(GetParam()),SPV_ENV_OPENCL_2_2);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr( GetParam().name +
                " built-in must match the DATA TYPE defined"
                " in the specification (see SPIR-V specification)"));
}

// clang-format off
INSTANTIATE_TEST_CASE_P(BuiltInTypeCL, ValidateBuiltInTypeCL,
  ::testing::Values(
       BuiltInCase{"NumWorkgroups","%arrf2","Input","Kernel"},
       // 25 = WorkgroupSize is tested individually
       BuiltInCase{"WorkgroupId","%vecf3","Input","Kernel"},
       BuiltInCase{"LocalInvocationId","%uint32","Input","Kernel"},
       BuiltInCase{"GlobalInvocationId","%float","Input","Kernel"},
       BuiltInCase{"LocalInvocationIndex","%vecu3","Input","Kernel"},
       BuiltInCase{"WorkDim","%float","Input","Kernel"},
       BuiltInCase{"GlobalSize","%vecu2","Input","Kernel"},
       BuiltInCase{"EnqueuedWorkgroupSize","%vecf3","Input","Kernel"},
       BuiltInCase{"GlobalOffset","%vecu4","Input","Kernel"},
       BuiltInCase{"GlobalLinearId","%float","Input","Kernel"},
       BuiltInCase{"SubgroupSize","%void","Input","Kernel"},
       BuiltInCase{"SubgroupMaxSize","%arru3","Input","Kernel"},
       BuiltInCase{"NumSubgroups","%float","Input","Kernel"},
       BuiltInCase{"NumEnqueuedSubgroups","%vecf4","Input","Kernel"},
       BuiltInCase{"SubgroupId","%void","Input","Kernel"},
       BuiltInCase{"SubgroupLocalInvocationId","%vecu3","Input","Kernel"}
  ));
// clang-format on

//
// INDIVIDUAL TESTS
//

TEST_F(ValidateBuiltIn, PrimitiveIdAsFragmentInput) {
  string spirv = R"(
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    OpEntryPoint Fragment %entry "entry" %bltin
    OpDecorate %bltin BuiltIn PrimitiveId
    %void    = OpTypeVoid
    %int32  = OpTypeInt 32 0
    %voidfun = OpTypeFunction %void
    %ptr   = OpTypePointer Input %int32
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

// TODO(jcaraban): DepthReplacing model needed when writing to FragDepth
//                 more info in CheckFragDepth(), validate_builtins.cpp
#if 0
TEST_F(ValidateBuiltIn, FragDepthNeedsDepthReplacing) {
  string spirv = R"(
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    OpEntryPoint Fragment %entry "entry" %bltin
    OpDecorate %bltin BuiltIn FragDepth
    %void  = OpTypeVoid
    %float = OpTypeFloat 32
    %voidf = OpTypeFunction %void
    %1234f = OpConstant %float 1234
    %ptr   = OpTypePointer Output %float
    %bltin = OpVariable %ptr Output
    %entry = OpFunction %void None %voidf
    %label = OpLabel
             OpStore %builtin %1234f
             OpReturn
             OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID,ValidateAndRetrieveValidationState());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("To write to FragDepth, a shader must declare the "
                        "DepthReplacing execution mode (see Vulkan spec)"));
}
#endif

TEST_F(ValidateBuiltIn, WorkgroupSizeOK) {
  string spirv = R"(
    OpCapability Shader
    OpCapability Linkage
    OpMemoryModel Logical GLSL450
    OpDecorate %bltin BuiltIn WorkgroupSize
    %int   = OpTypeInt 32 1
    %16i   = OpConstant %int 16
    %1i    = OpConstant %int 1
    %vec3i = OpTypeVector %int 3
    %bltin = OpConstantComposite %vec3i %16i %16i %1i)";

    CompileSuccessfully(spirv);
    EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateBuiltInCL, WorkgroupSizeCLOK) {
  string spirv = R"(
    OpCapability Kernel
    OpCapability Addresses
    OpCapability Linkage
    OpMemoryModel Physical64 OpenCL
    OpDecorate %bltin BuiltIn WorkgroupSize
    %uint  = OpTypeInt 32 0
    %16u   = OpConstant %uint 16
    %1u    = OpConstant %uint 1
    %vec3u = OpTypeVector %uint 3
    %bltin = OpConstantComposite %vec3u %16u %16u %1u)";

    CompileSuccessfully(spirv);
    EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateBuiltIn, WorkgroupSizeBAD) {
  string spirv = R"(
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    OpEntryPoint GLCompute %entry "entry" %bltin
    OpExecutionMode %entry LocalSize 32 32 1
    OpDecorate %bltin BuiltIn WorkgroupSize
    %void  = OpTypeVoid
    %voidf = OpTypeFunction %void
    %int   = OpTypeInt 32 0
    %vec3u = OpTypeVector %int 3
    %ptr   = OpTypePointer Input %vec3u
    %bltin = OpVariable %ptr Input
    %entry = OpFunction %void None %voidf
    %label = OpLabel
             OpReturn
             OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID,ValidateAndRetrieveValidationState());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("The object decorated with WorkgroupSize must be "
                        "a specialization constant or a constant"));
}

}  // anonymous namespace
