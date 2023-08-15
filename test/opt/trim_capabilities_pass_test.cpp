// Copyright (c) 2023 Google Inc.
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

#include "spirv-tools/optimizer.hpp"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using TrimCapabilitiesPassTest = PassTest<::testing::Test>;

TEST_F(TrimCapabilitiesPassTest, CheckKnownAliasTransformations) {
  // Those are expected changes caused by the test process:
  //  - SPV is assembled. -> capability goes from text to number.
  //  - SPV is optimized.
  //  - SPV is disassembled -> capability goes from number to text.
  //  - CHECK rule compares both text versions.
  // Because some capabilities share the same number (aliases), the text
  // compared with the CHECK rules depends on which alias is the first on the
  // SPIRV-Headers enum. This could change, and we want to easily distinguish
  // real failure from alias order change. This test is only here to list known
  // alias transformations. If this test breaks, it's not a bug in the
  // optimization pass, but just the SPIRV-Headers enum order that has changed.
  // If that happens, tests needs to be updated to the correct alias is used in
  // the CHECK rule.
  const std::string kTest = R"(
               OpCapability Linkage
               OpCapability StorageUniform16
               OpCapability StorageUniformBufferBlock16
               OpCapability ShaderViewportIndexLayerNV
               OpCapability FragmentBarycentricNV
               OpCapability ShadingRateNV
               OpCapability ShaderNonUniformEXT
               OpCapability RuntimeDescriptorArrayEXT
               OpCapability InputAttachmentArrayDynamicIndexingEXT
               OpCapability UniformTexelBufferArrayDynamicIndexingEXT
               OpCapability StorageTexelBufferArrayDynamicIndexingEXT
               OpCapability UniformBufferArrayNonUniformIndexingEXT
               OpCapability SampledImageArrayNonUniformIndexingEXT
               OpCapability StorageBufferArrayNonUniformIndexingEXT
               OpCapability StorageImageArrayNonUniformIndexingEXT
               OpCapability InputAttachmentArrayNonUniformIndexingEXT
               OpCapability UniformTexelBufferArrayNonUniformIndexingEXT
               OpCapability StorageTexelBufferArrayNonUniformIndexingEXT
               OpCapability VulkanMemoryModelKHR
               OpCapability VulkanMemoryModelDeviceScopeKHR
               OpCapability PhysicalStorageBufferAddressesEXT
               OpCapability DemoteToHelperInvocationEXT
               OpCapability DotProductInputAllKHR
               OpCapability DotProductInput4x8BitKHR
               OpCapability DotProductInput4x8BitPackedKHR
               OpCapability DotProductKHR
; CHECK: OpCapability Linkage
; CHECK-NOT: OpCapability StorageUniform16
; CHECK-NOT: OpCapability StorageUniformBufferBlock16
; CHECK-NOT: OpCapability ShaderViewportIndexLayerNV
; CHECK-NOT: OpCapability FragmentBarycentricNV
; CHECK-NOT: OpCapability ShadingRateNV
; CHECK-NOT: OpCapability ShaderNonUniformEXT
; CHECK-NOT: OpCapability RuntimeDescriptorArrayEXT
; CHECK-NOT: OpCapability InputAttachmentArrayDynamicIndexingEXT
; CHECK-NOT: OpCapability UniformTexelBufferArrayDynamicIndexingEXT
; CHECK-NOT: OpCapability StorageTexelBufferArrayDynamicIndexingEXT
; CHECK-NOT: OpCapability UniformBufferArrayNonUniformIndexingEXT
; CHECK-NOT: OpCapability SampledImageArrayNonUniformIndexingEXT
; CHECK-NOT: OpCapability StorageBufferArrayNonUniformIndexingEXT
; CHECK-NOT: OpCapability StorageImageArrayNonUniformIndexingEXT
; CHECK-NOT: OpCapability InputAttachmentArrayNonUniformIndexingEXT
; CHECK-NOT: OpCapability UniformTexelBufferArrayNonUniformIndexingEXT
; CHECK-NOT: OpCapability StorageTexelBufferArrayNonUniformIndexingEXT
; CHECK-NOT: OpCapability VulkanMemoryModelKHR
; CHECK-NOT: OpCapability VulkanMemoryModelDeviceScopeKHR
; CHECK-NOT: OpCapability PhysicalStorageBufferAddressesEXT
; CHECK-NOT: OpCapability DemoteToHelperInvocationEXT
; CHECK-NOT: OpCapability DotProductInputAllKHR
; CHECK-NOT: OpCapability DotProductInput4x8BitKHR
; CHECK-NOT: OpCapability DotProductInput4x8BitPackedKHR
; CHECK-NOT: OpCapability DotProductKHR
; CHECK: OpCapability UniformAndStorageBuffer16BitAccess
; CHECK: OpCapability StorageBuffer16BitAccess
; CHECK: OpCapability ShaderViewportIndexLayerEXT
; CHECK: OpCapability FragmentBarycentricKHR
; CHECK: OpCapability FragmentDensityEXT
; CHECK: OpCapability ShaderNonUniform
; CHECK: OpCapability RuntimeDescriptorArray
; CHECK: OpCapability InputAttachmentArrayDynamicIndexing
; CHECK: OpCapability UniformTexelBufferArrayDynamicIndexing
; CHECK: OpCapability StorageTexelBufferArrayDynamicIndexing
; CHECK: OpCapability UniformBufferArrayNonUniformIndexing
; CHECK: OpCapability SampledImageArrayNonUniformIndexing
; CHECK: OpCapability StorageBufferArrayNonUniformIndexing
; CHECK: OpCapability StorageImageArrayNonUniformIndexing
; CHECK: OpCapability InputAttachmentArrayNonUniformIndexing
; CHECK: OpCapability UniformTexelBufferArrayNonUniformIndexing
; CHECK: OpCapability StorageTexelBufferArrayNonUniformIndexing
; CHECK: OpCapability VulkanMemoryModel
; CHECK: OpCapability VulkanMemoryModelDeviceScope
; CHECK: OpCapability PhysicalStorageBufferAddresses
; CHECK: OpCapability DemoteToHelperInvocation
; CHECK: OpCapability DotProductInputAll
; CHECK: OpCapability DotProductInput4x8Bit
; CHECK: OpCapability DotProductInput4x8BitPacked
; CHECK: OpCapability DotProduct
               OpMemoryModel Logical Vulkan
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
          %1 = OpFunction %void None %3
          %6 = OpLabel
               OpReturn
               OpFunctionEnd;
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result =
      SinglePassRunAndMatch<EmptyPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest, LinkagePreventsChanges) {
  const std::string kTest = R"(
               OpCapability Linkage
               OpCapability ClipDistance
               OpCapability CullDistance
               OpCapability DemoteToHelperInvocation
               OpCapability DeviceGroup
               OpCapability DrawParameters
               OpCapability Float16
               OpCapability Float64
               OpCapability FragmentBarycentricKHR
               OpCapability FragmentFullyCoveredEXT
               OpCapability FragmentShadingRateKHR
               OpCapability GroupNonUniform
               OpCapability GroupNonUniformArithmetic
               OpCapability GroupNonUniformBallot
               OpCapability GroupNonUniformQuad
               OpCapability GroupNonUniformShuffle
               OpCapability Image1D
               OpCapability ImageBuffer
               OpCapability ImageGatherExtended
               OpCapability ImageMSArray
               OpCapability ImageQuery
               OpCapability InputAttachment
               OpCapability InputAttachmentArrayNonUniformIndexing
               OpCapability Int16
               OpCapability Int64
               OpCapability Int64Atomics
               OpCapability Int64ImageEXT
               OpCapability MeshShadingNV
               OpCapability MinLod
               OpCapability MultiView
               OpCapability MultiViewport
               OpCapability PhysicalStorageBufferAddresses
               OpCapability RayQueryKHR
               OpCapability RayTracingKHR
               OpCapability RayTracingNV
               OpCapability RayTraversalPrimitiveCullingKHR
               OpCapability RuntimeDescriptorArray
               OpCapability SampleMaskPostDepthCoverage
               OpCapability SampleRateShading
               OpCapability Sampled1D
               OpCapability SampledBuffer
               OpCapability SampledImageArrayNonUniformIndexing
               OpCapability Shader
               OpCapability ShaderClockKHR
               OpCapability ShaderLayer
               OpCapability ShaderNonUniform
               OpCapability ShaderViewportIndex
               OpCapability ShaderViewportIndexLayerEXT
               OpCapability SparseResidency
               OpCapability StencilExportEXT
               OpCapability StorageImageArrayNonUniformIndexingEXT
               OpCapability StorageImageExtendedFormats
               OpCapability StorageImageReadWithoutFormat
               OpCapability StorageImageWriteWithoutFormat
               OpCapability StorageInputOutput16
               OpCapability StoragePushConstant16
               OpCapability StorageTexelBufferArrayNonUniformIndexing
               OpCapability StorageUniform16
               OpCapability StorageUniformBufferBlock16
               OpCapability Tessellation
               OpCapability UniformTexelBufferArrayNonUniformIndexing
               OpCapability VulkanMemoryModel
               OpExtension "SPV_EXT_fragment_fully_covered"
               OpExtension "SPV_EXT_shader_image_int64"
               OpExtension "SPV_EXT_shader_stencil_export"
               OpExtension "SPV_EXT_shader_viewport_index_layer"
               OpExtension "SPV_KHR_fragment_shader_barycentric"
               OpExtension "SPV_KHR_fragment_shading_rate"
               OpExtension "SPV_KHR_post_depth_coverage"
               OpExtension "SPV_KHR_ray_query"
               OpExtension "SPV_KHR_ray_tracing"
               OpExtension "SPV_KHR_shader_clock"
               OpExtension "SPV_NV_mesh_shader"
               OpExtension "SPV_NV_ray_tracing"
               OpExtension "SPV_NV_viewport_array2"
; CHECK: OpCapability Linkage
; CHECK: OpCapability ClipDistance
; CHECK: OpCapability CullDistance
; CHECK: OpCapability DemoteToHelperInvocation
; CHECK: OpCapability DeviceGroup
; CHECK: OpCapability DrawParameters
; CHECK: OpCapability Float16
; CHECK: OpCapability Float64
; CHECK: OpCapability FragmentBarycentricKHR
; CHECK: OpCapability FragmentFullyCoveredEXT
; CHECK: OpCapability FragmentShadingRateKHR
; CHECK: OpCapability GroupNonUniform
; CHECK: OpCapability GroupNonUniformArithmetic
; CHECK: OpCapability GroupNonUniformBallot
; CHECK: OpCapability GroupNonUniformQuad
; CHECK: OpCapability GroupNonUniformShuffle
; CHECK: OpCapability Image1D
; CHECK: OpCapability ImageBuffer
; CHECK: OpCapability ImageGatherExtended
; CHECK: OpCapability ImageMSArray
; CHECK: OpCapability ImageQuery
; CHECK: OpCapability InputAttachment
; CHECK: OpCapability InputAttachmentArrayNonUniformIndexing
; CHECK: OpCapability Int16
; CHECK: OpCapability Int64
; CHECK: OpCapability Int64Atomics
; CHECK: OpCapability Int64ImageEXT
; CHECK: OpCapability MeshShadingNV
; CHECK: OpCapability MinLod
; CHECK: OpCapability MultiView
; CHECK: OpCapability MultiViewport
; CHECK: OpCapability PhysicalStorageBufferAddresses
; CHECK: OpCapability RayQueryKHR
; CHECK: OpCapability RayTracingKHR
; CHECK: OpCapability RayTracingNV
; CHECK: OpCapability RayTraversalPrimitiveCullingKHR
; CHECK: OpCapability RuntimeDescriptorArray
; CHECK: OpCapability SampleMaskPostDepthCoverage
; CHECK: OpCapability SampleRateShading
; CHECK: OpCapability Sampled1D
; CHECK: OpCapability SampledBuffer
; CHECK: OpCapability SampledImageArrayNonUniformIndexing
; CHECK: OpCapability Shader
; CHECK: OpCapability ShaderClockKHR
; CHECK: OpCapability ShaderLayer
; CHECK: OpCapability ShaderNonUniform
; CHECK: OpCapability ShaderViewportIndex
; CHECK: OpCapability ShaderViewportIndexLayerEXT
; CHECK: OpCapability SparseResidency
; CHECK: OpCapability StencilExportEXT
; CHECK: OpCapability StorageImageArrayNonUniformIndexing
; CHECK: OpCapability StorageImageExtendedFormats
; CHECK: OpCapability StorageImageReadWithoutFormat
; CHECK: OpCapability StorageImageWriteWithoutFormat
; CHECK: OpCapability StorageInputOutput16
; CHECK: OpCapability StoragePushConstant16
; CHECK: OpCapability StorageTexelBufferArrayNonUniformIndexing
; CHECK: OpCapability Tessellation
; CHECK: OpCapability UniformTexelBufferArrayNonUniformIndex
; CHECK: OpCapability VulkanMemoryModel
; CHECK: OpExtension "SPV_EXT_fragment_fully_covered"
; CHECK: OpExtension "SPV_EXT_shader_image_int64"
; CHECK: OpExtension "SPV_EXT_shader_stencil_export"
; CHECK: OpExtension "SPV_EXT_shader_viewport_index_layer"
; CHECK: OpExtension "SPV_KHR_fragment_shader_barycentric"
; CHECK: OpExtension "SPV_KHR_fragment_shading_rate"
; CHECK: OpExtension "SPV_KHR_post_depth_coverage"
; CHECK: OpExtension "SPV_KHR_ray_query"
; CHECK: OpExtension "SPV_KHR_ray_tracing"
; CHECK: OpExtension "SPV_KHR_shader_clock"
; CHECK: OpExtension "SPV_NV_mesh_shader"
; CHECK: OpExtension "SPV_NV_ray_tracing"
; CHECK: OpExtension "SPV_NV_viewport_array2"
               OpMemoryModel Logical Vulkan
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
          %1 = OpFunction %void None %3
          %6 = OpLabel
               OpReturn
               OpFunctionEnd;
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest, KeepShader) {
  const std::string kTest = R"(
               OpCapability Shader
; CHECK: OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
          %1 = OpFunction %void None %3
          %6 = OpLabel
               OpReturn
               OpFunctionEnd;
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest, KeepShaderClockWhenInUse) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability Int64
               OpCapability ShaderClockKHR
               OpExtension "SPV_KHR_shader_clock"
; CHECK: OpCapability ShaderClockKHR
; CHECK: OpExtension "SPV_KHR_shader_clock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
      %ulong = OpTypeInt 64 0
      %scope = OpConstant %uint 1
          %3 = OpTypeFunction %void
          %1 = OpFunction %void None %3
          %6 = OpLabel
          %7 = OpReadClockKHR %ulong %scope
               OpReturn
               OpFunctionEnd;
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest, TrimShaderClockWhenUnused) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability Int64
               OpCapability ShaderClockKHR
               OpExtension "SPV_KHR_shader_clock"
; CHECK-NOT: OpCapability ShaderClockKHR
; CHECK-NOT: OpExtension "SPV_KHR_shader_clock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
          %1 = OpFunction %void None %3
          %6 = OpLabel
               OpReturn
               OpFunctionEnd;
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest, AMDShaderBallotExtensionRemains) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability Groups
               OpExtension "SPV_AMD_shader_ballot"
; CHECK: OpCapability Groups
; CHECK: OpExtension "SPV_AMD_shader_ballot"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
          %1 = OpTypeFunction %void
     %uint_0 = OpConstant %uint 0
          %2 = OpFunction %void None %1
          %3 = OpLabel
          %4 = OpGroupIAddNonUniformAMD %uint %uint_0 ExclusiveScan %uint_0
               OpReturn
               OpFunctionEnd;
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest, AMDShaderBallotExtensionRemoved) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability Groups
               OpExtension "SPV_AMD_shader_ballot"
; CHECK-NOT: OpCapability Groups
; CHECK-NOT: OpExtension "SPV_AMD_shader_ballot"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
       %void = OpTypeVoid
          %1 = OpTypeFunction %void
          %2 = OpFunction %void None %1
          %3 = OpLabel
               OpReturn
               OpFunctionEnd;
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest, MinLod_RemovedIfNotUsed) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Sampled1D
                      OpCapability MinLod
; CHECK-NOT:          OpCapability MinLod
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %1 "main"
              %void = OpTypeVoid
             %float = OpTypeFloat 32
           %v3float = OpTypeVector %float 3
           %v4float = OpTypeVector %float 4
        %type_image = OpTypeImage %float Cube 2 0 0 1 Rgba32f
    %ptr_type_image = OpTypePointer UniformConstant %type_image
      %type_sampler = OpTypeSampler
  %ptr_type_sampler = OpTypePointer UniformConstant %type_sampler
           %float_0 = OpConstant %float 0
         %float_000 = OpConstantComposite %v3float %float_0 %float_0 %float_0
             %image = OpVariable %ptr_type_image UniformConstant
           %sampler = OpVariable %ptr_type_sampler UniformConstant
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                %21 = OpLoad %type_image %image
                %22 = OpLoad %type_sampler %sampler
                %24 = OpSampledImage %type_sampled_image %21 %22
                %25 = OpImageSampleImplicitLod %v4float %24 %float_000
                      OpReturn
                      OpFunctionEnd;
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest, MinLod_RemainsWithOpImageSampleImplicitLod) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Sampled1D
                      OpCapability MinLod
; CHECK:              OpCapability MinLod
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %1 "main"
              %void = OpTypeVoid
             %float = OpTypeFloat 32
           %v3float = OpTypeVector %float 3
           %v4float = OpTypeVector %float 4
        %type_image = OpTypeImage %float Cube 2 0 0 1 Rgba32f
    %ptr_type_image = OpTypePointer UniformConstant %type_image
      %type_sampler = OpTypeSampler
  %ptr_type_sampler = OpTypePointer UniformConstant %type_sampler
           %float_0 = OpConstant %float 0
         %float_000 = OpConstantComposite %v3float %float_0 %float_0 %float_0
             %image = OpVariable %ptr_type_image UniformConstant
           %sampler = OpVariable %ptr_type_sampler UniformConstant
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                %21 = OpLoad %type_image %image
                %22 = OpLoad %type_sampler %sampler
                %24 = OpSampledImage %type_sampled_image %21 %22
                %25 = OpImageSampleImplicitLod %v4float %24 %float_000 MinLod %float_0
                      OpReturn
                      OpFunctionEnd;
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       MinLod_RemainsWithOpImageSparseSampleImplicitLod) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability SparseResidency
                      OpCapability ImageGatherExtended
                      OpCapability MinLod
; CHECK:              OpCapability MinLod
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint Fragment %2 "main"
                      OpExecutionMode %2 OriginUpperLeft
              %void = OpTypeVoid
              %uint = OpTypeInt 32 0
             %float = OpTypeFloat 32
           %v2float = OpTypeVector %float 2
           %v3float = OpTypeVector %float 3
           %v4float = OpTypeVector %float 4
        %type_image = OpTypeImage %float 2D 2 0 0 1 Unknown
    %ptr_type_image = OpTypePointer UniformConstant %type_image
      %type_sampler = OpTypeSampler
  %ptr_type_sampler = OpTypePointer UniformConstant %type_sampler
%type_sampled_image = OpTypeSampledImage %type_image
     %sparse_struct = OpTypeStruct %uint %v4float
           %float_0 = OpConstant %float 0
          %float_00 = OpConstantComposite %v2float %float_0 %float_0
         %float_000 = OpConstantComposite %v3float %float_0 %float_0 %float_0
             %image = OpVariable %ptr_type_image UniformConstant
           %sampler = OpVariable %ptr_type_sampler UniformConstant
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                %21 = OpLoad %type_image %image
                %22 = OpLoad %type_sampler %sampler
                %24 = OpSampledImage %type_sampled_image %21 %22
                %25 = OpImageSparseSampleImplicitLod %sparse_struct %24 %float_00 MinLod %float_0
                      OpReturn
                      OpFunctionEnd;
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest, MinLod_DetectsMinLodWithBitmaskImageOperand) {
  const std::string kTest = R"(
                            OpCapability MinLod
; CHECK:                    OpCapability MinLod
                            OpCapability Shader
                            OpCapability SparseResidency
                            OpCapability ImageGatherExtended
                            OpMemoryModel Logical GLSL450
                            OpEntryPoint Fragment %1 "main"
                            OpExecutionMode %1 OriginUpperLeft
            %type_sampler = OpTypeSampler
                     %int = OpTypeInt 32 1
                   %float = OpTypeFloat 32
                   %v2int = OpTypeVector %int 2
                 %v2float = OpTypeVector %float 2
                 %v4float = OpTypeVector %float 4
             %ptr_sampler = OpTypePointer UniformConstant %type_sampler
              %type_image = OpTypeImage %float 2D 2 0 0 1 Unknown
               %ptr_image = OpTypePointer UniformConstant %type_image
                    %void = OpTypeVoid
                    %uint = OpTypeInt 32 0
      %type_sampled_image = OpTypeSampledImage %type_image
             %type_struct = OpTypeStruct %uint %v4float

                   %int_1 = OpConstant %int 1
                 %float_0 = OpConstant %float 0
                 %float_1 = OpConstant %float 1
                       %8 = OpConstantComposite %v2float %float_0 %float_0
                      %12 = OpConstantComposite %v2int %int_1 %int_1

                       %2 = OpVariable %ptr_sampler UniformConstant
                       %3 = OpVariable %ptr_image UniformConstant
                      %27 = OpTypeFunction %void
                       %1 = OpFunction %void None %27
                      %28 = OpLabel
                      %29 = OpLoad %type_image %3
                      %30 = OpLoad %type_sampler %2
                      %31 = OpSampledImage %type_sampled_image %29 %30
                      %32 = OpImageSparseSampleImplicitLod %type_struct %31 %8 ConstOffset|MinLod %12 %float_0
                            OpReturn
                            OpFunctionEnd
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointer_Vulkan1_0) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
               %ptr = OpTypePointer Input %half
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointer_Vulkan1_1) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
               %ptr = OpTypePointer Input %half
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerArray_Vulkan1_0) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
              %uint = OpTypeInt 32 0
            %uint_1 = OpConstant %uint 1
             %array = OpTypeArray %half %uint_1
               %ptr = OpTypePointer Input %array
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerArray_Vulkan1_1) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
              %uint = OpTypeInt 32 0
            %uint_1 = OpConstant %uint 1
             %array = OpTypeArray %half %uint_1
               %ptr = OpTypePointer Input %array
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerStruct_Vulkan1_0) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
               %ptr = OpTypePointer Input %struct
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerStruct_Vulkan1_1) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
               %ptr = OpTypePointer Input %struct
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerStructOfStruct_Vulkan1_0) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
             %float = OpTypeFloat 32
            %struct = OpTypeStruct %float %half
            %parent = OpTypeStruct %float %struct
               %ptr = OpTypePointer Input %parent
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerStructOfStruct_Vulkan1_1) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
             %float = OpTypeFloat 32
            %struct = OpTypeStruct %float %half
            %parent = OpTypeStruct %float %struct
               %ptr = OpTypePointer Input %parent
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerArrayOfStruct_Vulkan1_0) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
              %uint = OpTypeInt 32 0
            %uint_1 = OpConstant %uint 1
             %array = OpTypeArray %struct %uint_1
               %ptr = OpTypePointer Input %array
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerArrayOfStruct_Vulkan1_1) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
              %uint = OpTypeInt 32 0
            %uint_1 = OpConstant %uint 1
             %array = OpTypeArray %struct %uint_1
               %ptr = OpTypePointer Input %array
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerVector_Vulkan1_0) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %vector = OpTypeVector %half 4
               %ptr = OpTypePointer Input %vector
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerVector_Vulkan1_1) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %vector = OpTypeVector %half 4
               %ptr = OpTypePointer Input %vector
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerMatrix_Vulkan1_0) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %vector = OpTypeVector %half 4
            %matrix = OpTypeMatrix %vector 4
               %ptr = OpTypePointer Input %matrix
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithInputPointerMatrix_Vulkan1_1) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %vector = OpTypeVector %half 4
            %matrix = OpTypeMatrix %vector 4
               %ptr = OpTypePointer Input %matrix
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_IsRemovedWithoutInputPointer) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK-NOT:          OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithOutputPointer_Vulkan1_0) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
               %ptr = OpTypePointer Output %half
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemainsWithOutputPointer_Vulkan1_1) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
               %ptr = OpTypePointer Output %half
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageInputOutput16_RemovedWithoutOutputPointer) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageInputOutput16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK-NOT:          OpCapability StorageInputOutput16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StoragePushConstant16_RemainsSimplePointer_Vulkan1_0) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StoragePushConstant16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StoragePushConstant16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
               %ptr = OpTypePointer PushConstant %half
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StoragePushConstant16_RemainsSimplePointer_Vulkan1_1) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StoragePushConstant16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK:              OpCapability StoragePushConstant16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
               %ptr = OpTypePointer PushConstant %half
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest, StoragePushConstant16_RemovedSimplePointer) {
  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StoragePushConstant16
                      OpExtension "SPV_KHR_16bit_storage"
; CHECK-NOT:          OpCapability StoragePushConstant16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
               %ptr = OpTypePointer Function %half
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageUniformBufferBlock16_RemainsSimplePointer_Vulkan1_0) {
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/5354
  static_assert(spv::Capability::StorageUniformBufferBlock16 ==
                spv::Capability::StorageBuffer16BitAccess);

  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageBuffer16BitAccess
                      OpExtension "SPV_KHR_16bit_storage"

; CHECK:              OpCapability StorageBuffer16BitAccess
;                                   `-> StorageUniformBufferBlock16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"

                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
                      OpDecorate %struct BufferBlock
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
               %ptr = OpTypePointer Uniform %struct
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageUniformBufferBlock16_RemainsSimplePointer_Vulkan1_1) {
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/5354
  static_assert(spv::Capability::StorageUniformBufferBlock16 ==
                spv::Capability::StorageBuffer16BitAccess);

  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageBuffer16BitAccess
                      OpExtension "SPV_KHR_16bit_storage"

; CHECK:              OpCapability StorageBuffer16BitAccess
;                                   `-> StorageUniformBufferBlock16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"

                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
                      OpDecorate %struct BufferBlock
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
               %ptr = OpTypePointer Uniform %struct
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageUniformBufferBlock16_RemovedSimplePointer) {
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/5354
  static_assert(spv::Capability::StorageUniformBufferBlock16 ==
                spv::Capability::StorageBuffer16BitAccess);

  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageBuffer16BitAccess
                      OpExtension "SPV_KHR_16bit_storage"

; CHECK-NOT:          OpCapability StorageBuffer16BitAccess
;                                   `-> StorageUniformBufferBlock16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"

                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
               %ptr = OpTypePointer Function %struct
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageUniform16_RemovedWithBufferBlockPointer_Vulkan1_0) {
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/5354
  static_assert(spv::Capability::StorageUniformBufferBlock16 ==
                spv::Capability::StorageBuffer16BitAccess);
  static_assert(spv::Capability::StorageUniform16 ==
                spv::Capability::UniformAndStorageBuffer16BitAccess);

  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageBuffer16BitAccess
                      OpCapability UniformAndStorageBuffer16BitAccess
                      OpExtension "SPV_KHR_16bit_storage"

; CHECK:              OpCapability StorageBuffer16BitAccess
;                                   `-> StorageUniformBufferBlock16
; CHECK-NOT:          OpCapability UniformAndStorageBuffer16BitAccess
;                                   `-> StorageUniform16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"

                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
                      OpDecorate %struct BufferBlock
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
               %ptr = OpTypePointer Uniform %struct
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageUniform16_RemovedWithBufferBlockPointer_Vulkan1_1) {
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/5354
  static_assert(spv::Capability::StorageUniformBufferBlock16 ==
                spv::Capability::StorageBuffer16BitAccess);
  static_assert(spv::Capability::StorageUniform16 ==
                spv::Capability::UniformAndStorageBuffer16BitAccess);

  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageBuffer16BitAccess
                      OpCapability UniformAndStorageBuffer16BitAccess
                      OpExtension "SPV_KHR_16bit_storage"

; CHECK:              OpCapability StorageBuffer16BitAccess
;                                   `-> StorageUniformBufferBlock16
; CHECK-NOT:          OpCapability UniformAndStorageBuffer16BitAccess
;                                   `-> StorageUniform16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"

                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
                      OpDecorate %struct BufferBlock
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
               %ptr = OpTypePointer Uniform %struct
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageUniform16_RemovedWithNonBlockUniformPointer_Vulkan1_0) {
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/5354
  static_assert(spv::Capability::StorageUniformBufferBlock16 ==
                spv::Capability::StorageBuffer16BitAccess);
  static_assert(spv::Capability::StorageUniform16 ==
                spv::Capability::UniformAndStorageBuffer16BitAccess);

  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageBuffer16BitAccess
                      OpCapability UniformAndStorageBuffer16BitAccess
                      OpExtension "SPV_KHR_16bit_storage"

; CHECK-NOT:          OpCapability StorageBuffer16BitAccess
;                                   `-> StorageUniformBufferBlock16
; CHECK:              OpCapability UniformAndStorageBuffer16BitAccess
;                                   `-> StorageUniform16
; CHECK:              OpExtension "SPV_KHR_16bit_storage"

                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
               %ptr = OpTypePointer Uniform %struct
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_0);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimCapabilitiesPassTest,
       StorageUniform16_RemovedWithNonBlockUniformPointer_Vulkan1_1) {
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/5354
  static_assert(spv::Capability::StorageUniformBufferBlock16 ==
                spv::Capability::StorageBuffer16BitAccess);
  static_assert(spv::Capability::StorageUniform16 ==
                spv::Capability::UniformAndStorageBuffer16BitAccess);

  const std::string kTest = R"(
                      OpCapability Shader
                      OpCapability Float16
                      OpCapability StorageBuffer16BitAccess
                      OpCapability UniformAndStorageBuffer16BitAccess
                      OpExtension "SPV_KHR_16bit_storage"

; CHECK-NOT:          OpCapability StorageBuffer16BitAccess
;                                   `-> StorageUniformBufferBlock16
; CHECK:              OpCapability UniformAndStorageBuffer16BitAccess
;                                   `-> StorageUniform16
; CHECK-NOT:          OpExtension "SPV_KHR_16bit_storage"

                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %2 "main"
              %void = OpTypeVoid
              %half = OpTypeFloat 16
            %struct = OpTypeStruct %half
               %ptr = OpTypePointer Uniform %struct
                 %1 = OpTypeFunction %void
                 %2 = OpFunction %void None %1
                 %3 = OpLabel
                      OpReturn
                      OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto result =
      SinglePassRunAndMatch<TrimCapabilitiesPass>(kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
