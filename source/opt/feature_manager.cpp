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

#include "feature_manager.h"
#include <queue>
#include <stack>

#include "enum_string_mapping.h"

namespace spvtools {
namespace opt {

void FeatureManager::Analyze(ir::Module* module) {
  AddExtensions(module);
  AddCapabilities(module);
}

void FeatureManager::AddExtensions(ir::Module* module) {
  for (auto ext : module->extensions()) {
    const std::string name =
        reinterpret_cast<const char*>(ext.GetInOperand(0u).words.data());
    libspirv::Extension extension;
    if (libspirv::GetExtensionFromString(name, &extension)) {
      extensions_.Add(extension);
    }
  }
}

void FeatureManager::AddCapabilities(ir::Module* module) {
  std::stack<SpvCapability> capabilities_to_add;
  for (ir::Instruction& inst : module->capabilities()) {
    SpvCapability cap =
        static_cast<SpvCapability>(inst.GetSingleWordInOperand(0));
    if (!capabilities_.Contains(cap)) {
      capabilities_to_add.push(cap);
      capabilities_.Add(cap);
    }
  }

  while (!capabilities_to_add.empty()) {
    SpvCapability cap = capabilities_to_add.top();
    capabilities_to_add.pop();

    switch (cap) {
      case SpvCapabilityShader:
        if (!capabilities_.Contains(SpvCapabilityMatrix)) {
          capabilities_to_add.push(SpvCapabilityMatrix);
          capabilities_.Add(SpvCapabilityMatrix);
        }
        break;
      case SpvCapabilityGeometry:
      case SpvCapabilityTessellation:
      case SpvCapabilityAtomicStorage:
      case SpvCapabilityImageGatherExtended:
      case SpvCapabilityStorageImageMultisample:
      case SpvCapabilityUniformBufferArrayDynamicIndexing:
      case SpvCapabilitySampledImageArrayDynamicIndexing:
      case SpvCapabilityStorageBufferArrayDynamicIndexing:
      case SpvCapabilityStorageImageArrayDynamicIndexing:
      case SpvCapabilityClipDistance:
      case SpvCapabilityCullDistance:
      case SpvCapabilitySampleRateShading:
      case SpvCapabilitySampledRect:
      case SpvCapabilityInputAttachment:
      case SpvCapabilitySparseResidency:
      case SpvCapabilityMinLod:
      case SpvCapabilitySampledCubeArray:
      case SpvCapabilityImageMSArray:
      case SpvCapabilityStorageImageExtendedFormats:
      case SpvCapabilityImageQuery:
      case SpvCapabilityDerivativeControl:
      case SpvCapabilityInterpolationFunction:
      case SpvCapabilityTransformFeedback:
      case SpvCapabilityStorageImageReadWithoutFormat:
      case SpvCapabilityStorageImageWriteWithoutFormat:
      case SpvCapabilityMultiView:
      case SpvCapabilityVariablePointersStorageBuffer:
      case SpvCapabilityImageGatherBiasLodAMD:
      // case SpvCapabilityFragmentMaskAMD:
      case SpvCapabilityStencilExportEXT:
      case SpvCapabilityImageReadWriteLodAMD:
        if (!capabilities_.Contains(SpvCapabilityShader)) {
          capabilities_to_add.push(SpvCapabilityShader);
          capabilities_.Add(SpvCapabilityShader);
        }
        break;
      case SpvCapabilityVector16:
      case SpvCapabilityFloat16Buffer:
      case SpvCapabilityImageBasic:
      case SpvCapabilityPipes:
      case SpvCapabilityDeviceEnqueue:
      case SpvCapabilityLiteralSampler:
      case SpvCapabilityInt8:
      case SpvCapabilityNamedBarrier:
        if (!capabilities_.Contains(SpvCapabilityKernel)) {
          capabilities_to_add.push(SpvCapabilityKernel);
          capabilities_.Add(SpvCapabilityKernel);
        }
        break;
      case SpvCapabilityInt64Atomics:
        if (!capabilities_.Contains(SpvCapabilityInt64)) {
          capabilities_to_add.push(SpvCapabilityInt64);
          capabilities_.Add(SpvCapabilityInt64);
        }
        break;
      case SpvCapabilityImageReadWrite:
      case SpvCapabilityImageMipmap:
        if (!capabilities_.Contains(SpvCapabilityImageBasic)) {
          capabilities_to_add.push(SpvCapabilityImageBasic);
          capabilities_.Add(SpvCapabilityImageBasic);
        }
        break;
      case SpvCapabilityTessellationPointSize:
        if (!capabilities_.Contains(SpvCapabilityTessellation)) {
          capabilities_to_add.push(SpvCapabilityTessellation);
          capabilities_.Add(SpvCapabilityTessellation);
        }
        break;
      case SpvCapabilityGeometryPointSize:
      case SpvCapabilityGeometryStreams:
      case SpvCapabilityMultiViewport:
      case SpvCapabilityGeometryShaderPassthroughNV:
        if (!capabilities_.Contains(SpvCapabilityGeometry)) {
          capabilities_to_add.push(SpvCapabilityGeometry);
          capabilities_.Add(SpvCapabilityGeometry);
        }
        break;
      case SpvCapabilityImageCubeArray:
        if (!capabilities_.Contains(SpvCapabilitySampledCubeArray)) {
          capabilities_to_add.push(SpvCapabilitySampledCubeArray);
          capabilities_.Add(SpvCapabilitySampledCubeArray);
        }
        break;
      case SpvCapabilityImageRect:
        if (!capabilities_.Contains(SpvCapabilitySampledRect)) {
          capabilities_to_add.push(SpvCapabilitySampledRect);
          capabilities_.Add(SpvCapabilitySampledRect);
        }
        break;
      case SpvCapabilityGenericPointer:
        if (!capabilities_.Contains(SpvCapabilityAddresses)) {
          capabilities_to_add.push(SpvCapabilityAddresses);
          capabilities_.Add(SpvCapabilityAddresses);
        }
        break;
      case SpvCapabilityImage1D:
        if (!capabilities_.Contains(SpvCapabilitySampled1D)) {
          capabilities_to_add.push(SpvCapabilitySampled1D);
          capabilities_.Add(SpvCapabilitySampled1D);
        }
        break;
      case SpvCapabilityImageBuffer:
        if (!capabilities_.Contains(SpvCapabilitySampledBuffer)) {
          capabilities_to_add.push(SpvCapabilitySampledBuffer);
          capabilities_.Add(SpvCapabilitySampledBuffer);
        }
        break;
      case SpvCapabilitySubgroupDispatch:
        if (!capabilities_.Contains(SpvCapabilityDeviceEnqueue)) {
          capabilities_to_add.push(SpvCapabilityDeviceEnqueue);
          capabilities_.Add(SpvCapabilityDeviceEnqueue);
        }
        break;
      case SpvCapabilityPipeStorage:
        if (!capabilities_.Contains(SpvCapabilityPipes)) {
          capabilities_to_add.push(SpvCapabilityPipes);
          capabilities_.Add(SpvCapabilityPipes);
        }
        break;
      case SpvCapabilityStorageBuffer16BitAccess:
      case SpvCapabilityStorageUniform16:
        if (!capabilities_.Contains(SpvCapabilityStorageBuffer16BitAccess)) {
          capabilities_to_add.push(SpvCapabilityStorageBuffer16BitAccess);
          capabilities_.Add(SpvCapabilityStorageBuffer16BitAccess);
        }
        if (!capabilities_.Contains(SpvCapabilityStorageUniformBufferBlock16)) {
          capabilities_to_add.push(SpvCapabilityStorageUniformBufferBlock16);
          capabilities_.Add(SpvCapabilityStorageUniformBufferBlock16);
        }
        break;
      case SpvCapabilityVariablePointers:
        if (!capabilities_.Contains(
                SpvCapabilityVariablePointersStorageBuffer)) {
          capabilities_to_add.push(SpvCapabilityVariablePointersStorageBuffer);
          capabilities_.Add(SpvCapabilityVariablePointersStorageBuffer);
        }
        break;
      case SpvCapabilitySampleMaskOverrideCoverageNV:
        if (!capabilities_.Contains(SpvCapabilitySampleRateShading)) {
          capabilities_to_add.push(SpvCapabilitySampleRateShading);
          capabilities_.Add(SpvCapabilitySampleRateShading);
        }
        break;
      case SpvCapabilityShaderViewportIndexLayerEXT:
        // case SpvCapabilityShaderViewportIndexLayerNV: Same value as line
        // above.
        if (!capabilities_.Contains(SpvCapabilityMultiViewport)) {
          capabilities_to_add.push(SpvCapabilityMultiViewport);
          capabilities_.Add(SpvCapabilityMultiViewport);
        }
        break;
      case SpvCapabilityShaderViewportMaskNV:
        if (!capabilities_.Contains(SpvCapabilityShaderViewportIndexLayerNV)) {
          capabilities_to_add.push(SpvCapabilityShaderViewportIndexLayerNV);
          capabilities_.Add(SpvCapabilityShaderViewportIndexLayerNV);
        }
        break;
      case SpvCapabilityShaderStereoViewNV:
        if (!capabilities_.Contains(SpvCapabilityShaderViewportMaskNV)) {
          capabilities_to_add.push(SpvCapabilityShaderViewportMaskNV);
          capabilities_.Add(SpvCapabilityShaderViewportMaskNV);
        }
        break;
      case SpvCapabilityPerViewAttributesNV:
        if (!capabilities_.Contains(SpvCapabilityMultiView)) {
          capabilities_to_add.push(SpvCapabilityMultiView);
          capabilities_.Add(SpvCapabilityMultiView);
        }
        break;
      default:
        break;
    }
  }
}

}  // namespace opt
}  // namespace spvtools
