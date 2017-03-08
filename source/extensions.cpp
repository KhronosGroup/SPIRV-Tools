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

#include "extensions.h"

#include <string>

namespace libspirv {

bool ParseSpvExtensionFromString(const std::string& str, Extension* extension) {
  if (str == "SPV_KHR_shader_ballot") {
    *extension = Extension::kSPV_KHR_shader_ballot;
  } else if (str == "SPV_KHR_shader_draw_parameters") {
    *extension = Extension::kSPV_KHR_shader_draw_parameters;
  } else if (str == "SPV_KHR_subgroup_vote") {
    *extension = Extension::kSPV_KHR_subgroup_vote;
  } else if (str == "SPV_KHR_16bit_storage") {
    *extension = Extension::kSPV_KHR_16bit_storage;
  } else if (str == "SPV_KHR_device_group") {
    *extension = Extension::kSPV_KHR_device_group;
  } else if (str == "SPV_KHR_multiview") {
    *extension = Extension::kSPV_KHR_multiview;
  } else if (str == "SPV_NV_sample_mask_override_coverage") {
    *extension = Extension::kSPV_NV_sample_mask_override_coverage;
  } else if (str == "SPV_NV_geometry_shader_passthrough") {
    *extension = Extension::kSPV_NV_geometry_shader_passthrough;
  } else if (str == "SPV_NV_viewport_array2") {
    *extension = Extension::kSPV_NV_viewport_array2;
  } else if (str == "SPV_NV_stereo_view_rendering") {
    *extension = Extension::kSPV_NV_stereo_view_rendering;
  } else if (str == "SPV_NVX_multiview_per_view_attributes") {
    *extension = Extension::kSPV_NVX_multiview_per_view_attributes;
  } else {
    return false;
  }

  return true;
}

}  // namespace libspirv
