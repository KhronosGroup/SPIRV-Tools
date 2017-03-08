// Copyright (c) 2017 The Khronos Group Inc.
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

#ifndef LIBSPIRV_EXTENSIONS_H_
#define LIBSPIRV_EXTENSIONS_H_

#include <string>

#include "enum_set.h"
#include "message.h"
#include "spirv-tools/libspirv.hpp"

namespace libspirv {

// The known SPIR-V extensions.
// TODO(dneto): Consider auto-generating this list?
// When updating this list, consider also updating ParseSpvExtensionFromString.
enum class Extension {
  kSPV_KHR_shader_ballot,
  kSPV_KHR_shader_draw_parameters,
  kSPV_KHR_subgroup_vote,
  kSPV_KHR_16bit_storage,
  kSPV_KHR_device_group,
  kSPV_KHR_multiview,
  kSPV_NV_sample_mask_override_coverage,
  kSPV_NV_geometry_shader_passthrough,
  kSPV_NV_viewport_array2,
  kSPV_NV_stereo_view_rendering,
  kSPV_NVX_multiview_per_view_attributes,
};

using ExtensionSet = EnumSet<Extension>;

bool ParseSpvExtensionFromString(const std::string& str, Extension* extension);

}  // namespace libspirv

#endif  // LIBSPIRV_EXTENSIONS_H_
