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

#ifndef LIBSPIRV_SPIRV_VALIDATOR_OPTIONS_H_
#define LIBSPIRV_SPIRV_VALIDATOR_OPTIONS_H_

#include "spirv-tools/libspirv.h"

// Return true if the command line option for the validator limit is valid (Also
// returns the Enum for option in this case). Returns false otherwise.
bool spvParseUniversalLimitsOptions(const char* s, spv_validator_limit* limit);

struct validator_universal_limits_t {
  uint32_t max_struct_members;
  uint32_t max_struct_depth;
  uint32_t max_local_variables;
  uint32_t max_global_variables;
  // ...
  // TODO: Add more limits here
  // ...
};

const validator_universal_limits_t kDefaultValidatorUniversalLimits = {
    /* max_struct_members */ 16383,
    /* max_struct_depth */ 255,
    /* max_local_variables */ 524287,
    /* max_global_variables */ 65535,
    // ...
    // TODO: Add more default values here
    // ...
};

// Manages command line options passed to the SPIR-V Validator. New struct
// members may be added for any new option.
struct spv_validator_options_t {
  spv_validator_options_t()
      : universalLimits(kDefaultValidatorUniversalLimits) {}

  validator_universal_limits_t universalLimits;
};

#endif  // LIBSPIRV_SPIRV_VALIDATOR_OPTIONS_H_

