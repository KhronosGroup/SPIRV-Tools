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

#include <cassert>
#include <cstring>

#include "spirv_validator_options.h"

bool spvParseUniversalLimitsOptions(const char* s, spv_validator_limit* type) {
  auto match = [s](const char* b) {
    return s && (0 == strncmp(s, b, strlen(b)));
  };
  if (match("--max-struct-members")) {
    *type = validator_limit_max_struct_members;
  } else if (match("--max-local-variables")) {
    *type = validator_limit_max_local_variables;
  } else if (match("--max-global-variables")) {
    *type = validator_limit_max_global_variables;
  } else {
    // The command line option for this validator limit has not been added.
    // Therefore we return false.
    return false;
  }

  return true;
}

spv_validator_options spvValidatorOptionsCreate() {
  return new spv_validator_options_t;
}

void spvValidatorOptionsDestroy(spv_validator_options options) {
  delete options;
}

void spvValidatorOptionsSetUniversalLimit(spv_validator_options options,
                                          spv_validator_limit limit_type,
                                          uint32_t limit) {
  assert(options && "Validator options object may not be Null");
  switch(limit_type) {
#define LIMIT(TYPE, FIELD)                    \
    case TYPE:                                \
      options->universalLimits.FIELD = limit; \
      break;
  LIMIT(validator_limit_max_struct_members, max_struct_members)
  LIMIT(validator_limit_max_local_variables, max_local_variables)
  LIMIT(validator_limit_max_global_variables, max_global_variables)
#undef LIMIT
  }
}
