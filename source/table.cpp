// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "table.h"

#include <cassert>
#include <utility>

spv_context spvContextCreate(spv_target_env env) {
  switch (env) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_2_2:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
      break;
    default:
      return nullptr;
  }

  spv_opcode_table opcode_table;
  spv_operand_table operand_table;
  spv_ext_inst_table ext_inst_table;

  spvOpcodeTableGet(&opcode_table, env);
  spvOperandTableGet(&operand_table, env);
  spvExtInstTableGet(&ext_inst_table, env);

  return new spv_context_t{env, opcode_table, operand_table, ext_inst_table,
                           nullptr /* a null default consumer */};
}

void spvContextDestroy(spv_context context) { delete context; }

spv_validator_options spvValidatorOptionsCreate() {
  return new spv_validator_options_t;
}

void spvValidatorOptionsDestroy(spv_validator_options options) {
  delete options;
}

void spvValidatorOptionsSetMaxStructMembers(spv_validator_options options,
                                            const char* limit) {
  assert(options && "Validator options object may not be Null");
  if (limit) {
    int limit_int;
    int success = sscanf(limit, "%d", &limit_int);
    // The Minimum limits are specified in the SPIR-V Spec, so we only apply an
    // increase in the limit.
    if (success && limit_int > options->max_struct_members) {
      options->max_struct_members = limit_int;
    }
  }
}

int spvValidatorOptionsGetMaxStructMembers(
    spv_const_validator_options options) {
  assert(options && "Validator options object may not be Null");
  return options->max_struct_members;
}

void SetContextMessageConsumer(spv_context context,
                               spvtools::MessageConsumer consumer) {
  context->consumer = std::move(consumer);
}
