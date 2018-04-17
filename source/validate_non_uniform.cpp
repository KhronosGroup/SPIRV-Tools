// Copyright (c) 2018 Google LLC.
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

// Validates correctness of barrier SPIR-V instructions.

#include "validate.h"

#include "diagnostic.h"
#include "opcode.h"
#include "spirv_constant.h"
#include "spirv_target_env.h"
#include "util/bitutils.h"
#include "val/instruction.h"
#include "val/validation_state.h"

namespace libspirv {

namespace {

spv_result_t ValidateExecutionScope(ValidationState_t& _,
                                    const spv_parsed_instruction_t* inst,
                                    uint32_t scope) {
  SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = _.EvalInt32IfConst(scope);

  if (!is_int32) {
    return _.diag(SPV_ERROR_INVALID_DATA)
           << spvOpcodeString(opcode)
           << ": expected Execution Scope to be a 32-bit int";
  }

  if (!is_const_int32) {
    return SPV_SUCCESS;
  }

  if (spvIsVulkanEnv(_.context()->target_env) &&
      _.context()->target_env != SPV_ENV_VULKAN_1_0 &&
      value != SpvScopeSubgroup) {
    return _.diag(SPV_ERROR_INVALID_DATA)
           << spvOpcodeString(opcode)
           << ": in Vulkan environment Execution scope is limited to "
              "Subgroup";
  }

  if (value != SpvScopeSubgroup && value != SpvScopeWorkgroup) {
    return _.diag(SPV_ERROR_INVALID_DATA) << spvOpcodeString(opcode)
                                          << ": Execution scope is limited to "
                                             "Subgroup or Workgroup";
  }

  return SPV_SUCCESS;
}

}  // namespace

// Validates correctness of non-uniform group instructions.
spv_result_t NonUniformPass(ValidationState_t& _,
                            const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);

  if (spvOpcodeIsNonUniformGroupOperation(opcode)) {
    const uint32_t execution_scope = inst->words[3];
    if (auto error = ValidateExecutionScope(_, inst, execution_scope)) {
      return error;
    }
  }

  return SPV_SUCCESS;
}

}  // namespace libspirv
