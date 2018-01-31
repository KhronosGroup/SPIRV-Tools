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

#include "latest_version_spirv_header.h"
#include "validate.h"

#include <tuple>

#include "diagnostic.h"
#include "opcode.h"
#include "spirv_target_env.h"
#include "val/instruction.h"
#include "val/validation_state.h"

namespace libspirv {

namespace {

// Returns number of '1' bits in a word.
uint32_t CountSetBits(uint32_t word) {
  uint32_t count = 0;
  while (word) {
    word &= word - 1;
    ++count;
  }
  return count;
}

// Evaluates int32 value from |id| if possible,
// returning tuple <is_int32, is_const_int32, value>.
std::tuple<bool, bool, uint32_t> EvalInt32IfConst(ValidationState_t& _,
                                                  uint32_t id) {
  const Instruction* const inst = _.FindDef(id);
  assert(inst);
  const uint32_t type = inst->type_id();

  if (!_.IsIntScalarType(type) || _.GetBitWidth(type) != 32) {
    return std::make_tuple(false, false, 0);
  }

  if (inst->opcode() != SpvOpConstant && inst->opcode() != SpvOpSpecConstant) {
    return std::make_tuple(true, false, 0);
  }

  assert(inst->words().size() == 4);
  return std::make_tuple(true, true, inst->word(3));
}

// Validates Execution Scope operand.
spv_result_t ValidateExecutionScope(ValidationState_t& _,
                                    const spv_parsed_instruction_t* inst,
                                    uint32_t id) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = EvalInt32IfConst(_, id);

  if (!is_int32) {
    return _.diag(SPV_ERROR_INVALID_DATA)
           << spvOpcodeString(opcode)
           << ": expected Execution Scope to be a 32-bit int";
  }

  if (!is_const_int32) {
    return SPV_SUCCESS;
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (value != SpvScopeWorkgroup && value != SpvScopeSubgroup) {
      return _.diag(SPV_ERROR_INVALID_DATA)
             << spvOpcodeString(opcode)
             << ": in Vulkan environment Execution Scope is limited to "
                "Workgroup and Subgroup";
    }
  }

  return SPV_SUCCESS;
}

// Validates Memory Scope operand.
spv_result_t ValidateMemoryScope(ValidationState_t& _,
                                 const spv_parsed_instruction_t* inst,
                                 uint32_t id) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = EvalInt32IfConst(_, id);

  if (!is_int32) {
    return _.diag(SPV_ERROR_INVALID_DATA)
           << spvOpcodeString(opcode)
           << ": expected Memory Scope to be a 32-bit int";
  }

  if (!is_const_int32) {
    return SPV_SUCCESS;
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (value != SpvScopeDevice && value != SpvScopeWorkgroup &&
        value != SpvScopeInvocation) {
      return _.diag(SPV_ERROR_INVALID_DATA)
             << spvOpcodeString(opcode)
             << ": in Vulkan environment Memory Scope is limited to Device, "
                "Workgroup and Invocation";
    }
  }

  return SPV_SUCCESS;
}

// Validates Memory Semantics operand.
spv_result_t ValidateMemorySemantics(ValidationState_t& _,
                                     const spv_parsed_instruction_t* inst,
                                     uint32_t id) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = EvalInt32IfConst(_, id);

  if (!is_int32) {
    return _.diag(SPV_ERROR_INVALID_DATA)
           << spvOpcodeString(opcode)
           << ": expected Memory Semantics to be a 32-bit int";
  }

  if (!is_const_int32) {
    return SPV_SUCCESS;
  }

  const uint32_t num_memory_order_set_bits = CountSetBits(
      value & (SpvMemorySemanticsAcquireMask | SpvMemorySemanticsReleaseMask |
               SpvMemorySemanticsAcquireReleaseMask |
               SpvMemorySemanticsSequentiallyConsistentMask));

  if (num_memory_order_set_bits > 1) {
    return _.diag(SPV_ERROR_INVALID_DATA)
           << spvOpcodeString(opcode)
           << ": Memory Semantics can have at most one of the following bits "
              "set: Acquire, Release, AcquireRelease or SequentiallyConsistent";
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    // TODO(atgoo@github.com): the Vulkan spec is unclear if
    // SequentiallyConsistent is supported.

    const bool includes_storage_class =
        value & (SpvMemorySemanticsUniformMemoryMask |
                 SpvMemorySemanticsWorkgroupMemoryMask |
                 SpvMemorySemanticsImageMemoryMask);

    if (opcode == SpvOpMemoryBarrier && !includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA)
             << spvOpcodeString(opcode)
             << ": expected Memory Semantics to include a Vulkan-supported "
                "storage class";
    }

    if (opcode == SpvOpControlBarrier && value && !includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA)
             << spvOpcodeString(opcode)
             << ": expected Memory Semantics to include a Vulkan-supported "
                "storage class if Memory Semantics is not None";
    }
  }

  return SPV_SUCCESS;
}

}  // anonymous namespace

// Validates correctness of barrier instructions.
spv_result_t BarriersPass(ValidationState_t& _,
                          const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  const uint32_t result_type = inst->type_id;

  switch (opcode) {
    case SpvOpControlBarrier: {
      _.current_function().RegisterExecutionModelLimitation(
          [](SpvExecutionModel model, std::string* message) {
            if (model != SpvExecutionModelTessellationControl &&
                model != SpvExecutionModelGLCompute &&
                model != SpvExecutionModelKernel) {
              if (message) {
                *message =
                    "OpControlBarrier requires one of the following Execution "
                    "Models: TessellationControl, GLCompute or Kernel";
              }
              return false;
            }
            return true;
          });

      const uint32_t execution_scope = inst->words[1];
      const uint32_t memory_scope = inst->words[2];
      const uint32_t memory_semantics = inst->words[3];

      if (auto error = ValidateExecutionScope(_, inst, execution_scope)) {
        return error;
      }

      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      if (auto error = ValidateMemorySemantics(_, inst, memory_semantics)) {
        return error;
      }
      break;
    }

    case SpvOpMemoryBarrier: {
      const uint32_t memory_scope = inst->words[1];
      const uint32_t memory_semantics = inst->words[2];

      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      if (auto error = ValidateMemorySemantics(_, inst, memory_semantics)) {
        return error;
      }
      break;
    }

    case SpvOpNamedBarrierInitialize: {
      if (_.GetIdOpcode(result_type) != SpvOpTypeNamedBarrier) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Result Type to be OpTypeNamedBarrier";
      }

      const uint32_t subgroup_count_type = _.GetOperandTypeId(inst, 2);
      if (!_.IsIntScalarType(subgroup_count_type) ||
          _.GetBitWidth(subgroup_count_type) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Subgroup Count to be a 32-bit int";
      }
      break;
    }

    case SpvOpMemoryNamedBarrier: {
      const uint32_t named_barrier_type = _.GetOperandTypeId(inst, 0);
      if (_.GetIdOpcode(named_barrier_type) != SpvOpTypeNamedBarrier) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Named Barrier to be of type OpTypeNamedBarrier";
      }

      const uint32_t memory_scope = inst->words[2];
      const uint32_t memory_semantics = inst->words[3];

      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      if (auto error = ValidateMemorySemantics(_, inst, memory_semantics)) {
        return error;
      }
      break;
    }

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace libspirv
