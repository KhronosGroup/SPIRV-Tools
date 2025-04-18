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

#include "source/val/validate_memory_semantics.h"

#include "source/spirv_target_env.h"
#include "source/util/bitutils.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

spv_result_t ValidateMemorySemantics(ValidationState_t& _,
                                     const Instruction* inst,
                                     uint32_t operand_index,
                                     uint32_t memory_scope) {
  const spv::Op opcode = inst->opcode();
  const auto id = inst->GetOperandAs<const uint32_t>(operand_index);
  const bool validate_vulkan = spvIsVulkanEnv(_.context()->target_env) ||
                               _.memory_model() == spv::MemoryModel::VulkanKHR;
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = _.EvalInt32IfConst(id);

  if (!is_int32) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": expected Memory Semantics to be a 32-bit int";
  }

  if (!is_const_int32) {
    if (_.HasCapability(spv::Capability::Shader) &&
        !_.HasCapability(spv::Capability::CooperativeMatrixNV)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Memory Semantics ids must be OpConstant when Shader "
                "capability is present";
    }

    if (_.HasCapability(spv::Capability::Shader) &&
        _.HasCapability(spv::Capability::CooperativeMatrixNV) &&
        !spvOpcodeIsConstant(_.GetIdOpcode(id))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Memory Semantics must be a constant instruction when "
                "CooperativeMatrixNV capability is present";
    }
    return SPV_SUCCESS;
  }

  if (value & uint32_t(spv::MemorySemanticsMask::UniformMemory) &&
      !_.HasCapability(spv::Capability::Shader)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics UniformMemory requires capability Shader";
  }

  if (value & uint32_t(spv::MemorySemanticsMask::OutputMemoryKHR) &&
      !_.HasCapability(spv::Capability::VulkanMemoryModel)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics OutputMemoryKHR requires capability "
           << "VulkanMemoryModelKHR";
  }

  const size_t num_memory_order_set_bits = spvtools::utils::CountSetBits(
      value & uint32_t(spv::MemorySemanticsMask::Acquire |
                       spv::MemorySemanticsMask::Release |
                       spv::MemorySemanticsMask::AcquireRelease |
                       spv::MemorySemanticsMask::SequentiallyConsistent));

  if (num_memory_order_set_bits > 1) {
    if (validate_vulkan) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(10001) << spvOpcodeString(opcode)
             << ": Memory Semantics must have at most one non-relaxed "
                "memory order bit set (Acquire, Release, or AcquireRelease)";
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Memory Semantics must have at most one non-relaxed "
              "memory order bit set (Acquire, Release, AcquireRelease, or "
              "SequentiallyConsistent)";
  }

  if (opcode == spv::Op::OpAtomicLoad &&
      (value & uint32_t(spv::MemorySemanticsMask::Release) ||
       value & uint32_t(spv::MemorySemanticsMask::AcquireRelease))) {
    if (validate_vulkan) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(10002) << spvOpcodeString(opcode)
             << " must have Relaxed or Acquire memory order";
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << " must have Relaxed, Acquire, or SequentiallyConsistent "
              "memory order";
  }

  if ((opcode == spv::Op::OpAtomicStore ||
       opcode == spv::Op::OpAtomicFlagClear) &&
      (value & uint32_t(spv::MemorySemanticsMask::Acquire) ||
       value & uint32_t(spv::MemorySemanticsMask::AcquireRelease))) {
    if (validate_vulkan) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(10003) << spvOpcodeString(opcode)
             << " must have Relaxed or Release memory order";
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << spvOpcodeString(opcode)
           << " must have Relaxed, Release, or SequentiallyConsistent "
              "memory order";
  }

  // In OpenCL, a relaxed fence has no effect but is not explicitly forbidden
  if (validate_vulkan && opcode == spv::Op::OpMemoryBarrier &&
      !num_memory_order_set_bits) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << _.VkErrorID(10004) << spvOpcodeString(opcode)
           << " must have Acquire, Release, or AcquireRelease memory order";
  }

  if (validate_vulkan &&
      (value & uint32_t(spv::MemorySemanticsMask::SequentiallyConsistent))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << _.VkErrorID(10005) << spvOpcodeString(opcode)
           << ": Memory Semantics must not have SequentiallyConsistent "
              "memory order in Vulkan environment";
  }

  if (validate_vulkan) {
    // According to the Vulkan specification, storage class semantics
    // SubgroupMemory, CrossWorkgroupMemory, and AtomicCounterMemory are ignored
    const bool includes_storage_class =
        value & uint32_t(spv::MemorySemanticsMask::UniformMemory |
                         spv::MemorySemanticsMask::WorkgroupMemory |
                         spv::MemorySemanticsMask::ImageMemory |
                         spv::MemorySemanticsMask::OutputMemoryKHR);

    if (!num_memory_order_set_bits && includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(10006) << spvOpcodeString(opcode)
             << ": Memory Semantics with at least one storage class semantics "
                "flag (UniformMemory, WorkgroupMemory, ImageMemory, or "
                "OutputMemory) must have a non-relaxed memory order (Acquire, "
                "Release, or AcquireRelease)";
    }

    if (num_memory_order_set_bits && !includes_storage_class) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(10007) << spvOpcodeString(opcode)
             << ": Memory Semantics with a non-relaxed memory order (Acquire, "
                "Release, or AcquireRelease) must have at least one storage "
                "class semantics flag (UniformMemory, WorkgroupMemory, "
                "ImageMemory, or OutputMemory)";
    }
  }

  if (value & uint32_t(spv::MemorySemanticsMask::MakeAvailableKHR)) {
    if (!_.HasCapability(spv::Capability::VulkanMemoryModel)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": Memory Semantics MakeAvailableKHR requires capability "
             << "VulkanMemoryModelKHR";
    }
    if (!(value & uint32_t(spv::MemorySemanticsMask::Release |
                           spv::MemorySemanticsMask::AcquireRelease))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(10008) << spvOpcodeString(opcode)
             << ": Memory Semantics with MakeAvailable flag must have Release "
                "or AcquireRelease memory order";
    }
  }

  if (value & uint32_t(spv::MemorySemanticsMask::MakeVisibleKHR)) {
    if (!_.HasCapability(spv::Capability::VulkanMemoryModel)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": Memory Semantics MakeVisibleKHR requires capability "
             << "VulkanMemoryModelKHR";
    }
    if (!(value & uint32_t(spv::MemorySemanticsMask::Acquire |
                           spv::MemorySemanticsMask::AcquireRelease))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(10009) << spvOpcodeString(opcode)
             << ": Memory Semantics with MakeVisible flag must have "
                "Acquire or AcquireRelease memory order";
    }
  }

  if (value & uint32_t(spv::MemorySemanticsMask::Volatile)) {
    if (!_.HasCapability(spv::Capability::VulkanMemoryModel)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": Memory Semantics Volatile requires capability "
                "VulkanMemoryModelKHR";
    }
    if (!spvOpcodeIsAtomicOp(inst->opcode())) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(10010) << spvOpcodeString(opcode)
             << ": Memory Semantics Volatile flag must not be used with "
                "barrier instructions (MemoryBarrier, ControlBarrier, "
                "and MemoryNamedBarrier)";
    }
  }

  if ((opcode == spv::Op::OpAtomicCompareExchange ||
       opcode == spv::Op::OpAtomicCompareExchangeWeak) &&
      operand_index == 5) {
    if (value & uint32_t(spv::MemorySemanticsMask::Release) ||
        value & uint32_t(spv::MemorySemanticsMask::AcquireRelease)) {
      if (validate_vulkan) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << _.VkErrorID(10011) << spvOpcodeString(opcode)
               << " Unequal Memory Semantics must have Relaxed or Acquire "
                  "memory order";
      }
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << " Unequal Memory Semantics must have Relaxed, Acquire, "
                "or SequentiallyConsistent memory order";
    }

    bool is_equal_int32 = false;
    bool is_equal_const = false;
    uint32_t equal_value = 0;
    std::tie(is_equal_int32, is_equal_const, equal_value) =
        _.EvalInt32IfConst(inst->GetOperandAs<uint32_t>(4));

    if ((value & uint32_t(spv::MemorySemanticsMask::SequentiallyConsistent) &&
         !(equal_value &
           uint32_t(spv::MemorySemanticsMask::SequentiallyConsistent)))) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << " Unequal Memory Semantics can have SequentiallyConsistent "
                "memory order only if Equal Memory Semantics "
                "also have SequentiallyConsistent memory order";
    }

    if ((value & uint32_t(spv::MemorySemanticsMask::Acquire)) &&
        !(equal_value & uint32_t(spv::MemorySemanticsMask::Acquire)) &&
        !(equal_value & uint32_t(spv::MemorySemanticsMask::AcquireRelease)) &&
        !(equal_value &
          uint32_t(spv::MemorySemanticsMask::SequentiallyConsistent))) {
      if (validate_vulkan) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << _.VkErrorID(10012) << spvOpcodeString(opcode)
               << " Unequal Memory Semantics can have Acquire memory order "
                  "only if Equal Memory Semantics have Acquire or "
                  "AcquireRelease memory order";
      }
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << spvOpcodeString(opcode)
             << " Unequal Memory Semantics can have Acquire memory order "
                "only if Equal Memory Semantics have Acquire, AcquireRelease, "
                "or SequentiallyConsistent memory order";
    }

    if (validate_vulkan) {
      auto mask = uint32_t(spv::MemorySemanticsMask::UniformMemory |
                           spv::MemorySemanticsMask::WorkgroupMemory |
                           spv::MemorySemanticsMask::ImageMemory |
                           spv::MemorySemanticsMask::OutputMemoryKHR |
                           spv::MemorySemanticsMask::MakeAvailableKHR |
                           spv::MemorySemanticsMask::MakeVisibleKHR);

      if (mask & ((~equal_value) & value)) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << _.VkErrorID(10013) << spvOpcodeString(opcode)
               << " Unequal Memory Semantics must not have any storage class "
                  "semantics flag (UniformMemory, WorkgroupMemory, "
                  "ImageMemory, or OutputMemory) or MakeVisible flag, "
                  "unless these flags are also present in the Equal "
                  "Memory Semantics";
      }

      if (((equal_value & uint32_t(spv::MemorySemanticsMask::Volatile)) ^
           (value & uint32_t(spv::MemorySemanticsMask::Volatile)))) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << _.VkErrorID(10014) << spvOpcodeString(opcode)
               << " Unequal Memory Semantics Volatile flag must match the "
                  "Equal Memory Semantics flag";
      }
    }
  }

  if (validate_vulkan && num_memory_order_set_bits) {
    bool memory_is_int32 = false, memory_is_const_int32 = false;
    uint32_t memory_value = 0;
    std::tie(memory_is_int32, memory_is_const_int32, memory_value) =
        _.EvalInt32IfConst(memory_scope);
    if (memory_is_int32 && spv::Scope(memory_value) == spv::Scope::Invocation) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4641) << spvOpcodeString(opcode)
             << ": Vulkan specification requires Memory Semantics to be None "
                "if used with Invocation Memory Scope";
    }
  }

  // TODO(atgoo@github.com) Add checks for OpenCL and OpenGL environments.

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
