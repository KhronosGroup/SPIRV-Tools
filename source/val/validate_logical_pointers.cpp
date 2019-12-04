// Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

spv_result_t LogicalPointersPass(ValidationState_t& _,
                                 const Instruction* inst) {
  if (_.options()->relax_logical_pointer) return SPV_SUCCESS;

  // Success if the instruction does not produce a pointer.
  if (!inst->type_id()) return SPV_SUCCESS;

  auto result_type = _.FindDef(inst->type_id());
  if (!result_type) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "Missing result type";
  }
  if (result_type->opcode() != SpvOpTypePointer) return SPV_SUCCESS;

  const auto storage_class = result_type->GetOperandAs<uint32_t>(1);

  // Success if the instruction does not produce a logical pointer.
  if (_.addressing_model() == SpvAddressingModelPhysicalStorageBuffer64) {
    if (storage_class == SpvStorageClassPhysicalStorageBuffer)
      return SPV_SUCCESS;
  } else if (_.addressing_model() != SpvAddressingModelLogical) {
    return SPV_SUCCESS;
  }

  if (!spvOpcodeReturnsLogicalVariablePointer(inst->opcode())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Logical pointers must not be produced by this opcode";
  }

  if (spvOpcodeGeneratesVariablePointer(inst->opcode())) {
    // Check capabilities and storage class.
    if (!_.features().variable_pointers &&
        !_.features().variable_pointers_storage_buffer) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Generating variable pointers requires capability "
             << "VariablePointers or VariablePointersStorageBuffer";
    }

    if (storage_class != SpvStorageClassWorkgroup &&
        storage_class != SpvStorageClassStorageBuffer) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Variable pointers must point to Workgroup or StorageBuffer "
                "storage classes";
    }

    if (storage_class == SpvStorageClassWorkgroup &&
        !_.features().variable_pointers) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Variable pointers to Workgroup storage class require "
                "capability VariablePointers";
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
