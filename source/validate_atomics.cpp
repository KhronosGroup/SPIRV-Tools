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

// Validates correctness of atomic SPIR-V instructions.

#include "validate.h"

#include "diagnostic.h"
#include "opcode.h"
#include "val/instruction.h"
#include "val/validation_state.h"

namespace libspirv {

// Validates correctness of atomic instructions.
spv_result_t AtomicsPass(ValidationState_t& _,
                         const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  const uint32_t result_type = inst->type_id;

  switch (opcode) {
    case SpvOpAtomicLoad:
    case SpvOpAtomicStore:
    case SpvOpAtomicExchange:
    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:
    case SpvOpAtomicIIncrement:
    case SpvOpAtomicIDecrement:
    case SpvOpAtomicIAdd:
    case SpvOpAtomicISub:
    case SpvOpAtomicSMin:
    case SpvOpAtomicUMin:
    case SpvOpAtomicSMax:
    case SpvOpAtomicUMax:
    case SpvOpAtomicAnd:
    case SpvOpAtomicOr:
    case SpvOpAtomicXor:
    case SpvOpAtomicFlagTestAndSet:
    case SpvOpAtomicFlagClear: {
      if (opcode == SpvOpAtomicLoad || opcode == SpvOpAtomicExchange ||
          opcode == SpvOpAtomicCompareExchange) {
        if (!_.IsFloatScalarType(result_type) &&
            !_.IsIntScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Result Type to be int or float scalar type";
        }
      } else if (opcode == SpvOpAtomicFlagTestAndSet) {
        if (!_.IsBoolScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Result Type to be bool scalar type";
        }
      } else if (opcode == SpvOpAtomicFlagClear || opcode == SpvOpAtomicStore) {
        assert(result_type == 0);
      } else {
        if (!_.IsIntScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Result Type to be int scalar type";
        }
      }

      uint32_t operand_index =
          opcode == SpvOpAtomicFlagClear || opcode == SpvOpAtomicStore ? 0 : 2;
      const uint32_t pointer_type = _.GetOperandTypeId(inst, operand_index++);

      uint32_t data_type = 0;
      uint32_t storage_class = 0;
      if (!_.GetPointerTypeInfo(pointer_type, &data_type, &storage_class)) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Pointer to be of type OpTypePointer";
      }

      switch (storage_class) {
        case SpvStorageClassUniform:
        case SpvStorageClassWorkgroup:
        case SpvStorageClassCrossWorkgroup:
        case SpvStorageClassGeneric:
        case SpvStorageClassAtomicCounter:
        case SpvStorageClassImage:
        case SpvStorageClassStorageBuffer:
          break;
        default:
          return _.diag(SPV_ERROR_INVALID_DATA)
              << spvOpcodeString(opcode)
              << ": expected Pointer Storage Class to be Uniform, "
              << "Workgroup, CrossWorkgroup, Generic, AtomicCounter, Image or "
              << "StorageBuffer";
      }

      if (opcode == SpvOpAtomicFlagTestAndSet ||
          opcode == SpvOpAtomicFlagClear) {
        if (!_.IsIntScalarType(data_type) || _.GetBitWidth(data_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Pointer to point to a value of 32-bit int type";
        }
      } else if (opcode == SpvOpAtomicStore) {
        if (!_.IsFloatScalarType(data_type) && !_.IsIntScalarType(data_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Pointer to be a pointer to int or float "
                 << "scalar type";
        }
      } else {
        if (data_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Pointer to point to a value of type Result "
                    "Type";
        }
      }

      const uint32_t scope_type = _.GetOperandTypeId(inst, operand_index++);
      if (!_.IsIntScalarType(scope_type) || _.GetBitWidth(scope_type) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Scope to be 32-bit int";
      }

      const uint32_t memory_semantics_type1 =
          _.GetOperandTypeId(inst, operand_index++);
      if (!_.IsIntScalarType(memory_semantics_type1) ||
          _.GetBitWidth(memory_semantics_type1) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Memory Semantics to be 32-bit int";
      }

      if (opcode == SpvOpAtomicCompareExchange ||
          opcode == SpvOpAtomicCompareExchangeWeak) {
        const uint32_t memory_semantics_type2 =
            _.GetOperandTypeId(inst, operand_index++);
        if (!_.IsIntScalarType(memory_semantics_type2) ||
            _.GetBitWidth(memory_semantics_type2) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Memory Semantics to be 32-bit int";
        }
      }

      if (opcode == SpvOpAtomicStore) {
        const uint32_t value_type = _.GetOperandTypeId(inst, 3);
        if (value_type != data_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Value type and the type pointed to by Pointer "
                    "to"
                 << " be the same";
        }
      } else if (opcode != SpvOpAtomicLoad && opcode != SpvOpAtomicIIncrement &&
                 opcode != SpvOpAtomicIDecrement &&
                 opcode != SpvOpAtomicFlagTestAndSet &&
                 opcode != SpvOpAtomicFlagClear) {
        const uint32_t value_type = _.GetOperandTypeId(inst, operand_index++);
        if (value_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Value to be of type Result Type";
        }
      }

      if (opcode == SpvOpAtomicCompareExchange ||
          opcode == SpvOpAtomicCompareExchangeWeak) {
        const uint32_t comparator_type =
            _.GetOperandTypeId(inst, operand_index++);
        if (comparator_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Comparator to be of type Result Type";
        }
      }

      break;
    }

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace libspirv
