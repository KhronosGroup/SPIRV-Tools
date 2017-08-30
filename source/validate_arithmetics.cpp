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

// Ensures type declarations are unique unless allowed by the specification.

#include "validate.h"

#include "diagnostic.h"
#include "opcode.h"
#include "val/instruction.h"
#include "val/validation_state.h"

namespace libspirv {

namespace {

// Returns operand word for given instruction and operand index.
// The operand is expected to only have one word.
inline uint32_t GetOperandWord(const spv_parsed_instruction_t* inst,
                               size_t operand_index) {
  assert(operand_index < inst->num_operands);
  const spv_parsed_operand_t& operand = inst->operands[operand_index];
  assert(operand.num_words == 1);
  return inst->words[operand.offset];
}

}

// Validates correctness of arithmetic instructions.
spv_result_t ArithmeticsPass(ValidationState_t& _,
                            const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);

  switch (opcode) {
    case SpvOpFAdd:
    case SpvOpFSub:
    case SpvOpFMul:
    case SpvOpFDiv:
    case SpvOpFRem:
    case SpvOpFMod:
    case SpvOpFNegate: {
      if (!_.IsFloatScalarType(inst->type_id) &&
          !_.IsFloatVectorType(inst->type_id))
        return _.diag(SPV_ERROR_INVALID_DATA)
            << "Expected floating scalar or vector type as type_id: "
            << spvOpcodeString(opcode);

      for (size_t operand_index = 2; operand_index < inst->num_operands;
           ++operand_index) {
        if (_.GetTypeId(GetOperandWord(inst, operand_index)) != inst->type_id)
          return _.diag(SPV_ERROR_INVALID_DATA)
              << "Expected arithmetic operands to have type type_id: "
              << spvOpcodeString(opcode) << " operand index " << operand_index;
      }
      break;
    }

    case SpvOpUDiv:
    case SpvOpUMod: {
      if (!_.IsUnsignedIntScalarType(inst->type_id) &&
          !_.IsUnsignedIntVectorType(inst->type_id))
        return _.diag(SPV_ERROR_INVALID_DATA)
            << "Expected unsigned int scalar or vector type as type_id: "
            << spvOpcodeString(opcode);

      for (size_t operand_index = 2; operand_index < inst->num_operands;
           ++operand_index) {
        if (_.GetTypeId(GetOperandWord(inst, operand_index)) != inst->type_id)
          return _.diag(SPV_ERROR_INVALID_DATA)
              << "Expected arithmetic operands to have type type_id: "
              << spvOpcodeString(opcode) << " operand index " << operand_index;
      }
      break;
    }

    case SpvOpISub:
    case SpvOpIAdd:
    case SpvOpIMul:
    case SpvOpSDiv:
    case SpvOpSMod:
    case SpvOpSRem:
    case SpvOpSNegate: {
      if (!_.IsIntScalarType(inst->type_id) &&
          !_.IsIntVectorType(inst->type_id))
        return _.diag(SPV_ERROR_INVALID_DATA)
            << "Expected int scalar or vector type as type_id: "
            << spvOpcodeString(opcode);

      const uint32_t dimension = _.GetDimension(inst->type_id);
      const uint32_t bit_width = _.GetBitWidth(inst->type_id);

      for (size_t operand_index = 2; operand_index < inst->num_operands;
           ++operand_index) {

        const uint32_t type_id =
            _.GetTypeId(GetOperandWord(inst, operand_index));
        if (!type_id ||
            (!_.IsIntScalarType(type_id) && !_.IsIntVectorType(type_id)))
          return _.diag(SPV_ERROR_INVALID_DATA)
              << "Expected int scalar or vector type as operand: "
              << spvOpcodeString(opcode) << " operand index " << operand_index;

        if (_.GetDimension(type_id) != dimension)
          return _.diag(SPV_ERROR_INVALID_DATA)
              << "Expected arithmetic operands to have the same dimension "
              << "as type_id: "
              << spvOpcodeString(opcode) << " operand index " << operand_index;

        if (_.GetBitWidth(type_id) != bit_width)
          return _.diag(SPV_ERROR_INVALID_DATA)
              << "Expected arithmetic operands to have the same bit width "
              << "as type_id: "
              << spvOpcodeString(opcode) << " operand index " << operand_index;
      }
      break;
    }

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace libspirv
