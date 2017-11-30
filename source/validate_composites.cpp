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

// Validates correctness of composite SPIR-V instructions.

#include "validate.h"

#include "diagnostic.h"
#include "opcode.h"
#include "val/instruction.h"
#include "val/validation_state.h"

namespace libspirv {

// Validates correctness of composite instructions.
spv_result_t CompositesPass(ValidationState_t& _,
                            const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  const uint32_t result_type = inst->type_id;
  const uint32_t num_operands = static_cast<uint32_t>(inst->num_operands);

  switch (opcode) {
    case SpvOpVectorExtractDynamic: {
      const SpvOp result_opcode = _.GetIdOpcode(result_type);
      if (!spvOpcodeIsScalarType(result_opcode)) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Result Type to be a scalar type";
      }

      const uint32_t vector_type = _.GetOperandTypeId(inst, 2);
      const SpvOp vector_opcode = _.GetIdOpcode(vector_type);
      if (vector_opcode != SpvOpTypeVector) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Vector type to be OpTypeVector";
      }

      if (_.GetComponentType(vector_type) != result_type) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Vector component type to be equal to Result Type";
      }

      const uint32_t index_type = _.GetOperandTypeId(inst, 3);
      if (!_.IsIntScalarType(index_type)) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Index to be int scalar";
      }

      break;
    }

    case SpvOpVectorInsertDynamic: {
      const SpvOp result_opcode = _.GetIdOpcode(result_type);
      if (result_opcode != SpvOpTypeVector) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Result Type to be OpTypeVector";
      }

      const uint32_t vector_type = _.GetOperandTypeId(inst, 2);
      if (vector_type != result_type) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Vector type to be equal to Result Type";
      }

      const uint32_t component_type = _.GetOperandTypeId(inst, 3);
      if (_.GetComponentType(result_type) != component_type) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Component type to be equal to Result Type "
               << "component type";
      }

      const uint32_t index_type = _.GetOperandTypeId(inst, 4);
      if (!_.IsIntScalarType(index_type)) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Index to be int scalar";
      }

      break;
    }

    case SpvOpVectorShuffle: {
      // Handled in validate_id.cpp.
      // TODO(atgoo@github.com) Consider moving it here.
      break;
    }

    case SpvOpCompositeConstruct: {
      const SpvOp result_opcode = _.GetIdOpcode(result_type);
      switch (result_opcode) {
        case SpvOpTypeVector: {
          const uint32_t num_result_components = _.GetDimension(result_type);
          const uint32_t result_component_type =
              _.GetComponentType(result_type);
          uint32_t given_component_count = 0;

          if (num_operands <= 3) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << spvOpcodeString(opcode)
                   << ": expected number of constituents to be at least 2";
          }

          for (uint32_t operand_index = 2; operand_index < num_operands;
               ++operand_index) {
            const uint32_t operand_type =
                _.GetOperandTypeId(inst, operand_index);
            if (operand_type == result_component_type) {
              ++given_component_count;
            } else {
              if (_.GetIdOpcode(operand_type) != SpvOpTypeVector ||
                  _.GetComponentType(operand_type) != result_component_type) {
                return _.diag(SPV_ERROR_INVALID_DATA)
                       << spvOpcodeString(opcode)
                       << ": expected Constituents to be scalars or vectors of "
                       << "the same type as Result Type components";
              }

              given_component_count += _.GetDimension(operand_type);
            }
          }

          if (num_result_components != given_component_count) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << spvOpcodeString(opcode)
                   << ": expected total number of given components to be equal "
                   << "to the size of Result Type vector";
          }

          break;
        }
        case SpvOpTypeMatrix: {
          uint32_t result_num_rows = 0;
          uint32_t result_num_cols = 0;
          uint32_t result_col_type = 0;
          uint32_t result_component_type = 0;
          if (!_.GetMatrixTypeInfo(result_type, &result_num_rows,
                                   &result_num_cols, &result_col_type,
                                   &result_component_type)) {
            assert(0);
          }

          if (result_num_cols + 2 != num_operands) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << spvOpcodeString(opcode)
                   << ": expected total number of Constituents to be equal "
                   << "to the number of columns of Result Type matrix";
          }

          for (uint32_t operand_index = 2; operand_index < num_operands;
               ++operand_index) {
            const uint32_t operand_type =
                _.GetOperandTypeId(inst, operand_index);
            if (operand_type != result_col_type) {
              return _.diag(SPV_ERROR_INVALID_DATA)
                     << spvOpcodeString(opcode)
                     << ": expected Constituent type to be equal to the column "
                     << "type Result Type matrix";
            }
          }

          break;
        }
        case SpvOpTypeArray: {
          const Instruction* const array_inst = _.FindDef(result_type);
          assert(array_inst);
          assert(array_inst->opcode() == SpvOpTypeArray);

          uint64_t array_size = 0;
          if (!_.GetConstantValUint64(array_inst->word(3), &array_size)) {
            assert(0 && "Array type definition is corrupt");
          }

          if (array_size + 2 != num_operands) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << spvOpcodeString(opcode)
                   << ": expected total number of Constituents to be equal "
                   << "to the number of elements of Result Type array";
          }

          const uint32_t result_component_type = array_inst->word(2);
          for (uint32_t operand_index = 2; operand_index < num_operands;
               ++operand_index) {
            const uint32_t operand_type =
                _.GetOperandTypeId(inst, operand_index);
            if (operand_type != result_component_type) {
              return _.diag(SPV_ERROR_INVALID_DATA)
                     << spvOpcodeString(opcode)
                     << ": expected Constituent type to be equal to the column "
                     << "type Result Type array";
            }
          }

          break;
        }
        case SpvOpTypeStruct: {
          const Instruction* const struct_inst = _.FindDef(result_type);
          assert(struct_inst);
          assert(struct_inst->opcode() == SpvOpTypeStruct);

          if (struct_inst->operands().size() + 1 != num_operands) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << spvOpcodeString(opcode)
                   << ": expected total number of Constituents to be equal "
                   << "to the number of members of Result Type struct";
          }

          for (uint32_t operand_index = 2; operand_index < num_operands;
               ++operand_index) {
            const uint32_t operand_type =
                _.GetOperandTypeId(inst, operand_index);
            const uint32_t member_type = struct_inst->word(operand_index);
            if (operand_type != member_type) {
              return _.diag(SPV_ERROR_INVALID_DATA)
                     << spvOpcodeString(opcode)
                     << ": expected Constituent type to be equal to the "
                     << "corresponding member type of Result Type struct";
            }
          }

          break;
        }
        default: {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << spvOpcodeString(opcode)
                 << ": expected Result Type to be a composite type";
        }
      }

      break;
    }

    case SpvOpCompositeExtract:
    case SpvOpCompositeInsert: {
      // Handled in validate_id.cpp.
      // TODO(atgoo@github.com) Consider moving it here.
      break;
    }

#if 0
    // TODO(atgoo@github.com) Reenable this after this check passes Vulkan CTS.
    // A change to Vulkan CTS has been sent for review.
    case SpvOpCopyObject: {
      if (!spvOpcodeGeneratesType(_.GetIdOpcode(result_type))) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Result Type to be a type";
      }

      const uint32_t operand_type = _.GetOperandTypeId(inst, 2);
      if (operand_type != result_type) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Result Type and Operand type to be the same";
      }

      break;
    }
#endif

    case SpvOpTranspose: {
      uint32_t result_num_rows = 0;
      uint32_t result_num_cols = 0;
      uint32_t result_col_type = 0;
      uint32_t result_component_type = 0;
      if (!_.GetMatrixTypeInfo(result_type, &result_num_rows, &result_num_cols,
                               &result_col_type, &result_component_type)) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Result Type to be a matrix type";
      }

      const uint32_t matrix_type = _.GetOperandTypeId(inst, 2);
      uint32_t matrix_num_rows = 0;
      uint32_t matrix_num_cols = 0;
      uint32_t matrix_col_type = 0;
      uint32_t matrix_component_type = 0;
      if (!_.GetMatrixTypeInfo(matrix_type, &matrix_num_rows, &matrix_num_cols,
                               &matrix_col_type, &matrix_component_type)) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected Matrix to be of type OpTypeMatrix";
      }

      if (result_component_type != matrix_component_type) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected component types of Matrix and Result Type to be "
               << "identical";
      }

      if (result_num_rows != matrix_num_cols ||
          result_num_cols != matrix_num_rows) {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << spvOpcodeString(opcode)
               << ": expected number of columns and the column size of Matrix "
               << "to be the reverse of those of Result Type";
      }

      break;
    }

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace libspirv
