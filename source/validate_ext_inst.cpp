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

// Validates correctness of ExtInst SPIR-V instructions.

#include "validate.h"

#include <sstream>

#include "latest_version_glsl_std_450_header.h"
#include "latest_version_opencl_std_header.h"

#include "diagnostic.h"
#include "opcode.h"
#include "val/instruction.h"
#include "val/validation_state.h"

namespace libspirv {

// Validates correctness of ExtInst instructions.
spv_result_t ExtInstPass(ValidationState_t& _,
                         const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  const uint32_t result_type = inst->type_id;
  const uint32_t num_operands = inst->num_operands;

  if (opcode != SpvOpExtInst) return SPV_SUCCESS;

  const uint32_t ext_inst_set = inst->words[3];
  const uint32_t ext_inst_index = inst->words[4];
  const spv_ext_inst_type_t ext_inst_type =
      spv_ext_inst_type_t(inst->ext_inst_type);

  auto ext_inst_name = [&_, ext_inst_set, ext_inst_type, ext_inst_index]() {
    spv_ext_inst_desc desc = nullptr;
    if (_.grammar().lookupExtInst(ext_inst_type, ext_inst_index, &desc) !=
            SPV_SUCCESS ||
        !desc) {
      return std::string("Unknown ExtInst");
    }

    auto* import_inst = _.FindDef(ext_inst_set);
    assert(import_inst);

    std::ostringstream ss;
    ss << reinterpret_cast<const char*>(import_inst->words().data() + 2);
    ss << " ";
    ss << desc->name;

    return ss.str();
  };

  if (ext_inst_type == SPV_EXT_INST_TYPE_GLSL_STD_450) {
    const GLSLstd450 ext_inst_key = GLSLstd450(ext_inst_index);
    switch (ext_inst_key) {
      case GLSLstd450Round:
      case GLSLstd450RoundEven:
      case GLSLstd450FAbs:
      case GLSLstd450Trunc:
      case GLSLstd450FSign:
      case GLSLstd450Floor:
      case GLSLstd450Ceil:
      case GLSLstd450Fract:
      case GLSLstd450Sqrt:
      case GLSLstd450InverseSqrt:
      case GLSLstd450FMin:
      case GLSLstd450FMax:
      case GLSLstd450FClamp:
      case GLSLstd450FMix:
      case GLSLstd450Step:
      case GLSLstd450SmoothStep:
      case GLSLstd450Fma:
      case GLSLstd450Normalize:
      case GLSLstd450FaceForward:
      case GLSLstd450Reflect:
      case GLSLstd450NMin:
      case GLSLstd450NMax:
      case GLSLstd450NClamp: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case GLSLstd450SAbs:
      case GLSLstd450SSign:
      case GLSLstd450UMin:
      case GLSLstd450SMin:
      case GLSLstd450UMax:
      case GLSLstd450SMax:
      case GLSLstd450UClamp:
      case GLSLstd450SClamp:
      case GLSLstd450FindILsb:
      case GLSLstd450FindUMsb:
      case GLSLstd450FindSMsb: {
        if (!_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int scalar or vector type";
        }

        const uint32_t result_type_bit_width = _.GetBitWidth(result_type);
        const uint32_t result_type_dimension = _.GetDimension(result_type);

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (!_.IsIntScalarOrVectorType(operand_type)) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << ext_inst_name() << ": "
                   << "expected all operands to be int scalars or vectors";
          }

          if (result_type_dimension != _.GetDimension(operand_type)) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << ext_inst_name() << ": "
                   << "expected all operands to have the same dimension as "
                   << "Result Type";
          }

          if (result_type_bit_width != _.GetBitWidth(operand_type)) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << ext_inst_name() << ": "
                   << "expected all operands to have the same bit width as "
                   << "Result Type";
          }

          if (ext_inst_key == GLSLstd450FindUMsb ||
              ext_inst_key == GLSLstd450FindSMsb) {
            if (result_type_bit_width != 32) {
              return _.diag(SPV_ERROR_INVALID_DATA)
                     << ext_inst_name() << ": "
                     << "this instruction is currently limited to 32-bit width "
                     << "components";
            }
          }
        }
        break;
      }

      case GLSLstd450Radians:
      case GLSLstd450Degrees:
      case GLSLstd450Sin:
      case GLSLstd450Cos:
      case GLSLstd450Tan:
      case GLSLstd450Asin:
      case GLSLstd450Acos:
      case GLSLstd450Atan:
      case GLSLstd450Sinh:
      case GLSLstd450Cosh:
      case GLSLstd450Tanh:
      case GLSLstd450Asinh:
      case GLSLstd450Acosh:
      case GLSLstd450Atanh:
      case GLSLstd450Exp:
      case GLSLstd450Exp2:
      case GLSLstd450Log:
      case GLSLstd450Log2:
      case GLSLstd450Atan2:
      case GLSLstd450Pow: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 16 or 32-bit scalar or "
                    "vector float type";
        }

        const uint32_t result_type_bit_width = _.GetBitWidth(result_type);
        if (result_type_bit_width != 16 && result_type_bit_width != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 16 or 32-bit scalar or "
                    "vector float type";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case GLSLstd450Determinant: {
        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;
        uint32_t col_type = 0;
        uint32_t component_type = 0;
        if (!_.GetMatrixTypeInfo(x_type, &num_rows, &num_cols, &col_type,
                                 &component_type) ||
            num_rows != num_cols) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X to be a square matrix";
        }

        if (result_type != component_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X component type to be equal to "
                 << "Result Type";
        }
        break;
      }

      case GLSLstd450MatrixInverse: {
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;
        uint32_t col_type = 0;
        uint32_t component_type = 0;
        if (!_.GetMatrixTypeInfo(result_type, &num_rows, &num_cols, &col_type,
                                 &component_type) ||
            num_rows != num_cols) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a square matrix";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (result_type != x_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }
        break;
      }

      case GLSLstd450Modf: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or vector float type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t i_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        uint32_t i_storage_class = 0;
        uint32_t i_data_type = 0;
        if (!_.GetPointerTypeInfo(i_type, &i_data_type, &i_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand I to be a pointer";
        }

        if (i_data_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand I data type to be equal to Result Type";
        }

        break;
      }

      case GLSLstd450ModfStruct: {
        std::vector<uint32_t> result_types;
        if (!_.GetStructMemberTypes(result_type, &result_types) ||
            result_types.size() != 2 ||
            !_.IsFloatScalarOrVectorType(result_types[0]) ||
            result_types[1] != result_types[0]) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a struct with two identical "
                 << "scalar or vector float type members";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (x_type != result_types[0]) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to members of "
                 << "Result Type struct";
        }
        break;
      }

      case GLSLstd450Frexp: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or vector float type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t exp_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        uint32_t exp_storage_class = 0;
        uint32_t exp_data_type = 0;
        if (!_.GetPointerTypeInfo(exp_type, &exp_data_type,
                                  &exp_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand Exp to be a pointer";
        }

        if (!_.IsIntScalarOrVectorType(exp_data_type) ||
            _.GetBitWidth(exp_data_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand Exp data type to be a 32-bit int scalar "
                 << "or vector type";
        }

        if (_.GetDimension(result_type) != _.GetDimension(exp_data_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand Exp data type to have the same component "
                 << "number as Result Type";
        }

        break;
      }

      case GLSLstd450Ldexp: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or vector float type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t exp_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        if (!_.IsIntScalarOrVectorType(exp_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand Exp to be a 32-bit int scalar "
                 << "or vector type";
        }

        if (_.GetDimension(result_type) != _.GetDimension(exp_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand Exp to have the same component "
                 << "number as Result Type";
        }

        break;
      }

      case GLSLstd450FrexpStruct: {
        std::vector<uint32_t> result_types;
        if (!_.GetStructMemberTypes(result_type, &result_types) ||
            result_types.size() != 2 ||
            !_.IsFloatScalarOrVectorType(result_types[0]) ||
            !_.IsIntScalarOrVectorType(result_types[1]) ||
            _.GetBitWidth(result_types[1]) != 32 ||
            _.GetDimension(result_types[0]) !=
                _.GetDimension(result_types[1])) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a struct with two members, "
                 << "first member a float scalar or vector, second member "
                 << "a 32-bit int scalar or vector with the same number of "
                 << "components as the first member";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (x_type != result_types[0]) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to the first member "
                 << "of Result Type struct";
        }
        break;
      }

      case GLSLstd450PackSnorm4x8:
      case GLSLstd450PackUnorm4x8: {
        if (!_.IsIntScalarType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be 32-bit int scalar type";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatVectorType(v_type) || _.GetDimension(v_type) != 4 ||
            _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 32-bit float vector of size 4";
        }
        break;
      }

      case GLSLstd450PackSnorm2x16:
      case GLSLstd450PackUnorm2x16:
      case GLSLstd450PackHalf2x16: {
        if (!_.IsIntScalarType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be 32-bit int scalar type";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatVectorType(v_type) || _.GetDimension(v_type) != 2 ||
            _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 32-bit float vector of size 2";
        }
        break;
      }

      case GLSLstd450PackDouble2x32: {
        if (!_.IsFloatScalarType(result_type) ||
            _.GetBitWidth(result_type) != 64) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be 64-bit float scalar type";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntVectorType(v_type) || _.GetDimension(v_type) != 2 ||
            _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 32-bit int vector of size 2";
        }
        break;
      }

      case GLSLstd450UnpackSnorm4x8:
      case GLSLstd450UnpackUnorm4x8: {
        if (!_.IsFloatVectorType(result_type) ||
            _.GetDimension(result_type) != 4 ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit float vector of size "
                    "4";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntScalarType(v_type) || _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a 32-bit int scalar";
        }
        break;
      }

      case GLSLstd450UnpackSnorm2x16:
      case GLSLstd450UnpackUnorm2x16:
      case GLSLstd450UnpackHalf2x16: {
        if (!_.IsFloatVectorType(result_type) ||
            _.GetDimension(result_type) != 2 ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit float vector of size "
                    "2";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntScalarType(v_type) || _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a 32-bit int scalar";
        }
        break;
      }

      case GLSLstd450UnpackDouble2x32: {
        if (!_.IsIntVectorType(result_type) ||
            _.GetDimension(result_type) != 2 ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit int vector of size "
                    "2";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarType(v_type) || _.GetBitWidth(v_type) != 64) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 64-bit float scalar";
        }
        break;
      }

      case GLSLstd450Length: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(x_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X to be of float scalar or vector type";
        }

        if (result_type != _.GetComponentType(x_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X component type to be equal to Result "
                    "Type";
        }
        break;
      }

      case GLSLstd450Distance: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t p0_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(p0_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand P0 to be of float scalar or vector type";
        }

        if (result_type != _.GetComponentType(p0_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand P0 component type to be equal to "
                 << "Result Type";
        }

        const uint32_t p1_type = _.GetOperandTypeId(inst, 5);
        if (!_.IsFloatScalarOrVectorType(p1_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand P1 to be of float scalar or vector type";
        }

        if (result_type != _.GetComponentType(p1_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand P1 component type to be equal to "
                 << "Result Type";
        }

        if (_.GetDimension(p0_type) != _.GetDimension(p1_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operands P0 and P1 to have the same number of "
                 << "components";
        }
        break;
      }

      case GLSLstd450Cross: {
        if (!_.IsFloatVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float vector type";
        }

        if (_.GetDimension(result_type) != 3) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 3 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t y_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        if (y_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand Y type to be equal to Result Type";
        }
        break;
      }

      case GLSLstd450Refract: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t i_type = _.GetOperandTypeId(inst, 4);
        const uint32_t n_type = _.GetOperandTypeId(inst, 5);
        const uint32_t eta_type = _.GetOperandTypeId(inst, 6);

        if (result_type != i_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand I to be of type equal to Result Type";
        }

        if (result_type != n_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand N to be of type equal to Result Type";
        }

        const uint32_t eta_type_bit_width = _.GetBitWidth(eta_type);
        if (!_.IsFloatScalarType(eta_type) ||
            (eta_type_bit_width != 16 && eta_type_bit_width != 32)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected operand Eta to be a 16 or 32-bit float scalar";
        }
        break;
      }

      case GLSLstd450InterpolateAtCentroid:
      case GLSLstd450InterpolateAtSample:
      case GLSLstd450InterpolateAtOffset: {
        if (!_.HasCapability(SpvCapabilityInterpolationFunction)) {
          return _.diag(SPV_ERROR_INVALID_CAPABILITY)
                 << ext_inst_name()
                 << " requires capability InterpolationFunction";
        }

        if (!_.IsFloatScalarOrVectorType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit float scalar "
                 << "or vector type";
        }

        const uint32_t interpolant_type = _.GetOperandTypeId(inst, 4);
        uint32_t interpolant_storage_class = 0;
        uint32_t interpolant_data_type = 0;
        if (!_.GetPointerTypeInfo(interpolant_type, &interpolant_data_type,
                                  &interpolant_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Interpolant to be a pointer";
        }

        if (result_type != interpolant_data_type) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Interpolant data type to be equal to Result Type";
        }

        if (interpolant_storage_class != SpvStorageClassInput) {
          return _.diag(SPV_ERROR_INVALID_DATA)
                 << ext_inst_name() << ": "
                 << "expected Interpolant storage class to be Input";
        }

        if (ext_inst_key == GLSLstd450InterpolateAtSample) {
          const uint32_t sample_type = _.GetOperandTypeId(inst, 5);
          if (!_.IsIntScalarType(sample_type) ||
              _.GetBitWidth(sample_type) != 32) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << ext_inst_name() << ": "
                   << "expected Sample to be 32-bit integer";
          }
        }

        if (ext_inst_key == GLSLstd450InterpolateAtOffset) {
          const uint32_t offset_type = _.GetOperandTypeId(inst, 5);
          if (!_.IsFloatVectorType(offset_type) ||
              _.GetDimension(offset_type) != 2 ||
              _.GetBitWidth(offset_type) != 32) {
            return _.diag(SPV_ERROR_INVALID_DATA)
                   << ext_inst_name() << ": "
                   << "expected Offset to be a vector of 2 32-bit floats";
          }
        }

        _.current_function().RegisterExecutionModelLimitation(
            SpvExecutionModelFragment,
            ext_inst_name() +
                std::string(" requires Fragment execution model"));
        break;
      }

      case GLSLstd450IMix: {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << "Extended instruction GLSLstd450IMix is not supported";
      }

      case GLSLstd450Bad: {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << "Encountered extended instruction GLSLstd450Bad";
      }

      case GLSLstd450Count: {
        assert(0);
        break;
      }
    }
  } else if (ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_STD) {
    // TODO(atgoo@github.com) Add validation rules for OpenCL extended
    // instructions.
  }

  return SPV_SUCCESS;
}

}  // namespace libspirv
