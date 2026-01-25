// Copyright (c) 2026 LunarG Inc.
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

#include <cstdint>

#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateGroupAnyAll(ValidationState_t& _,
                                 const Instruction* inst) {
  if (!_.IsBoolScalarType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a boolean scalar type";
  }

  if (!_.IsBoolScalarType(_.GetOperandTypeId(inst, 3))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Predicate must be a boolean scalar type";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateGroupBroadcast(ValidationState_t& _,
                                    const Instruction* inst) {
  const uint32_t type_id = inst->type_id();
  if (!_.IsFloatScalarOrVectorType(type_id) &&
      !_.IsIntScalarOrVectorType(type_id) &&
      !_.IsBoolScalarOrVectorType(type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a scalar or vector of integer, floating-point, "
              "or boolean type";
  }

  const uint32_t value_type_id = _.GetOperandTypeId(inst, 3);
  if (value_type_id != type_id) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "The type of Value must match the Result type";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateGroupFloat(ValidationState_t& _, const Instruction* inst) {
  const uint32_t type_id = inst->type_id();
  if (!_.IsFloatScalarOrVectorType(type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a scalar or vector of float type";
  }

  const uint32_t x_type_id = _.GetOperandTypeId(inst, 4);
  if (x_type_id != type_id) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "The type of X must match the Result type";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidateGroupInt(ValidationState_t& _, const Instruction* inst) {
  const uint32_t type_id = inst->type_id();
  if (!_.IsIntScalarOrVectorType(type_id)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Result must be a scalar or vector of integer type";
  }

  const uint32_t x_type_id = _.GetOperandTypeId(inst, 4);
  if (x_type_id != type_id) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "The type of X must match the Result type";
  }
  return SPV_SUCCESS;
}

}  // namespace

spv_result_t GroupPass(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();

  switch (opcode) {
    case spv::Op::OpGroupAny:
    case spv::Op::OpGroupAll:
      return ValidateGroupAnyAll(_, inst);
    case spv::Op::OpGroupBroadcast:
      return ValidateGroupBroadcast(_, inst);
    case spv::Op::OpGroupFAdd:
    case spv::Op::OpGroupFMax:
    case spv::Op::OpGroupFMin:
      return ValidateGroupFloat(_, inst);
    case spv::Op::OpGroupIAdd:
    case spv::Op::OpGroupUMin:
    case spv::Op::OpGroupSMin:
    case spv::Op::OpGroupUMax:
    case spv::Op::OpGroupSMax:
      return ValidateGroupInt(_, inst);
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
