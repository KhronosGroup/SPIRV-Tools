// Copyright (c) 2017 NVIDIA Corporation
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

// Validates branch instructions

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

// Returns the type id of instruction operand at |operand_index|.
// The operand is expected to be an id.
inline uint32_t GetOperandTypeId(ValidationState_t& _,
                               const spv_parsed_instruction_t* inst,
                               size_t operand_index) {
  return _.GetTypeId(GetOperandWord(inst, operand_index));
}

} // namespace

spv_result_t BranchPass(ValidationState_t& _,
                        const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);

  switch(opcode) {
    case SpvOpBranchConditional:
      if (inst->num_operands < 3) {
        return _.diag(SPV_ERROR_INVALID_DATA)
            << "Expected a minimum of 3 operands in " << spvOpcodeString(opcode);
      }

      if (!_.IsBoolScalarType(GetOperandTypeId(_, inst, 0))) {
        return _.diag(SPV_ERROR_INVALID_DATA)
            << "Condition operand for " << spvOpcodeString(opcode) << " must be of bool scalar type.";
        }
    }

    return SPV_SUCCESS;
}

} // namespace libspirv
