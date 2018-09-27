// Copyright (c) 2018 LunarG Inc.
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

// Validates correctness of the intra-block preconditions of SPIR-V
// instructions.

#include "source/val/validate.h"

#include <string>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

enum {
  IN_NEW_FUNCTION,
  IN_ENTRY_BLOCK,
  PHI_VALID,
  PHI_INVALID,
};

spv_result_t ValidateAdjacency(ValidationState_t& _) {
  const auto& instructions = _.ordered_instructions();
  int phi_check = PHI_INVALID;

  for (size_t i = 0; i < instructions.size(); ++i) {
    const auto& inst = instructions[i];
    switch (inst.opcode()) {
      case SpvOpFunction:
      case SpvOpFunctionParameter:
        phi_check = IN_NEW_FUNCTION;
        break;
      case SpvOpLabel:
        phi_check = phi_check == IN_NEW_FUNCTION ? IN_ENTRY_BLOCK : PHI_VALID;
        break;
      case SpvOpPhi:
        if (phi_check != PHI_VALID) {
          return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                 << "OpPhi must appear within a non-entry block before all "
                 << "non-OpPhi instructions "
                 << "(except for OpLine, which can be mixed with OpPhi).";
        }
        break;
      case SpvOpLine:
        break;
      case SpvOpLoopMerge:
        phi_check = PHI_INVALID;
        if (i != (instructions.size() - 1)) {
          switch (instructions[i + 1].opcode()) {
            case SpvOpBranch:
            case SpvOpBranchConditional:
              break;
            default:
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "OpLoopMerge must immediately precede either an "
                     << "OpBranch or OpBranchConditional instruction. "
                     << "OpLoopMerge must be the second-to-last instruction in "
                     << "its block.";
          }
        }
        break;
      case SpvOpSelectionMerge:
        phi_check = PHI_INVALID;
        if (i != (instructions.size() - 1)) {
          switch (instructions[i + 1].opcode()) {
            case SpvOpBranchConditional:
            case SpvOpSwitch:
              break;
            default:
              return _.diag(SPV_ERROR_INVALID_DATA, &inst)
                     << "OpSelectionMerge must immediately precede either an "
                     << "OpBranchConditional or OpSwitch instruction. "
                     << "OpSelectionMerge must be the second-to-last "
                     << "instruction in its block.";
          }
        }
        break;
      default:
        phi_check = PHI_INVALID;
        break;
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
