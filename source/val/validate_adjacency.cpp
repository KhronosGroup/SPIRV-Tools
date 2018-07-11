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

spv_result_t ValidateAdjacency(ValidationState_t& _) {
  const auto& instructions = _.ordered_instructions();
  for (auto i = instructions.cbegin(); i != instructions.cend(); ++i) {
    switch (i->opcode()) {
      case SpvOpPhi:
        if (i != instructions.cbegin()) {
          switch (prev(i)->opcode()) {
            case SpvOpLabel:
            case SpvOpPhi:
            case SpvOpLine:
              break;
            default:
              return _.diag(SPV_ERROR_INVALID_DATA)
                     << "OpPhi must appear before all non-OpPhi instructions "
                     << "(except for OpLine, which can be mixed with OpPhi).";
          }
        }
        break;
      case SpvOpLoopMerge:
        if (next(i) != instructions.cend()) {
          switch (next(i)->opcode()) {
            case SpvOpBranch:
            case SpvOpBranchConditional:
              break;
            default:
              return _.diag(SPV_ERROR_INVALID_DATA)
                     << "OpLoopMerge must immediately precede either an "
                     << "OpBranch or OpBranchConditional instruction. "
                     << "OpLoopMerge must be the second-to-last instruction in "
                     << "its block.";
          }
        }
        break;
      case SpvOpSelectionMerge:
        if (next(i) != instructions.cend()) {
          switch (next(i)->opcode()) {
            case SpvOpBranchConditional:
            case SpvOpSwitch:
              break;
            default:
              return _.diag(SPV_ERROR_INVALID_DATA)
                     << "OpSelectionMerge must immediately precede either an "
                     << "OpBranchConditional or OpSwitch instruction. "
                     << "OpSelectionMerge must be the second-to-last "
                     << "instruction in its block.";
          }
        }
      default:
        break;
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
