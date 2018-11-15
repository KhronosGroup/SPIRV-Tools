// Copyright (c) 2018 Google Inc.
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

#include "remove_unreferenced_instruction_reduction_pass.h"
#include "remove_instruction_reduction_opportunity.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace reduce {

using namespace opt;

std::vector<std::unique_ptr<ReductionOpportunity>>
RemoveUnreferencedInstructionReductionPass::GetAvailableOpportunities(
    opt::IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  for (auto& function : *context->module()) {
    for (auto& block : function) {
      for (auto& inst : block) {
        if (context->get_def_use_mgr()->NumUses(&inst) == 0) {
          switch (inst.opcode()) {
            case SpvOpBranch:
            case SpvOpBranchConditional:
            case SpvOpLoopMerge:
            case SpvOpSelectionMerge:
            case SpvOpReturn:
            case SpvOpSwitch:
              // TODO: this should ultimately capture all opcodes that relate to
              // control flow; we don't want to mess with these here.
              break;
            default:
              // Given that we're in a block, we should only get here if the
              // instruction is not directly related to control flow; i.e., it's
              // some straightforward instruction with an unused result, like an
              // arithmetic operation or function call.
              result.push_back(
                  MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
          }
        }
      }
    }
  }
  return result;
}

std::string RemoveUnreferencedInstructionReductionPass::GetName() const {
  return "RemoveUnreferencedInstructionReductionPass";
}

}  // namespace reduce
}  // namespace spvtools
