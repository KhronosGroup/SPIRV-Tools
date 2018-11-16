// Copyright (c) 2018 Google LLC
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

#include "operand_to_const_reduction_pass.h"
#include "change_operand_reduction_opportunity.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace reduce {

using namespace opt;

std::vector<std::unique_ptr<ReductionOpportunity>>
OperandToConstReductionPass::GetAvailableOpportunities(
    opt::IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;
  assert(result.empty());

  for (const auto& constant : context->GetConstants()) {
    for (auto& function : *context->module()) {
      for (auto& block : function) {
        for (auto& inst : block) {
          for (uint32_t i = 0; i < inst.NumOperands(); i++) {
            const auto& operand = inst.GetOperand(i);
            if (operand.type == SPV_OPERAND_TYPE_ID) {
              const auto id = operand.words[0];
              auto def = context->get_def_use_mgr()->GetDef(id);
              if (context->get_constant_mgr()->GetConstantFromInst(def)) {
                // The argument is already a constant.
                continue;
              }
              auto type_id = def->type_id();
              if (type_id) {
                if (constant->type_id() == type_id) {
                  result.push_back(
                      MakeUnique<ChangeOperandReductionOpportunity>(
                          &inst, i, id, constant->result_id()));
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

std::string OperandToConstReductionPass::GetName() const {
  return "OperandToConstReductionPass";
}

}  // namespace reduce
}  // namespace spvtools
