// Copyright (c) 2018 Google LLC.
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

#include "interface_repair_pass.h"
#include "source/spirv_constant.h"
namespace spvtools {
namespace opt {
InterfaceRepairPass::Status InterfaceRepairPass::Process() {
  std::unordered_map<uint32_t, Function*> id2function_;
  for (auto& fn : *get_module()) id2function_[fn.result_id()] = &fn;
  bool modified = false;
  for (auto& entry : get_module()->entry_points()) {
    std::set<uint32_t> used_variables;
    std::function<void(Function * func)> traverser = [&](Function* func) {
      for (auto bi = func->begin(); bi != func->end(); ++bi) {
        for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
          if (ii->opcode() == SpvOpFunctionCall) {
            const auto called_id = ii->GetOperand(2).words[0];
            traverser(id2function_.at(called_id));
          } else {
            for (uint32_t i = 0; i < ii->NumOperands(); ++i) {
              auto& op = ii->GetOperand(i);
              if (op.type == SPV_OPERAND_TYPE_ID) {
                auto id = op.words[0];
                if (used_variables.count(id)) continue;
                auto* var = get_def_use_mgr()->GetDef(id);
                if (!var || var->opcode() != SpvOpVariable) continue;
                auto storage_class = var->GetSingleWordInOperand(0);
                if (storage_class != SpvStorageClassFunction &&
                    (get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 4) ||
                     storage_class == SpvStorageClassInput ||
                     storage_class == SpvStorageClassOutput)) {
                  used_variables.insert(id);
                }
              }
            }
          }
        }
      }
    };
    traverser(id2function_.at(entry.GetSingleWordInOperand(1)));
    {
      std::set<uint32_t> old_variables;
      for (int i = entry.NumInOperands() - 1; i >= 3; --i) {
        auto variable = entry.GetInOperand(i).words[0];
        if (!used_variables.count(variable))  // There is unused ids.
          goto repair;
        if (old_variables.count(variable))  // There is duplicate ids.
          goto repair;
        old_variables.insert(variable);
      }
      if (old_variables.size() != used_variables.size())  // There is missing.
        goto repair;
    }
    continue;
  repair:
    modified = true;
    for (int i = entry.NumInOperands() - 1; i >= 3; --i)
      entry.RemoveInOperand(i);
    for (auto id : used_variables) {
      entry.AddOperand(Operand(SPV_OPERAND_TYPE_ID, {id}));
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}
}  // namespace opt
}  // namespace spvtools