// Copyright (c) 2021 Google LLC.
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

class EntryRepairContext {
  InterfaceRepairPass& parent_;
  Instruction& entry_;
  std::unordered_set<uint32_t> used_variables_;
  IRContext::ProcessFunction pfn_ = std::bind(
      &EntryRepairContext::processFunction, this, std::placeholders::_1);

  bool processFunction(Function* func) {
    for (const auto& basicBlock : *func)
      for (const auto& instruction : basicBlock)
        for (const auto& operand : instruction)
          if (operand.type == SPV_OPERAND_TYPE_ID) {
            auto id = operand.words[0];
            if (used_variables_.count(id)) continue;
            auto* var = parent_.get_def_use_mgr()->GetDef(id);
            if (!var || var->opcode() != SpvOpVariable) continue;
            auto storage_class = var->GetSingleWordInOperand(0);
            if (storage_class != SpvStorageClassFunction &&
                (parent_.get_module()->version() >=
                     SPV_SPIRV_VERSION_WORD(1, 4) ||
                 storage_class == SpvStorageClassInput ||
                 storage_class == SpvStorageClassOutput))
              used_variables_.insert(id);
          }
    return false;
  }

 public:
  EntryRepairContext(InterfaceRepairPass& parent, Instruction& entry)
      : parent_(parent), entry_(entry) {}

  void collectUsedVariables() {
    std::queue<uint32_t> roots;
    roots.push(entry_.result_id());
    parent_.context()->ProcessCallTreeFromRoots(pfn_, &roots);
  }

  bool shouldRepair() {
    std::unordered_set<uint32_t> old_variables;
    for (int i = entry_.NumInOperands() - 1; i >= 3; --i) {
      auto variable = entry_.GetInOperand(i).words[0];
      if (!used_variables_.count(variable)) return true;  // It is unused.
      if (old_variables.count(variable)) return true;     // It is duplicate.
      old_variables.insert(variable);
    }
    if (old_variables.size() != used_variables_.size())  // Missing IDs.
      return true;
    return false;
  }

  void repair() {
    for (int i = entry_.NumInOperands() - 1; i >= 3; --i)
      entry_.RemoveInOperand(i);
    for (auto id : used_variables_) {
      entry_.AddOperand(Operand(SPV_OPERAND_TYPE_ID, {id}));
    }
  }
};

InterfaceRepairPass::Status InterfaceRepairPass::Process() {
  bool modified = false;
  for (auto& entry : get_module()->entry_points()) {
    EntryRepairContext context(*this, entry);
    context.collectUsedVariables();
    if (context.shouldRepair()) {
      context.repair();
      modified = true;
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}
}  // namespace opt
}  // namespace spvtools