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

#include "local_redundancy_elimination.h"

#include "value_number_table.h"

namespace spvtools {
namespace opt {

Pass::Status LocalRedundancyEliminationPass::Process(ir::IRContext* c) {
  InitializeProcessing(c);

  bool modified = false;

  for (auto& func : *get_module()) {
    for (auto& bb : func) {
      // Resetting the value number table for every basic block because we just
      // want the opportunities within a basic block. This will help keep
      // register pressure down.
      ValueNumberTable vnTable(context());

      // Keeps track of all ids that contain a given value number. We keep
      // track of multiple values because they could have the same value, but
      // different decorations.
      std::vector<std::vector<uint32_t>> value_to_ids;
      if (EliminateRedundanciesInBB(&bb, &vnTable, &value_to_ids))
        modified = true;
    }
  }

  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool LocalRedundancyEliminationPass::EliminateRedundanciesInBB(
    ir::BasicBlock* block, ValueNumberTable* vnTable,
    std::vector<std::vector<uint32_t>>* value_to_ids) {
  bool modified = false;

  auto func = [this, vnTable, &modified, value_to_ids](ir::Instruction* inst) {
    if (inst->result_id() == 0) {
      return;
    }

    uint32_t value = vnTable->GetValueNumber(inst);
    if (value >= value_to_ids->size()) {
      value_to_ids->resize(value + 1);
    }

    // Now that we have the value number of the instruction, we use
    // |value_to_ids| to get other ids that contain the same value.  If we can
    // find an id in that set which has the same decorations, we can replace all
    // uses of the result of |inst| by that id.
    std::vector<uint32_t>& candidate_set = (*value_to_ids)[value];
    bool found_replacement = false;
    for (uint32_t candidate_id : candidate_set) {
      if (get_decoration_mgr()->HaveTheSameDecorations(inst->result_id(),
                                                       candidate_id)) {
        context()->KillNamesAndDecorates(inst);
        context()->ReplaceAllUsesWith(inst->result_id(), candidate_id);
        context()->KillInst(inst);
        modified = true;
        found_replacement = true;
        break;
      }
    }

    // If we did not find a replacement, then add it as a candidate for later
    // instructions.
    if (!found_replacement) {
      candidate_set.push_back(inst->result_id());
    }
  };
  block->ForEachInst(func);
  return modified;
}
}  // namespace opt
}  // namespace spvtools
