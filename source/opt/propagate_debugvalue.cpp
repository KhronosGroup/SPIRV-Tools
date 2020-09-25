// Copyright (c) 2020 Google LLC
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

#include "source/opt/propagate_debugvalue.h"

static const uint32_t kDebugValueOperandLocalVariableIndex = 4;

namespace spvtools {
namespace opt {

Pass::Status PropagateDebugvalue::Process() {
  Status status = Status::SuccessWithoutChange;
  for (auto& fn : *get_module()) {
    status = CombineStatus(status, PropagateDebugvalueForFunction(&fn));
    if (status == Status::Failure) {
      break;
    }
  }
  return status;
}

Pass::Status PropagateDebugvalue::PropagateDebugvalueForFunction(Function* fp) {
  bool modified = false;
  std::unordered_map<BasicBlock*,
                     std::unordered_map<uint32_t, const Instruction*>>
      bb_to_available_dbgvalues;
  bool succeeded = cfg()->WhileEachBlockInReversePostOrder(
      fp->entry().get(),
      [this, &bb_to_available_dbgvalues, &modified](BasicBlock* bb) {
        auto dominators = context()->GetDominatorAnalysis(bb->GetParent());
        if (dominators == nullptr) return false;

        // If first instructions in the basic block are DebugValues, we
        // have to avoid adding DebugValues from immediate dominator for
        // the same local variables. |live_dbgvalues| keeps the information.
        std::unordered_map<uint32_t, const Instruction*> live_dbgvalues;
        auto ii = bb->begin();
        while (ii != bb->end() &&
               ii->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugValue) {
          live_dbgvalues[ii->GetSingleWordOperand(
              kDebugValueOperandLocalVariableIndex)] = &*ii;
          ++ii;
        }

        // Add DebugValues from the immediate dominator to the basic block.
        auto immediate_dominator_dbgvalues_itr =
            bb_to_available_dbgvalues.find(dominators->ImmediateDominator(bb));
        if (immediate_dominator_dbgvalues_itr !=
            bb_to_available_dbgvalues.end()) {
          for (auto& live_dbgvalue_in_top :
               immediate_dominator_dbgvalues_itr->second) {
            uint32_t var_id = live_dbgvalue_in_top.second->GetSingleWordOperand(
                kDebugValueOperandLocalVariableIndex);
            if (live_dbgvalues.find(var_id) != live_dbgvalues.end()) continue;

            std::unique_ptr<Instruction> inst(
                live_dbgvalue_in_top.second->Clone(context()));
            uint32_t newId = context()->TakeNextId();
            if (newId == 0) {
              return false;
            }
            inst->SetResultId(newId);
            auto* new_dbgvalue = ii->InsertBefore(std::move(inst));
            live_dbgvalues[var_id] = new_dbgvalue;
            modified = true;
          }
        }

        // Set remaining DebugValues as the available DebugValues for
        // immediate dominatees.
        while (ii != bb->end()) {
          if (ii->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugValue) {
            live_dbgvalues[ii->GetSingleWordOperand(
                kDebugValueOperandLocalVariableIndex)] = &*ii;
          }
          ++ii;
        }
        bb_to_available_dbgvalues[bb] = live_dbgvalues;
        return true;
      });
  if (!succeeded) {
    return Status::Failure;
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

}  // namespace opt
}  // namespace spvtools
