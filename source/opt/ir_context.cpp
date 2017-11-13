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

#include "ir_context.h"
#include "log.h"

namespace spvtools {
namespace ir {

void IRContext::BuildInvalidAnalyses(IRContext::Analysis set) {
  if (set & kAnalysisDefUse) {
    BuildDefUseManager();
  }
  if (set & kAnalysisInstrToBlockMapping) {
    BuildInstrToBlockMapping();
  }
}

void IRContext::InvalidateAnalysesExceptFor(
    IRContext::Analysis preserved_analyses) {
  uint32_t analyses_to_invalidate = valid_analyses_ & (~preserved_analyses);
  InvalidateAnalyses(static_cast<IRContext::Analysis>(analyses_to_invalidate));
}

void IRContext::InvalidateAnalyses(IRContext::Analysis analyses_to_invalidate) {
  if (analyses_to_invalidate & kAnalysisDefUse) {
    def_use_mgr_.reset(nullptr);
  }
  if (analyses_to_invalidate & kAnalysisInstrToBlockMapping) {
    instr_to_block_.clear();
  }
  valid_analyses_ = Analysis(valid_analyses_ & ~analyses_to_invalidate);
}

void IRContext::KillInst(ir::Instruction* inst) {
  if (!inst) {
    return;
  }

  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->ClearInst(inst);
  }
  if (AreAnalysesValid(kAnalysisInstrToBlockMapping)) {
    instr_to_block_.erase(inst);
  }

  inst->ToNop();
}

bool IRContext::KillDef(uint32_t id) {
  ir::Instruction* def = get_def_use_mgr()->GetDef(id);
  if (def != nullptr) {
    KillInst(def);
    return true;
  }
  return false;
}

bool IRContext::ReplaceAllUsesWith(uint32_t before, uint32_t after) {
  if (before == after) return false;
  opt::analysis::UseList* uses = get_def_use_mgr()->GetUses(before);
  if (uses == nullptr) return false;

  std::vector<opt::analysis::Use> uses_to_update;
  for (auto it = uses->cbegin(); it != uses->cend(); ++it) {
    uses_to_update.push_back(*it);
  }

  for (opt::analysis::Use& use : uses_to_update) {
    ForgetUses(use.inst);
    const uint32_t type_result_id_count =
        (use.inst->result_id() != 0) + (use.inst->type_id() != 0);

    if (use.operand_index < type_result_id_count) {
      // Update the type_id. Note that result id is immutable so it should
      // never be updated.
      if (use.inst->type_id() != 0 && use.operand_index == 0) {
        use.inst->SetResultType(after);
      } else if (use.inst->type_id() == 0) {
        SPIRV_ASSERT(consumer_, false,
                     "Result type id considered as use while the instruction "
                     "doesn't have a result type id.");
        (void)consumer_;  // Makes the compiler happy for release build.
      } else {
        SPIRV_ASSERT(consumer_, false,
                     "Trying setting the immutable result id.");
      }
    } else {
      // Update an in-operand.
      uint32_t in_operand_pos = use.operand_index - type_result_id_count;
      // Make the modification in the instruction.
      use.inst->SetInOperand(in_operand_pos, {after});
    }
    AnalyzeUses(use.inst);
  }
  return true;
}

bool IRContext::IsConsistent() {
#ifndef SPIRV_CHECK_CONTEXT
  return true;
#endif

  if (AreAnalysesValid(kAnalysisDefUse)) {
    opt::analysis::DefUseManager new_def_use(module());
    if (*get_def_use_mgr() != new_def_use) {
      return false;
    }
  }
  return true;
}

void spvtools::ir::IRContext::ForgetUses(Instruction* inst) {
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->EraseUseRecordsOfOperandIds(inst);
  }
}

void IRContext::AnalyzeUses(Instruction* inst) {
  if (AreAnalysesValid(kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstUse(inst);
  }
}

}  // namespace ir
}  // namespace spvtools
