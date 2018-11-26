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

#include "source/opt/ir_context.h"

#include "cut_loop_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

bool CutLoopReductionOpportunity::PreconditionHolds() { return true; }

void CutLoopReductionOpportunity::Apply() {
  loop_->GetContext()->get_def_use_mgr()->ForEachUse(
      loop_->GetContinueBlock()->GetLabel().get(),
      [this](Instruction* inst, uint32_t index) {
        inst->SetOperand(index,
                         {loop_->GetMergeBlock()->GetLabel()->result_id()});
      });
  auto loop_merge_inst = loop_->GetHeaderBlock()->GetLoopMergeInst();
  loop_merge_inst->context()->KillInst(loop_merge_inst);
}

}  // namespace reduce
}  // namespace spvtools
