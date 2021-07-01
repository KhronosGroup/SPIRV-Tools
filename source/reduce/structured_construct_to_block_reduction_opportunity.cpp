// Copyright (c) 2021 Alastair F. Donaldson
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

#include "source/reduce/structured_construct_to_block_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

bool StructuredConstructToBlockReductionOpportunity::PreconditionHolds() {
  return context_->get_def_use_mgr()->GetDef(construct_header_) != nullptr;
}

void StructuredConstructToBlockReductionOpportunity::Apply() {
  auto header_block = context_->cfg()->block(construct_header_);
  auto merge_block = context_->cfg()->block(header_block->MergeBlockId());
  std::unordered_set<opt::BasicBlock*> to_erase;
  auto* enclosing_function = header_block->GetParent();
  auto* dominators = context_->GetDominatorAnalysis(enclosing_function);
  auto* postdominators = context_->GetPostDominatorAnalysis(enclosing_function);

  for (auto block_it = enclosing_function->begin();
       block_it != enclosing_function->end();) {
    if (header_block != &*block_it && merge_block != &*block_it &&
        dominators->Dominates(header_block, &*block_it) &&
        postdominators->Dominates(merge_block, &*block_it)) {
      block_it = block_it.Erase();
    } else {
      ++block_it;
    }
  }
  context_->KillInst(header_block->GetMergeInst());
  header_block->terminator()->SetOpcode(SpvOpBranch);
  opt::Instruction::OperandList operands;
  operands.push_back(opt::Operand(SPV_OPERAND_TYPE_ID, {merge_block->id()}));
  header_block->terminator()->SetInOperands(std::move(operands));
  context_->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

}  // namespace reduce
}  // namespace spvtools
