// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

FuzzerPass::FuzzerPass(opt::IRContext* ir_context, FactManager* fact_manager,
                       FuzzerContext* fuzzer_context,
                       protobufs::TransformationSequence* transformations)
    : ir_context_(ir_context),
      fact_manager_(fact_manager),
      fuzzer_context_(fuzzer_context),
      transformations_(transformations) {}

FuzzerPass::~FuzzerPass() = default;

std::vector<opt::Instruction*> FuzzerPass::FindAvailableInstructions(
        const opt::Function& function,
        opt::BasicBlock* block,
        opt::BasicBlock::iterator inst_it,
        std::function<bool(opt::IRContext*, opt::Instruction*)> instruction_is_relevant) {

  // Populate list of available instructions that satisfy |instruction_is_relevant|.
  // TODO(afd) The following is (relatively) simple, but may end up being
  //  prohibitively inefficient, as it walks the whole dominator tree for
  //  every copy that is added.

  std::vector<opt::Instruction*> result;
  // Consider all global declarations
  for (auto& global : GetIRContext()->module()->types_values()) {
    if (instruction_is_relevant(GetIRContext(), &global)) {
      result.push_back(&global);
    }
  }

  // Consider all previous instructions in this block
  for (auto prev_inst_it = block->begin(); prev_inst_it != inst_it;
       ++prev_inst_it) {
    if (instruction_is_relevant(GetIRContext(),
                                &*prev_inst_it)) {
      result.push_back(&*prev_inst_it);
    }
  }

  // Walk the dominator tree to consider all instructions from dominating
  // blocks
  auto dominator_analysis =
          GetIRContext()->GetDominatorAnalysis(&function);
  for (auto next_dominator =
          dominator_analysis->ImmediateDominator(block);
       next_dominator != nullptr;
       next_dominator =
               dominator_analysis->ImmediateDominator(next_dominator)) {
    for (auto& dominating_inst : *next_dominator) {
      if (instruction_is_relevant(GetIRContext(),
                                  &dominating_inst)) {
        result.push_back(&dominating_inst);
      }
    }
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
