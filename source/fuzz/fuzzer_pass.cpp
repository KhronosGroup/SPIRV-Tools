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
    const opt::Function& function, opt::BasicBlock* block,
    opt::BasicBlock::iterator inst_it,
    std::function<bool(opt::IRContext*, opt::Instruction*)>
        instruction_is_relevant) {
  // TODO(afd) The following is (relatively) simple, but may end up being
  //  prohibitively inefficient, as it walks the whole dominator tree for
  //  every instruction that is considered.

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
    if (instruction_is_relevant(GetIRContext(), &*prev_inst_it)) {
      result.push_back(&*prev_inst_it);
    }
  }

  // Walk the dominator tree to consider all instructions from dominating
  // blocks
  auto dominator_analysis = GetIRContext()->GetDominatorAnalysis(&function);
  for (auto next_dominator = dominator_analysis->ImmediateDominator(block);
       next_dominator != nullptr;
       next_dominator =
           dominator_analysis->ImmediateDominator(next_dominator)) {
    for (auto& dominating_inst : *next_dominator) {
      if (instruction_is_relevant(GetIRContext(), &dominating_inst)) {
        result.push_back(&dominating_inst);
      }
    }
  }
  return result;
}

void FuzzerPass::MaybeAddTransformationBeforeEachInstruction(
    std::function<uint32_t(
        const opt::Function& function, opt::BasicBlock* block,
        opt::BasicBlock::iterator inst_it, uint32_t base, uint32_t offset)>
        maybe_apply_transformation) {
  // Consider every block in every function.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // We now consider every instruction in the block, randomly deciding
      // whether to apply a transformation before it.

      // In order for transformations to insert new instructions, they need to
      // be able to identify the instruction to insert before.  We enable this
      // by tracking a base instruction, which must generate a result id, and
      // an offset (to allow us to identify instructions that do not generate
      // result ids).

      // The initial base instruction is the block label.
      uint32_t base = block.id();
      uint32_t offset = 0;
      // Consider every instruction in the block.
      for (auto inst_it = block.begin(); inst_it != block.end(); ++inst_it) {
        if (inst_it->HasResultId()) {
          // In the case that the instruction has a result id, we use the
          // instruction as its own base, with zero offset.
          base = inst_it->result_id();
          offset = 0;
        } else {
          // The instruction does not have a result id, so we need to identify
          // it via the latest instruction that did have a result id (base), and
          // an incremented offset.
          offset++;
        }

        // Invoke the provided function, which might apply a transformation.
        // Its return value informs us of how many instructions it inserted.
        // (This will be 0 if no transformation was applied.)
        uint32_t num_instructions_inserted =
            maybe_apply_transformation(function, &block, inst_it, base, offset);

        if (!inst_it->HasResultId()) {
          // We are tracking the current id-less instruction via an offset,
          // |offset|, from a previous instruction, |base|, that has an id. We
          // increment |offset| to reflect any newly-inserted instructions.
          offset += num_instructions_inserted;
        }
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
