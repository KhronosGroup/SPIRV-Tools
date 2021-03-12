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

#include "source/fuzz/available_instructions.h"

namespace spvtools {
namespace fuzz {

AvailableInstructions::AvailableInstructions(
    opt::IRContext* ir_context,
    const std::function<bool(opt::IRContext*, opt::Instruction*)>& filter)
    : ir_context_(ir_context) {
  // Consider all global declarations
  for (auto& global : ir_context->module()->types_values()) {
    if (filter(ir_context, &global)) {
      available_globals_.push_back(&global);
    }
  }

  for (auto& function : *ir_context->module()) {
    std::vector<opt::Instruction*> params;
    function.ForEachParam(
        [this, &filter, ir_context, &params](opt::Instruction* param) {
          if (filter(ir_context, param)) {
            params.push_back(param);
          }
        });
    available_params_.insert({&function, params});

    auto dominator_analysis = ir_context->GetDominatorAnalysis(&function);
    for (auto& block : function) {
      if (&block == &*function.begin()) {
        num_available_at_block_entry_.insert(
            {&block, params.size() + available_globals_.size()});
      } else {
        auto immediate_dominator =
            dominator_analysis->ImmediateDominator(&block);
        num_available_at_block_entry_.insert(
            {&block,
             generated_by_block_.at(immediate_dominator).size() +
                 num_available_at_block_entry_.at(immediate_dominator)});
      }
      std::vector<opt::Instruction*> generated;
      for (auto& inst : block) {
        num_available_before_instruction_.insert(
            {&inst,
             num_available_at_block_entry_.at(&block) + generated.size()});
        if (filter(ir_context, &inst)) {
          generated.push_back(&inst);
        }
      }
      generated_by_block_.insert({&block, generated});
    }
  }
}

AvailableInstructions::AvailableBeforeInstruction::AvailableBeforeInstruction(
    const AvailableInstructions& available_instructions, opt::Instruction* inst)
    : available_instructions_(available_instructions), inst_(inst) {}

uint32_t AvailableInstructions::AvailableBeforeInstruction::size() const {
  return available_instructions_.num_available_before_instruction_.at(inst_);
}

bool AvailableInstructions::AvailableBeforeInstruction::empty() const {
  return size() == 0;
}

opt::Instruction* AvailableInstructions::AvailableBeforeInstruction::operator[](
    uint32_t index) const {
  if (index < available_instructions_.available_globals_.size()) {
    return available_instructions_.available_globals_[index];
  }
  auto block = available_instructions_.ir_context_->get_instr_block(inst_);
  auto function = block->GetParent();

  if (index <
      available_instructions_.available_globals_.size() +
          available_instructions_.available_params_.at(function).size()) {
    return available_instructions_.available_params_.at(
        function)[index - available_instructions_.available_globals_.size()];
  }

  auto dominator_analysis =
      available_instructions_.ir_context_->GetDominatorAnalysis(function);

  for (auto* ancestor = block;;
       ancestor = dominator_analysis->ImmediateDominator(ancestor)) {
    if (index >=
        available_instructions_.num_available_at_block_entry_.at(ancestor)) {
      return available_instructions_.generated_by_block_.at(
          ancestor)[index - available_instructions_
                                .num_available_at_block_entry_.at(ancestor)];
    }
  }

  assert(false && "Unreachable.");
  return nullptr;
}

}  // namespace fuzz
}  // namespace spvtools
