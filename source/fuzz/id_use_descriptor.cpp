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

#include "source/fuzz/id_use_descriptor.h"

namespace spvtools {
namespace fuzz {
namespace module_navigation {

uint32_t IdUseDescriptor::GetIdOfInterest() const { return id_of_interest_; }

opt::Instruction* IdUseDescriptor::FindInstruction(
    spvtools::opt::IRContext* context) const {
  for (auto& function : *context->module()) {
    for (auto& block : function) {
      bool found_base = block.id() == base_instruction_result_id_;
      uint32_t num_ignored = 0;
      for (auto& instruction : block) {
        if (instruction.HasResultId() &&
            instruction.result_id() == base_instruction_result_id_) {
          assert(!found_base &&
                 "It should not be possible to find the base instruction "
                 "multiple times.");
          found_base = true;
          assert(num_ignored == 0 &&
                 "The skipped instruction count should only be incremented "
                 "after the instruction base has been found.");
        }
        if (found_base && instruction.opcode() == target_instruction_opcode_) {
          if (num_ignored == num_opcodes_to_ignore_) {
            if (in_operand_index_ >= instruction.NumInOperands()) {
              return nullptr;
            }
            auto in_operand = instruction.GetInOperand(in_operand_index_);
            if (in_operand.type != SPV_OPERAND_TYPE_ID) {
              return nullptr;
            }
            if (in_operand.words[0] != id_of_interest_) {
              return nullptr;
            }
            return &instruction;
          }
          num_ignored++;
        }
      }
      if (found_base) {
        // We found the base instruction, but did not find the target
        // instruction in the same block.
        return nullptr;
      }
    }
  }
  return nullptr;
}

}  // namespace module_navigation
}  // namespace fuzz
}  // namespace spvtools
