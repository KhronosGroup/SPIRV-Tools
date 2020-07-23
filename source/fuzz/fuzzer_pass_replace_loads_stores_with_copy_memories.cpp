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

#include "source/fuzz/fuzzer_pass_replace_loads_stores_with_copy_memories.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_replace_load_store_with_copy_memory.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceLoadsStoresWithCopyMemories::
    FuzzerPassReplaceLoadsStoresWithCopyMemories(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceLoadsStoresWithCopyMemories::
    ~FuzzerPassReplaceLoadsStoresWithCopyMemories() = default;

void FuzzerPassReplaceLoadsStoresWithCopyMemories::Apply() {
  // This is the vector of matching OpLoad and OpStore instructions.
  std::vector<std::pair<opt::Instruction*, opt::Instruction*>>
      op_load_store_pairs;
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // Consider separately every block.
      std::unordered_map<uint32_t, opt::Instruction*> current_op_loads = {};
      for (auto& instruction : block) {
        // Add an potential OpLoad instruction.
        if (instruction.opcode() == SpvOpLoad) {
          current_op_loads[instruction.result_id()] = &instruction;
        }
        if (instruction.opcode() == SpvOpStore) {
          // We have found the matching OpLoad instruction to the current
          // OpStore instruction.
          if (current_op_loads.find(instruction.GetSingleWordOperand(1)) !=
              current_op_loads.end()) {
            op_load_store_pairs.push_back(std::make_pair(
                current_op_loads[instruction.GetSingleWordOperand(1)],
                &instruction));
            // We need to clear the hash map. If we don't, there might be
            // interfering OpStore instructions. Consider for example:
            // %a = OpLoad %ptr1
            // OpStore %ptr2 %a <-- we haven't clear the map
            // OpStore %ptr3 %a <-- if %ptr2 points to the same variable as
            //    %ptr1, then replacing this instruction with OpCopyMemory %ptr3
            //    %ptr1 is unsafe.
            current_op_loads.clear();
          }
        }
      }
    }
  }
  for (auto instr_pair : op_load_store_pairs) {
    // Randomly decide to apply the transformation for the potaential pairs.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfReplacingLoadStoreWithCopyMemory())) {
      ApplyTransformation(TransformationReplaceLoadStoreWithCopyMemory(
          MakeInstructionDescriptor(GetIRContext(), instr_pair.first),
          MakeInstructionDescriptor(GetIRContext(), instr_pair.second)));
    }
  }
}
}  // namespace fuzz
}  // namespace spvtools