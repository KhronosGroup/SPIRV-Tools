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
#include "source/opt/instruction.h"

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
  // A vector of matching OpLoad and OpStore instructions.
  std::vector<std::pair<opt::Instruction*, opt::Instruction*>>
      op_load_store_pairs;

  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // A hash map storing potential OpLoad instructions.
      std::unordered_map<uint32_t, opt::Instruction*> current_op_loads;
      for (auto& instruction : block) {
        // Add an potential OpLoad instruction.
        if (instruction.opcode() == SpvOpLoad) {
          current_op_loads[instruction.result_id()] = &instruction;
        } else if (instruction.opcode() == SpvOpStore) {
          if (current_op_loads.find(instruction.GetSingleWordOperand(1)) !=
              current_op_loads.end()) {
            // We have found the matching OpLoad instruction to the current
            // OpStore instruction.
            op_load_store_pairs.push_back(std::make_pair(
                current_op_loads[instruction.GetSingleWordOperand(1)],
                &instruction));
          }
          current_op_loads.clear();
        } else if (instruction.opcode() == SpvOpCopyMemory ||
                   instruction.opcode() == SpvOpCopyMemorySized ||
                   instruction.IsAtomicOp()) {
          // We need to make sure that the value pointed by source of OpLoad
          // hasn't changed by the time we see matching OpStore instruction.
          current_op_loads.clear();
        } else if (instruction.opcode() == SpvOpMemoryBarrier ||
                   instruction.opcode() == SpvOpMemoryNamedBarrier) {
          for (auto it = current_op_loads.begin();
               it != current_op_loads.end();) {
            opt::Instruction* source_id =
                GetIRContext()->get_def_use_mgr()->GetDef(
                    it->second->GetSingleWordOperand(2));
            SpvStorageClass storage_class =
                fuzzerutil::GetStorageClassFromPointerType(
                    GetIRContext(), source_id->type_id());
            switch (storage_class) {
                // These storage classes of the source variable of an potential
                // OpLoad instruction don't invalidate it.
              case SpvStorageClassUniformConstant:
              case SpvStorageClassInput:
              case SpvStorageClassUniform:
              case SpvStorageClassPrivate:
              case SpvStorageClassFunction:
                it++;
                break;
              default:
                it = current_op_loads.erase(it);
                break;
            }
          }
        }
      }
    }
  }
  for (auto instr_pair : op_load_store_pairs) {
    // Randomly decide to apply the transformation for the
    // potential pairs.
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
