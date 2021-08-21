// Copyright (c) 2021 Mostafa Ashraf
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

#include "source/fuzz/fuzzer_pass_changing_memory_scope.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation_changing_memory_scope.h"

namespace spvtools {
namespace fuzz {

FuzzerPassChangingMemoryScope::FuzzerPassChangingMemoryScope(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassChangingMemoryScope::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* /* unused */, opt::BasicBlock* /* unused */,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor) {
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfChangingMemoryScope())) {
          return;
        }

        // Instruction must be atomic instruction only.
        if (!TransformationChangingMemoryScope::IsAtomicInstruction(
                inst_it->opcode())) {
          return;
        }

        auto needed_index =
            TransformationChangingMemoryScope::GetMemoryScopeInOperandIndex(
                inst_it->opcode());

        auto old_memory_scope_value = static_cast<SpvScope>(
            GetIRContext()
                ->get_def_use_mgr()
                ->GetDef(inst_it->GetSingleWordInOperand(needed_index))
                ->GetSingleWordInOperand(0));

        auto new_memory_scopes =
            GetSuitableNewMemoryScope(old_memory_scope_value);

        if (new_memory_scopes.empty()) {
          return;
        }

        auto memory_scope_new_value =
            new_memory_scopes[GetFuzzerContext()->RandomIndex(
                new_memory_scopes)];

        uint32_t new_memory_scope_id = FindOrCreateConstant(
            {static_cast<uint32_t>(memory_scope_new_value)},
            FindOrCreateIntegerType(32, GetFuzzerContext()->ChooseEven()),
            false);

        ApplyTransformation(TransformationChangingMemoryScope(
            instruction_descriptor, new_memory_scope_id));
      });
}

std::vector<SpvScope> FuzzerPassChangingMemoryScope::GetSuitableNewMemoryScope(
    SpvScope old_memory_scope_value) {
  std::vector<SpvScope> potential_new_scopes_orders{
      SpvScopeCrossDevice, SpvScopeDevice, SpvScopeWorkgroup, SpvScopeSubgroup,
      SpvScopeInvocation};

  // std::remove_if does not actually remove any elements from the vector; it
  // just reorders them to the end of the vector.
  auto reordered_memory_semantics = std::remove_if(
      potential_new_scopes_orders.begin(), potential_new_scopes_orders.end(),
      [old_memory_scope_value](SpvScope memory_scope) {
        return (old_memory_scope_value == memory_scope ||
                memory_scope > old_memory_scope_value);
      });

  // Erase the old memory scope and scopes narrower than older.
  potential_new_scopes_orders.erase(reordered_memory_semantics,
                                    potential_new_scopes_orders.end());

  return potential_new_scopes_orders;
}

}  // namespace fuzz
}  // namespace spvtools
