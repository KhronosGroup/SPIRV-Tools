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

#include "source/fuzz/fuzzer_pass_changing_memory_semantics.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation_changing_memory_semantics.h"

namespace spvtools {
namespace fuzz {

FuzzerPassChangingMemorySemantics::FuzzerPassChangingMemorySemantics(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassChangingMemorySemantics::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* /* unused */, opt::BasicBlock* /* unused */,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor) {
        // The instruction must have at least one memory semantics operand.
        auto number_of_memory_semantics =
            TransformationChangingMemorySemantics::GetNumberOfMemorySemantics(
                inst_it->opcode());
        if (number_of_memory_semantics == 0) {
          return;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfChangingMemorySemantics())) {
          return;
        }

        // If the instruction has two memory semantic operands pick one
        // randomly.
        uint32_t memory_semantics_operand_position = 0;
        if (number_of_memory_semantics == 2) {
          std::vector<uint32_t> operand_positions{0, 1};
          memory_semantics_operand_position =
              GetFuzzerContext()->RandomIndex(operand_positions);
        }

        auto needed_index = TransformationChangingMemorySemantics::
            GetMemorySemanticsInOperandIndex(inst_it->opcode(),
                                             memory_semantics_operand_position);

        auto memory_semantics_value =
            GetIRContext()
                ->get_def_use_mgr()
                ->GetDef(inst_it->GetSingleWordInOperand(needed_index))
                ->GetSingleWordInOperand(0);

        auto lower_bits_old_memory_semantics =
            static_cast<SpvMemorySemanticsMask>(
                memory_semantics_value & TransformationChangingMemorySemantics::
                                             kMemorySemanticsLowerBitmask);
        auto higher_bits_old_memory_semantics =
            static_cast<SpvMemorySemanticsMask>(
                memory_semantics_value & TransformationChangingMemorySemantics::
                                             kMemorySemanticsHigherBitmask);

        std::vector<SpvMemorySemanticsMask> potential_new_memory_orders{
            SpvMemorySemanticsMaskNone, SpvMemorySemanticsAcquireMask,
            SpvMemorySemanticsReleaseMask, SpvMemorySemanticsAcquireReleaseMask,
            SpvMemorySemanticsSequentiallyConsistentMask};

        // Remove the memory mask is not applicable for the new instruction.
        std::remove_if(potential_new_memory_orders.begin(),
                       potential_new_memory_orders.end(),
                       [inst_it, lower_bits_old_memory_semantics](
                           SpvMemorySemanticsMask /*unused*/) {
                         switch (inst_it->opcode()) {
                           case SpvOpAtomicLoad:
                             return TransformationChangingMemorySemantics::
                                 IsAtomicLoadMemorySemanticsValue(
                                     lower_bits_old_memory_semantics);

                           case SpvOpAtomicStore:
                             return TransformationChangingMemorySemantics::
                                 IsAtomicStoreMemorySemanticsValue(
                                     lower_bits_old_memory_semantics);

                           case SpvOpAtomicExchange:
                           case SpvOpAtomicIIncrement:
                           case SpvOpAtomicIDecrement:
                           case SpvOpAtomicIAdd:
                           case SpvOpAtomicISub:
                           case SpvOpAtomicSMin:
                           case SpvOpAtomicUMin:
                           case SpvOpAtomicSMax:
                           case SpvOpAtomicUMax:
                           case SpvOpAtomicAnd:
                           case SpvOpAtomicOr:
                           case SpvOpAtomicXor:
                           case SpvOpAtomicCompareExchange:
                           case SpvOpAtomicCompareExchangeWeak:

                             return TransformationChangingMemorySemantics::
                                 IsAtomicRMWInstructionsemorySemanticsValue(
                                     lower_bits_old_memory_semantics);

                           case SpvOpControlBarrier:
                           case SpvOpMemoryBarrier:
                           case SpvOpMemoryNamedBarrier:

                             return TransformationChangingMemorySemantics::
                                 IsBarrierInstructionsMemorySemanticsValue(
                                     lower_bits_old_memory_semantics);

                           default:
                             return false;
                         }
                       });

        // Randomly choose a new memory semantics value (lower bits).
        auto lower_bits_new_memory_semantics =
            potential_new_memory_orders[GetFuzzerContext()->RandomIndex(
                potential_new_memory_orders)];
        // Bitwise-OR with the higher bits of the old value, to get the new
        // value (all bits).
        auto memory_semantic_new_total_value =
            lower_bits_new_memory_semantics | higher_bits_old_memory_semantics;

        uint32_t new_memory_semantics_id = FindOrCreateConstant(
            {static_cast<uint32_t>(memory_semantic_new_total_value)},
            FindOrCreateIntegerType(32, GetFuzzerContext()->ChooseEven()),
            false);

        auto needed_instruction =
            FindInstruction(instruction_descriptor, GetIRContext());
        auto memory_model = static_cast<SpvMemoryModel>(
            GetIRContext()->module()->GetMemoryModel()->GetSingleWordInOperand(
                1));

        if (!TransformationChangingMemorySemantics::IsSuitableStrengthening(
                GetIRContext(), needed_instruction,
                lower_bits_old_memory_semantics,
                lower_bits_new_memory_semantics,
                memory_semantics_operand_position, memory_model)) {
          return;
        }

        ApplyTransformation(TransformationChangingMemorySemantics(
            instruction_descriptor, memory_semantics_operand_position,
            new_memory_semantics_id));
      });
}

}  // namespace fuzz
}  // namespace spvtools
