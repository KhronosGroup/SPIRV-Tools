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

#include "source/fuzz/fuzzer_pass_add_memory_barrier.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_memory_barrier.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddMemoryBarrier::FuzzerPassAddMemoryBarrier(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddMemoryBarrier::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* /* Unused */, opt::BasicBlock* /* Unused */,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        assert(inst_it->opcode() ==
                   instruction_descriptor.target_instruction_opcode() &&
               "The opcode of the instruction we might insert before must be "
               "the same as the opcode in the descriptor for the instruction");

        // Randomly decide whether to try inserting memory barrier instruction
        // here.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingMemoryBarrier())) {
          return;
        }

        // Check whether if legitimate to insert memory barrier instruction
        // before this instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpMemoryBarrier,
                                                          inst_it)) {
          return;
        }

        uint32_t memory_scope_id = FindOrCreateConstant(
            {SpvScopeInvocation},
            FindOrCreateIntegerType(32, GetFuzzerContext()->ChooseEven()),
            false);

        std::vector<SpvMemorySemanticsMask>
            potential_memory_semantics_values_higher_bits{
                SpvMemorySemanticsUniformMemoryMask,
                SpvMemorySemanticsWorkgroupMemoryMask};

        auto total_memory_semantics_value = static_cast<uint32_t>(
            SpvMemorySemanticsMaskNone |
            potential_memory_semantics_values_higher_bits
                [GetFuzzerContext()->RandomIndex(
                    potential_memory_semantics_values_higher_bits)]);

        uint32_t memory_semantics_id = FindOrCreateConstant(
            {total_memory_semantics_value},
            FindOrCreateIntegerType(32, GetFuzzerContext()->ChooseEven()),
            false);

        // Create and apply the transformation.
        ApplyTransformation(TransformationAddMemoryBarrier(
            memory_scope_id, memory_semantics_id, instruction_descriptor));
      });
}

}  // namespace fuzz
}  // namespace spvtools
