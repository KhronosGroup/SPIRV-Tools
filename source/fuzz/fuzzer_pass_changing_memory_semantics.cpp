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
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& /*unused*/) {
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfChangingMemorySemantics())) {
          return;
        }

        // Get all atomic instructions.
        std::vector<opt::Instruction*> atomic_instructions =
            FindAvailableInstructions(
                function, block, inst_it,
                [this](opt::IRContext* /*unused*/,
                       opt::Instruction* instruction) -> bool {
                  switch (instruction->opcode()) {
                    case SpvOpAtomicLoad:
                    case SpvOpAtomicStore:
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
                    case SpvOpAtomicFlagTestAndSet:
                    case SpvOpAtomicFlagClear:
                    case SpvOpAtomicFAddEXT:
                    case SpvOpAtomicCompareExchange:
                    case SpvOpAtomicCompareExchangeWeak:
                      return true;
                    default:
                      return false;
                  }
                });

        if (atomic_instructions.empty()) {
          return;
        }
        auto chosen_instruction =
            atomic_instructions[GetFuzzerContext()->RandomIndex(
                atomic_instructions)];

        // It will have a value of range from 0 to 3.
        uint32_t index =
            GetFuzzerContext()->RandomIndex(atomic_instructions) % 4;

        std::vector<SpvMemorySemanticsMask> memory_semanitcs_masks{
            SpvMemorySemanticsMaskNone, SpvMemorySemanticsAcquireMask,
            SpvMemorySemanticsReleaseMask, SpvMemorySemanticsAcquireReleaseMask,
            SpvMemorySemanticsSequentiallyConsistentMask};
        uint32_t new_memory_semantics_id = FindOrCreateConstant(
            {static_cast<uint32_t>(
                memory_semanitcs_masks[GetFuzzerContext()->RandomIndex(
                    memory_semanitcs_masks)])},
            FindOrCreateIntegerType(32, GetFuzzerContext()->ChooseEven()),
            false);

        // (NOTE - Need suggestion here) This valid for atomic instructions has
        // result id only (NOT FINAL).
        ApplyTransformation(TransformationChangingMemorySemantics(
            MakeInstructionDescriptor(chosen_instruction->result_id(),
                                      chosen_instruction->opcode(), 0),
            index, new_memory_semantics_id));
        // ANOTHER SOLUTION.
        /*
        - First will check instruction related to instruction_descriptor, then
          check if the instruction is atomic instruction. Then....

        - ApplyTransformation(TransformationChangingMemorySemantics(
        instruction_descriptor,
        index, new_memory_semantics_id));

        */
      });
}

}  // namespace fuzz
}  // namespace spvtools
