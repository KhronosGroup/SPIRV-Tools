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

#include "source/fuzz/fuzzer_pass_add_composite_inserts.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/pseudo_random_generator.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddCompositeInserts::FuzzerPassAddCompositeInserts(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddCompositeInserts::~FuzzerPassAddCompositeInserts() = default;

void FuzzerPassAddCompositeInserts::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator instruction_iterator,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        assert(instruction_iterator->opcode() ==
                   instruction_descriptor.target_instruction_opcode() &&
               "The opcode of the instruction we might insert before must be "
               "the same as the opcode in the descriptor for the instruction");

        // Randomly decide whether to try adding an OpCompositeInsert
        // instruction.
        /*if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingCompositeInsert())) {
          return;
        }*/

        // It must be valid to insert an OpCompositeInsert instruction
        // before |instruction_iterator|.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                SpvOpCompositeInsert, instruction_iterator)) {
          return;
        }

        // Look for available constants that have type OpCompositeConstant.
        std::vector<opt::Instruction*> available_composite_constants =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [this, instruction_descriptor](
                    opt::IRContext* ir_context,
                    opt::Instruction* instruction) -> bool {

                  if (instruction->opcode() != SpvOpConstantComposite) {
                    return false;
                  }

                  return fuzzerutil::IdIsAvailableBeforeInstruction(
                      ir_context,
                      FindInstruction(instruction_descriptor, ir_context),
                      instruction->result_id());
                });

        // If there are no available constants, then return.
        if (available_composite_constants.empty()) {
          return;
        }

        // Choose randomly one available composite constant.
        auto available_composite_constant =
            available_composite_constants[GetFuzzerContext()->RandomIndex(
                available_composite_constants)];

        // Take a random component of the chosen composite constant.
        auto num_constants = available_composite_constant->NumInOperands();
        uint32_t index_to_replace =
            GetFuzzerContext()->GetRandomIndexForAccessChain(num_constants);
        auto constant_to_replace_id =
            available_composite_constant->GetSingleWordInOperand(
                index_to_replace);
        auto constant_to_replace_type_id = GetIRContext()
                                               ->get_def_use_mgr()
                                               ->GetDef(constant_to_replace_id)
                                               ->type_id();

        // Look for available constants that have the same type id as the
        // |constant_to_replace|.
        std::vector<opt::Instruction*> available_constants =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [this, instruction_descriptor, constant_to_replace_type_id](
                    opt::IRContext* ir_context,
                    opt::Instruction* instruction) -> bool {
                  if (instruction->type_id() != constant_to_replace_type_id) {
                    return false;
                  }
                  return fuzzerutil::IdIsAvailableBeforeInstruction(
                      ir_context,
                      FindInstruction(instruction_descriptor, ir_context),
                      instruction->result_id());
                });

        // Choose randomly one available constant.
        auto available_constant =
            available_constants[GetFuzzerContext()->RandomIndex(
                available_constants)];

        auto new_result_id = GetFuzzerContext()->GetFreshId();

        // Insert an OpCompositeInsert instructions which copies
        // |available_composite_constant| and in the copy replaces
        // |constant_to_replace| with |available_constant|.
        FindInstruction(instruction_descriptor, GetIRContext())
            ->InsertBefore(MakeUnique<opt::Instruction>(
                GetIRContext(), SpvOpCompositeInsert,
                available_composite_constant->type_id(), new_result_id,
                opt::Instruction::OperandList(
                    {{SPV_OPERAND_TYPE_ID, {available_constant->result_id()}},
                     {SPV_OPERAND_TYPE_ID,
                      {available_composite_constant->result_id()}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {index_to_replace}}})));

        fuzzerutil::UpdateModuleIdBound(GetIRContext(), new_result_id);
        GetIRContext()->InvalidateAnalysesExceptFor(
            opt::IRContext::kAnalysisNone);

        // Every every element which hasn't been changed in the copy is
        // synonymous to the corresponding element in the original composite
        // constant. The element which has been changed is synonymous to the
        // constant |available_constant| itself.
        for (uint32_t i = 0; i < num_constants; i++) {
          if (i != index_to_replace) {
            GetTransformationContext()->GetFactManager()->AddFactDataSynonym(
                MakeDataDescriptor(new_result_id, {i}),
                MakeDataDescriptor(available_composite_constant->result_id(),
                                   {i}),

                GetIRContext());
          } else {
            GetTransformationContext()->GetFactManager()->AddFactDataSynonym(
                MakeDataDescriptor(new_result_id, {index_to_replace}),
                MakeDataDescriptor(available_constant->result_id(), {}),
                GetIRContext());
          }
        }

      });
}

}  // namespace fuzz
}  // namespace spvtools
