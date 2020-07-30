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
#include "source/fuzz/transformation_composite_insert.h"

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
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingCompositeInsert())) {
          return;
        }

        // It must be valid to insert an OpCompositeInsert instruction
        // before |instruction_iterator|.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                SpvOpCompositeInsert, instruction_iterator)) {
          return;
        }

        // Look for available values that have the composite type.
        std::vector<opt::Instruction*> available_composite_values =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [this, instruction_descriptor](
                    opt::IRContext* ir_context,
                    opt::Instruction* instruction) -> bool {

                  auto instruction_type = ir_context->get_type_mgr()->GetType(
                      instruction->type_id());
                  if (!fuzzerutil::IsCompositeType(instruction_type)) {
                    return false;
                  }

                  return fuzzerutil::IdIsAvailableBeforeInstruction(
                      ir_context,
                      FindInstruction(instruction_descriptor, ir_context),
                      instruction->result_id());
                });

        // If there are no available values, then return.
        if (available_composite_values.empty()) {
          return;
        }

        // Choose randomly one available composite value.
        auto available_composite_value =
            available_composite_values[GetFuzzerContext()->RandomIndex(
                available_composite_values)];

        // Take a random component of the chosen composite value.
        auto num_components = 0;
        uint32_t index_to_replace =
            GetFuzzerContext()->GetRandomIndexForAccessChain(num_components);
        auto component_to_replace_id =
            available_composite_value->GetSingleWordInOperand(index_to_replace);
        auto component_to_replace_type_id =
            GetIRContext()
                ->get_def_use_mgr()
                ->GetDef(component_to_replace_id)
                ->type_id();

        // Look for available values that have the same type id as the
        // |constant_to_replace|.
        std::vector<opt::Instruction*> available_values =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [this, instruction_descriptor, component_to_replace_type_id](
                    opt::IRContext* ir_context,
                    opt::Instruction* instruction) -> bool {

                  if (instruction->type_id() != component_to_replace_type_id) {
                    return false;
                  }

                  return fuzzerutil::IdIsAvailableBeforeInstruction(
                      ir_context,
                      FindInstruction(instruction_descriptor, ir_context),
                      instruction->result_id());
                });

        // Choose randomly one available value.
        auto available_value =
            available_values[GetFuzzerContext()->RandomIndex(available_values)];

        auto new_result_id = GetFuzzerContext()->GetFreshId();

        // Insert an OpCompositeInsert instruction which copies
        // |available_composite_value| and in the copied composite constant
        // replaces |component_to_replace| with |available_value|.
        TransformationAddCompositeInsert transformation =
            TransformationAddCompositeInsert(
                new_result_id, available_value->result_id(),
                available_composite_value->result_id(), index_to_replace,
                instruction_descriptor);
        ApplyTransformation(transformation);

        // Every element which hasn't been changed in the copy is
        // synonymous to the corresponding element in the original composite
        // value. The element which has been changed is synonymous to the
        // value |available_value| itself.
        for (uint32_t i = 0; i < num_components; i++) {
          if (i != index_to_replace) {
            GetTransformationContext()->GetFactManager()->AddFactDataSynonym(
                MakeDataDescriptor(new_result_id, {i}),
                MakeDataDescriptor(available_composite_value->result_id(), {i}),

                GetIRContext());
          } else {
            GetTransformationContext()->GetFactManager()->AddFactDataSynonym(
                MakeDataDescriptor(new_result_id, {index_to_replace}),
                MakeDataDescriptor(available_composite_value->result_id(), {}),
                GetIRContext());
          }
        }

      });
}
/*
uint32_t WalkOneCompositeTypeIndex(opt::IRContext* context,
                                   uint32_t base_object_type_id,
                                   uint32_t index) {
  auto should_be_composite_type =
      context->get_def_use_mgr()->GetDef(base_object_type_id);
  assert(should_be_composite_type && "The type should exist.");
  switch (should_be_composite_type->opcode()) {
    case SpvOpTypeArray: {
      auto array_length = GetArraySize(*should_be_composite_type, context);
      if (array_length == 0 || index >= array_length) {
        return 0;
      }
      return should_be_composite_type->GetSingleWordInOperand(0);
    }
    case SpvOpTypeMatrix:
    case SpvOpTypeVector: {
      auto count = should_be_composite_type->GetSingleWordInOperand(1);
      if (index >= count) {
        return 0;
      }
      return should_be_composite_type->GetSingleWordInOperand(0);
    }
    case SpvOpTypeStruct: {
      if (index >= GetNumberOfStructMembers(*should_be_composite_type)) {
        return 0;
      }
      return should_be_composite_type->GetSingleWordInOperand(index);
    }
    default:
      return 0;
  }
}
*/

uint32_t FuzzerPassAddCompositeInserts::GetNumberOfComponents(
    opt::IRContext* ir_context, uint32_t composite_type_id) {
  auto composite_type =
      ir_context->get_def_use_mgr()->GetDef(composite_type_id);
  assert(composite_type && "The type should exist.");
  uint32_t size;
  switch (composite_type->opcode()) {
    case SpvOpTypeArray:
      size = fuzzerutil::GetArraySize(*composite_type, ir_context);
      break;
    case SpvOpTypeMatrix:
    case SpvOpTypeVector:
      size = composite_type->GetSingleWordInOperand(1);
      break;
    case SpvOpTypeStruct:
      size = fuzzerutil::GetNumberOfStructMembers(*composite_type);
      break;
    default:
      size = 0;
      break;
  }
  return size;
}

}  // namespace fuzz
}  // namespace spvtools
