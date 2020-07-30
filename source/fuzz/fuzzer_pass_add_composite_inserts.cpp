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
#include "source/fuzz/transformation_composite_extract.h"
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

        // Look for available values that have composite type.
        std::vector<opt::Instruction*> available_composites =
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
                  // No components of the composite can have type
                  // OpTypeRuntimeArray.
                  if (ContainsRuntimeArray(instruction_type)) {
                    return false;
                  }

                  return fuzzerutil::IdIsAvailableBeforeInstruction(
                      ir_context,
                      FindInstruction(instruction_descriptor, ir_context),
                      instruction->result_id());
                });

        // If there are no available values, then return.
        if (available_composites.empty()) {
          return;
        }

        // Choose randomly one available composite value.
        auto available_composite =
            available_composites[GetFuzzerContext()->RandomIndex(
                available_composites)];

        // Take a random component of the chosen composite value. If the chosen
        // component is itself a composite, then randomly decide whether to take
        // its component itself and repeat. Use OpCompositeExtract to get the
        // component.
        bool reached_end_node = false;
        uint32_t current_node_type_id = available_composite->type_id();
        uint32_t one_selected_index;
        uint32_t num_of_components;
        std::vector<uint32_t> index_to_replace;
        while (!reached_end_node) {
          num_of_components =
              GetNumberOfComponents(GetIRContext(), current_node_type_id);
          one_selected_index = GetFuzzerContext()->GetRandomIndexForAccessChain(
              num_of_components);
          /*
          TransformationCompositeExtract transformation_extract =
              TransformationCompositeExtract(instruction_descriptor, fresh_id,
                                             current_node->result_id(),
                                             {one_selected_index});
          ApplyTransformation(transformation_extract);
           */
          // Construct a final index by appending current index.
          index_to_replace.push_back(one_selected_index);
          current_node_type_id = fuzzerutil::WalkOneCompositeTypeIndex(
              GetIRContext(), current_node_type_id, one_selected_index);

          // If the component is not a composite or if we decide not to go
          // deeper, then end the iteration.
          if (!fuzzerutil::IsCompositeType(
                  GetIRContext()->get_type_mgr()->GetType(
                      current_node_type_id)) ||
              !GetFuzzerContext()->ChoosePercentage(
                  GetFuzzerContext()
                      ->GetChanceOfGoingDeeperToInsertInComposite())) {
            reached_end_node = true;
          }
        }

        // Look for available objects that have the type id
        // |component_to_replace_type_id|
        std::vector<opt::Instruction*> available_objects =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [this, instruction_descriptor, current_node_type_id](
                    opt::IRContext* ir_context,
                    opt::Instruction* instruction) -> bool {

                  if (instruction->type_id() != current_node_type_id) {
                    return false;
                  }

                  return fuzzerutil::IdIsAvailableBeforeInstruction(
                      ir_context,
                      FindInstruction(instruction_descriptor, ir_context),
                      instruction->result_id());
                });

        // If there are no objects of the specific type available, create a zero
        // constant of this type.
        uint32_t available_object_id;
        if (available_objects.empty()) {
          available_object_id =
              FindOrCreateZeroConstant(current_node_type_id, false);
        } else {
          available_object_id =
              available_objects[GetFuzzerContext()->RandomIndex(
                                    available_objects)]
                  ->result_id();
        }

        auto new_result_id = GetFuzzerContext()->GetFreshId();

        // Insert an OpCompositeInsert instruction which copies
        // |available_composite| and in the copied composite inserts the object
        // of type |available_object_id| at index |index_to_replace|.
        TransformationCompositeInsert transformation =
            TransformationCompositeInsert(instruction_descriptor, new_result_id,
                                          available_composite->result_id(),
                                          available_object_id,
                                          std::move(index_to_replace));
        ApplyTransformation(transformation);

        // MakeDataDescriptor(new_result_id, {1});
        // Every element which hasn't been changed in the copy is
        // synonymous to the corresponding element in the original composite
        // value. The element which has been changed is synonymous to the
        // value |available_value| itself.
        /*
        for (uint32_t i = 0; i < num_of_components; i++) {
          if (i != index_to_replace) {
            GetTransformationContext()->GetFactManager()->AddFactDataSynonym(
                MakeDataDescriptor(new_result_id, {i}),
                MakeDataDescriptor(available_composite->result_id(), {i}),

                GetIRContext());
          } else {
            GetTransformationContext()->GetFactManager()->AddFactDataSynonym(
                MakeDataDescriptor(new_result_id, {index_to_replace}),
                MakeDataDescriptor(available_composite->result_id(), {}),
                GetIRContext());
          }
        }*/

      });
}

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

bool FuzzerPassAddCompositeInserts::ContainsRuntimeArray(
    const opt::analysis::Type* type) {
  switch (type->kind()) {
    case opt::analysis::Type::kRuntimeArray:
      return true;
    case opt::analysis::Type::kStruct:
      return std::any_of(type->AsStruct()->element_types().begin(),
                         type->AsStruct()->element_types().end(),
                         [](const opt::analysis::Type* element_type) {
                           return ContainsRuntimeArray(element_type);
                         });
    default:
      return false;
  }
}

}  // namespace fuzz
}  // namespace spvtools
