// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/fuzzer_pass_push_ids_through_variables.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_local_variable.h"
#include "source/fuzz/transformation_add_type_pointer.h"
#include "source/fuzz/transformation_push_id_through_variable.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPushIdsThroughVariables::FuzzerPassPushIdsThroughVariables(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassPushIdsThroughVariables::
    ~FuzzerPassPushIdsThroughVariables() = default;

void FuzzerPassPushIdsThroughVariables::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function,
             opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor) -> void {
        assert(inst_it->opcode() == instruction_descriptor.target_instruction_opcode() &&
               "The opcode of the instruction we might insert before must be "
               "the same as the opcode in the descriptor for the instruction");

        // Check whether the store and load instructions can be
        // inserted before this instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpStore, inst_it) ||
            !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, inst_it)) {
          return;
        }

        // Randomly decide whether to try pushing an id through a variable.
        if (!GetFuzzerContext()->ChoosePercentage(GetFuzzerContext()->GetChanceOfPushingIdThroughVariable())) {
          return;
        }

        // Storage class for a global or local variable.
        SpvStorageClass variable_storage_class;
        if (GetFuzzerContext()->ChooseEven()) {
          variable_storage_class = SpvStorageClassPrivate;
        } else {
          variable_storage_class = SpvStorageClassFunction;
        }

        // Gets the basic type and pointers to basic types.
        auto basic_type_ids_and_pointers = GetAvailableBasicTypesAndPointers(variable_storage_class);
        auto& basic_types = basic_type_ids_and_pointers.first;
        uint32_t basic_type_id = basic_types[GetFuzzerContext()->RandomIndex(basic_types)];

        // Look for values we might consider to store.
        std::vector<opt::Instruction*> value_instructions =
            FindAvailableInstructions(
                function, block, inst_it,
                [basic_type_id](opt::IRContext* /*unused*/,
                          opt::Instruction* instruction) -> bool {
                  if (!instruction->result_id() || !instruction->type_id()) {
                    return false;
                  }
                  return instruction->type_id() == basic_type_id;
                });

        if (value_instructions.empty()) {
          return;
        }

        auto& basic_type_to_pointers = basic_type_ids_and_pointers.second;
        std::vector<uint32_t>& type_pointer_ids = basic_type_to_pointers.at(basic_type_id);
        uint32_t type_pointer_id;

        if (type_pointer_ids.empty()) {
          type_pointer_id = GetFuzzerContext()->GetFreshId();
          type_pointer_ids.push_back(type_pointer_id);
          ApplyTransformation(TransformationAddTypePointer(type_pointer_id, variable_storage_class, basic_type_id));
        } else {
          type_pointer_id = type_pointer_ids[GetFuzzerContext()->RandomIndex(type_pointer_ids)];
        }

        // Create whether a global variable
        // or a local variable.
        uint32_t variable_id = GetFuzzerContext()->GetFreshId();
        if (variable_storage_class == SpvStorageClassPrivate) {
          ApplyTransformation(TransformationAddGlobalVariable(
              variable_id, type_pointer_id,
              variable_storage_class, FindOrCreateZeroConstant(basic_type_id), false));
        } else {
          ApplyTransformation(TransformationAddLocalVariable(
              variable_id, type_pointer_id, function->result_id(),
              FindOrCreateZeroConstant(basic_type_id), false));
        }

        ApplyTransformation(TransformationPushIdThroughVariable(
            GetFuzzerContext()->GetFreshId(),
            variable_id,
            value_instructions[GetFuzzerContext()->RandomIndex(value_instructions)]->result_id(),
            instruction_descriptor));
      });
}

}  // namespace fuzz
}  // namespace spvtools
