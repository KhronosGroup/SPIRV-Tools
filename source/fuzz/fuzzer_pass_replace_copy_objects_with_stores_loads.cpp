// Copyright (c) 2020 Google
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

#include "source/fuzz/fuzzer_pass_replace_copy_objects_with_stores_loads.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_replace_copy_object_with_store_load.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceCopyObjectsWithStoresLoads::
    FuzzerPassReplaceCopyObjectsWithStoresLoads(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceCopyObjectsWithStoresLoads::
    ~FuzzerPassReplaceCopyObjectsWithStoresLoads() = default;

void FuzzerPassReplaceCopyObjectsWithStoresLoads::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator instruction_iterator,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        assert(instruction_iterator->opcode() ==
                   instruction_descriptor.target_instruction_opcode() &&
               "The opcode of the instruction we might insert before must be "
               "the same as the opcode in the descriptor for the instruction");

        // Randomly decide whether to replace OpCopyObject.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()
                    ->GetChanceOfReplacingCopyObjectWithStoreLoad())) {
          return;
        }

        // It must be valid to insert OpStore and OpLoad instructions
        // before the instruction to insert before.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                SpvOpStore, instruction_iterator) ||
            !fuzzerutil::CanInsertOpcodeBeforeInstruction(
                SpvOpLoad, instruction_iterator)) {
          return;
        }

        // Randomly decides whether a global or local variable will be added.
        auto variable_storage_class = GetFuzzerContext()->ChooseEven()
                                          ? SpvStorageClassPrivate
                                          : SpvStorageClassFunction;

        // Gets the available basic and pointer types.
        auto basic_type_ids_and_pointers =
            GetAvailableBasicTypesAndPointers(variable_storage_class);
        auto& basic_types = basic_type_ids_and_pointers.first;
        uint32_t basic_type_id =
            basic_types[GetFuzzerContext()->RandomIndex(basic_types)];

        // Looks for OpCopyObject instructions we would like to replace
        std::vector<opt::Instruction*> value_instructions =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [basic_type_id, instruction_descriptor](
                    opt::IRContext*, opt::Instruction* instruction) -> bool {
                  if (!instruction->result_id() || !instruction->type_id()) {
                    return false;
                  }

                  if (instruction->type_id() != basic_type_id) {
                    return false;
                  }
                  return (instruction->opcode() == SpvOpCopyObject);
                });

        if (value_instructions.empty()) {
          return;
        }

        // If the pointer type does not exist, then create it.
        FindOrCreatePointerType(basic_type_id, variable_storage_class);

        // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3403):
        //  type support here is limited by the FindOrCreateZeroConstant
        //  function.
        const auto* type_inst =
            GetIRContext()->get_def_use_mgr()->GetDef(basic_type_id);
        assert(type_inst);
        switch (type_inst->opcode()) {
          case SpvOpTypeBool:
          case SpvOpTypeFloat:
          case SpvOpTypeInt:
          case SpvOpTypeArray:
          case SpvOpTypeMatrix:
          case SpvOpTypeVector:
          case SpvOpTypeStruct:
            break;
          default:
            return;
        }

        // Create a constant to initialize the variable from. This might update
        // module's id bound so it must be done before any fresh ids are
        // computed.
        auto initializer_id = FindOrCreateZeroConstant(basic_type_id);

        // Applies the transformation: replace CopyObject with Store, Load.
        ApplyTransformation(TransformationReplaceCopyObjectWithStoreLoad(
            value_instructions[GetFuzzerContext()->RandomIndex(
                                   value_instructions)]
                ->result_id(),
            GetFuzzerContext()->GetFreshId(), variable_storage_class,
            initializer_id));
      });
}

}  // namespace fuzz
}  // namespace spvtools
