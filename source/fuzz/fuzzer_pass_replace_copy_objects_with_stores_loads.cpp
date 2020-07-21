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
  GetIRContext()->module()->ForEachInst([this](opt::Instruction* instruction) {
    // The instruction must be OpCopyObject.
    if (instruction->opcode() != SpvOpCopyObject) return;
    // The |type_id()| of the instruction cannot be a pointer,
    if (instruction->type_id() == SpvOpTypePointer) return;

    // It must be valid to insert OpStore and OpLoad instructions
    // before the instruction OpCopyObject.
    if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpStore,
                                                      instruction) ||
        !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, instruction)) {
      return;
    }

    // Randomly decide whether to replace OpCopyObject.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfReplacingCopyObjectWithStoreLoad())) {
      return;
    }

    // Randomly decides whether a global or local variable will be added.
    auto variable_storage_class = GetFuzzerContext()->ChooseEven()
                                      ? SpvStorageClassPrivate
                                      : SpvStorageClassFunction;

    // Find or create a constant to initialize the variable from. This might
    // update module's id bound so it must be done before any fresh ids are
    // computed.

    auto variable_initializer_id =
        FindOrCreateZeroConstant(instruction->type_id());

    // Make sure that pointer type is defined.
    FindOrCreatePointerType(instruction->type_id(), variable_storage_class);
    // Applies the transformation replacing OpCopyObject with Store and Load.
    ApplyTransformation(TransformationReplaceCopyObjectWithStoreLoad(
        instruction->result_id(), GetFuzzerContext()->GetFreshId(),
        variable_storage_class, variable_initializer_id));
  });
}

}  // namespace fuzz
}  // namespace spvtools
