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

#include "source/fuzz/transformation_push_id_through_variable.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_store.h"
#include "source/fuzz/transformation_load.h"

namespace spvtools {
namespace fuzz {

TransformationPushIdThroughVariable::
    TransformationPushIdThroughVariable(
        const spvtools::fuzz::protobufs::
            TransformationPushIdThroughVariable& message)
    : message_(message) {}

TransformationPushIdThroughVariable::
    TransformationPushIdThroughVariable(
        uint32_t fresh_id,
        uint32_t pointer_id, uint32_t value_id,
        const protobufs::InstructionDescriptor& instruction_descriptor) {
  message_.set_fresh_id(fresh_id);
  message_.set_pointer_id(pointer_id);
  message_.set_value_id(value_id);
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationPushIdThroughVariable::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/
    ) const {
  // The pointer must exist and have a type.
  auto pointer = ir_context->get_def_use_mgr()->GetDef(message_.pointer_id());
  if (!pointer || !pointer->type_id()) {
    return false;
  }

  // The pointer type must indeed be a pointer.
  auto pointer_type = ir_context->get_def_use_mgr()->GetDef(pointer->type_id());
  assert(pointer_type && "Type id must be defined.");
  if (pointer_type->opcode() != SpvOpTypePointer) {
    return false;
  }

  // The pointer must not be read only.
  if (pointer->IsReadOnlyPointer()) {
    return false;
  }

  // Determine which instruction we should be inserting before.
  auto insert_before = FindInstruction(message_.instruction_descriptor(), ir_context);
  // It must exist, ...
  if (!insert_before) {
    return false;
  }
  // ... and it must be legitimate to insert a store before it.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpStore, insert_before) ||
      !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, insert_before)) {
    return false;
  }

  // The value being stored needs to exist and have a type.
  auto value = ir_context->get_def_use_mgr()->GetDef(message_.value_id());
  if (!value || !value->type_id()) {
    return false;
  }

  // The type of the value must match the pointee type.
  if (pointer_type->GetSingleWordInOperand(1) != value->type_id()) {
    return false;
  }

  // The pointer needs to be available at the insertion point.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                  message_.pointer_id())) {
    return false;
  }

  // The value needs to be available at the insertion point.
  return fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                    message_.value_id());
}

void TransformationPushIdThroughVariable::Apply(opt::IRContext* ir_context,
                                                TransformationContext* transformation_context) const {
  TransformationStore(message_.pointer_id(),
                      message_.value_id(),
                      message_.instruction_descriptor()).Apply(ir_context, transformation_context);

  TransformationLoad(message_.fresh_id(),
                     message_.pointer_id(),
                     message_.instruction_descriptor()).Apply(ir_context, transformation_context);

  transformation_context
    ->GetFactManager()
    ->AddFactDataSynonym(MakeDataDescriptor(message_.fresh_id(), {}),
                         MakeDataDescriptor(message_.value_id(), {}),
                         ir_context);
}

protobufs::Transformation
TransformationPushIdThroughVariable::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_push_id_through_variable() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
