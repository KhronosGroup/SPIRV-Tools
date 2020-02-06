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

#include "source/fuzz/transformation_store.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationStore::TransformationStore(
    const spvtools::fuzz::protobufs::TransformationStore& message)
    : message_(message) {}

TransformationStore::TransformationStore(
    uint32_t pointer_id, uint32_t value_id,
    const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  message_.set_pointer_id(pointer_id);
  message_.set_value_id(value_id);
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
}

bool TransformationStore::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& fact_manager) const {
  auto pointer = context->get_def_use_mgr()->GetDef(message_.pointer_id());
  if (!pointer || !pointer->type_id()) {
    return false;
  }
  auto pointer_type = context->get_def_use_mgr()->GetDef(pointer->type_id());
  assert(pointer_type && "Type id must be defined.");
  if (pointer_type->opcode() != SpvOpTypePointer) {
    return false;
  }
  switch (pointer->opcode()) {
    case SpvOpConstantNull:
    case SpvOpUndef:
      // Do not store to a null or undefined pointer.
      return false;
    default:
      break;
  }

  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), context);
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpStore,
                                                    insert_before)) {
    return false;
  }

  if (!fact_manager.BlockIsDead(
          context->get_instr_block(insert_before)->id()) &&
      !fact_manager.PointeeValueIsIrrelevant(message_.pointer_id())) {
    return false;
  }

  if (pointer_type->GetSingleWordInOperand(1) !=
      context->get_def_use_mgr()->GetDef(message_.value_id())->type_id()) {
    return false;
  }

  if (pointer_type->GetSingleWordInOperand(0) == SpvStorageClassInput) {
    return false;
  }

  return fuzzerutil::IdsIsAvailableBeforeInstruction(context, insert_before,
                                                     message_.pointer_id()) &&
         fuzzerutil::IdsIsAvailableBeforeInstruction(context, insert_before,
                                                     message_.value_id());
}

void TransformationStore::Apply(opt::IRContext* context,
                                spvtools::fuzz::FactManager* /*unused*/) const {
  FindInstruction(message_.instruction_to_insert_before(), context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.value_id()}}})));
  context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationStore::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_store() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
