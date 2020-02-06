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

#include "source/fuzz/transformation_load.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationLoad::TransformationLoad(
    const spvtools::fuzz::protobufs::TransformationLoad& message)
    : message_(message) {}

TransformationLoad::TransformationLoad(
    uint32_t fresh_id, uint32_t pointer_id,
    const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  message_.set_fresh_id(fresh_id);
  message_.set_pointer_id(pointer_id);
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
}

bool TransformationLoad::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  if (!fuzzerutil::IsFreshId(context, message_.fresh_id())) {
    return false;
  }
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
      // Do not load from a null or undefined pointer.
      return false;
    default:
      break;
  }

  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), context);
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, insert_before)) {
    return false;
  }

  return fuzzerutil::IdsIsAvailableBeforeInstruction(context, insert_before,
                                                     message_.pointer_id());
}

void TransformationLoad::Apply(opt::IRContext* context,
                               spvtools::fuzz::FactManager* /*unused*/) const {
  uint32_t result_type = context->get_def_use_mgr()
                             ->GetDef(context->get_def_use_mgr()
                                          ->GetDef(message_.pointer_id())
                                          ->type_id())
                             ->GetSingleWordInOperand(1);
  fuzzerutil::UpdateModuleIdBound(context, message_.fresh_id());
  FindInstruction(message_.instruction_to_insert_before(), context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          context, SpvOpLoad, result_type, message_.fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.pointer_id()}}})));
  context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationLoad::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_load() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
