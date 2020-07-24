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

#include "transformation_replace_load_store_with_copy_memory.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceLoadStoreWithCopyMemory::
    TransformationReplaceLoadStoreWithCopyMemory(
        const spvtools::fuzz::protobufs::
            TransformationReplaceLoadStoreWithCopyMemory& message)
    : message_(message) {}

TransformationReplaceLoadStoreWithCopyMemory::
    TransformationReplaceLoadStoreWithCopyMemory(
        const protobufs::InstructionDescriptor& load_instruction_descriptor,
        const protobufs::InstructionDescriptor& store_instruction_descriptor) {
  *message_.mutable_load_instruction_descriptor() = load_instruction_descriptor;
  *message_.mutable_store_instruction_descriptor() =
      store_instruction_descriptor;
}
bool TransformationReplaceLoadStoreWithCopyMemory::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The OpLoad instruction must be defined.
  auto load_instruction =
      FindInstruction(message_.load_instruction_descriptor(), ir_context);
  if (!load_instruction || load_instruction->opcode() != SpvOpLoad) {
    return false;
  }
  // The OpStore instruction must be defined.
  auto store_instruction =
      FindInstruction(message_.store_instruction_descriptor(), ir_context);
  if (!store_instruction || store_instruction->opcode() != SpvOpStore) {
    return false;
  }

  // Intermediate values of the OpLoad and the OpStore must match.
  return load_instruction->result_id() ==
         store_instruction->GetSingleWordOperand(1);
}

void TransformationReplaceLoadStoreWithCopyMemory::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // OpLoad and OpStore instructions must be defined.
  auto load_instruction =
      FindInstruction(message_.load_instruction_descriptor(), ir_context);
  assert(load_instruction && load_instruction->opcode() == SpvOpLoad &&
         "The required OpLoad instruction must be defined.");
  auto store_instruction =
      FindInstruction(message_.store_instruction_descriptor(), ir_context);
  assert(store_instruction && store_instruction->opcode() == SpvOpStore &&
         "The required OpStore instruction must be defined.");

  // Intermediate values of the OpLoad and the OpStore must match.
  assert(load_instruction->result_id() ==
             store_instruction->GetSingleWordOperand(1) &&
         "OpLoad and OpStore must refer to the same value.");

  // Coherence check: Both operands must be pointers.

  // Get types of ids used as the source of OpLoad and the target of OpStore
  auto source = ir_context->get_def_use_mgr()->GetDef(
      load_instruction->GetSingleWordOperand(2));
  auto target = ir_context->get_def_use_mgr()->GetDef(
      store_instruction->GetSingleWordInOperand(0));
  auto source_type_opcode =
      ir_context->get_def_use_mgr()->GetDef(source->type_id())->opcode();
  auto target_type_opcode =
      ir_context->get_def_use_mgr()->GetDef(target->type_id())->opcode();

  // Keep release-mode compilers happy. (No unused variables.)
  (void)target;
  (void)source;
  (void)target_type_opcode;
  (void)source_type_opcode;

  assert(target_type_opcode == SpvOpTypePointer &&
         source_type_opcode == SpvOpTypePointer &&
         "The target of OpStore and the source of OpLoad must be of type "
         "OpTypePointer");

  // Coherence check: |source| and |target| must point to the same type.
  uint32_t target_pointee_type = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, target->type_id());
  uint32_t source_pointee_type = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, source->type_id());

  (void)target_pointee_type;
  (void)source_pointee_type;

  assert(target_pointee_type == source_pointee_type &&
         "The target of OpStore and the source of OpLoad must point to the "
         "same type.");

  // Coherence check: First operand of the OpLoad must match the type to which
  // the source of OpLoad points to.

  assert(load_instruction->GetSingleWordOperand(0) == source_pointee_type &&
         "First operand of the OpLoad must match the type to which the source "
         "of OpLoad points to.");
  // First, insert the OpCopyMemory instruction before the OpStore instruction
  FindInstruction(message_.store_instruction_descriptor(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpCopyMemory, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {target->result_id()}},
               {SPV_OPERAND_TYPE_ID, {source->result_id()}}})));

  // Remove the OpCopyMemory instruction.
  ir_context->KillInst(store_instruction);

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation
TransformationReplaceLoadStoreWithCopyMemory::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_load_store_with_copy_memory() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools