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

#include "source/fuzz/transformation_replace_copy_memory_with_load_store.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceCopyMemoryWithLoadStore::
    TransformationReplaceCopyMemoryWithLoadStore(
        const spvtools::fuzz::protobufs::
            TransformationReplaceCopyMemoryWithLoadStore& message)
    : message_(message) {}

TransformationReplaceCopyMemoryWithLoadStore::
    TransformationReplaceCopyMemoryWithLoadStore(
        uint32_t source_value,
        const protobufs::InstructionDescriptor& instruction_descriptor) {
  message_.set_source_value(source_value);
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationReplaceCopyMemoryWithLoadStore::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.source_value| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.source_value())) return false;

  // The instruction to insert before must be defined and of opcode
  // OpCopyMemory.
  auto copy_memory_instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  if (!copy_memory_instruction ||
      copy_memory_instruction->opcode() != SpvOpCopyMemory) {
    return false;
  }
  // It must be valid to insert the OpStore and OpLoad instruction before it.
  return (fuzzerutil::CanInsertOpcodeBeforeInstruction(
              SpvOpStore, copy_memory_instruction) &&
          fuzzerutil::CanInsertOpcodeBeforeInstruction(
              SpvOpLoad, copy_memory_instruction));
}

void TransformationReplaceCopyMemoryWithLoadStore::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto copy_memory_instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  auto target = ir_context->get_def_use_mgr()->GetDef(
      copy_memory_instruction->GetSingleWordInOperand(0));
  auto source = ir_context->get_def_use_mgr()->GetDef(
      copy_memory_instruction->GetSingleWordInOperand(1));
  auto target_type_opcode =
      ir_context->get_def_use_mgr()->GetDef(target->type_id())->opcode();
  auto source_type_opcode =
      ir_context->get_def_use_mgr()->GetDef(source->type_id())->opcode();
  assert(target_type_opcode == SpvOpTypePointer &&
         source_type_opcode == SpvOpTypePointer &&
         "Operands must be of type OpTypePointer");
  uint32_t target_pointee_type = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, target->type_id());
  uint32_t source_pointee_type = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, source->type_id());
  assert(target_pointee_type == source_pointee_type &&
         "Operands must have the same type to which they point to.");

  // First, insert the OpStore instruction before the OpCopyMemory instruction
  // and then insert the OpLoad instruction before the OpStore instruction.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.source_value());
  FindInstruction(message_.instruction_descriptor(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {target->result_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.source_value()}}})))
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpLoad, target_pointee_type, message_.source_value(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {source->result_id()}}})));

  // Remove the CopyObject instruction.
  ir_context->KillInst(copy_memory_instruction);

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation
TransformationReplaceCopyMemoryWithLoadStore::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_copy_memory_with_load_store() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
