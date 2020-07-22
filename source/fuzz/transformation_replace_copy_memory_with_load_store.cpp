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
  if ((!copy_memory_instruction) ||
      (copy_memory_instruction->opcode() != SpvOpCopyMemory)) {
    return false;
  }
  // It must be valid to insert the OpStore and OpLoad instruction before it.
  return (fuzzerutil::CanInsertOpcodeBeforeInstruction(
              SpvOpStore, copy_memory_instruction) &&
          fuzzerutil::CanInsertOpcodeBeforeInstruction(
              SpvOpLoad, copy_memory_instruction));
}

void TransformationReplaceCopyMemoryWithLoadStore::Apply(
    opt::IRContext* ir_context, TransformationContext*) const {
  auto copy_memory_instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  auto operand_target = copy_memory_instruction->GetSingleWordOperand(0);
  auto operand_source = copy_memory_instruction->GetSingleWordOperand(1);
  std::cout << operand_target << std::endl;
  std::cout << operand_source << std::endl;
}
/*
auto value_instruction =
    ir_context->get_def_use_mgr()->GetDef(message_.value_id());

// A pointer type instruction pointing to the value type must be defined.
auto pointer_type_id = fuzzerutil::MaybeGetPointerType(
    ir_context, value_instruction->type_id(),
    static_cast<SpvStorageClass>(message_.variable_storage_class()));
assert(pointer_type_id && "The required pointer type must be available.");

// Adds whether a global or local variable.
if (message_.variable_storage_class() == SpvStorageClassPrivate) {
  fuzzerutil::AddGlobalVariable(ir_context, message_.variable_id(),
                                pointer_type_id, SpvStorageClassPrivate,
                                message_.initializer_id());
} else {
  auto function_id = ir_context
                         ->get_instr_block(FindInstruction(
                             message_.instruction_descriptor(), ir_context))
                         ->GetParent()
                         ->result_id();
  fuzzerutil::AddLocalVariable(ir_context, message_.variable_id(),
                               pointer_type_id, function_id,
                               message_.initializer_id());
}

// First, insert the OpLoad instruction before |instruction_descriptor| and
// then insert the OpStore instruction before the OpLoad instruction.
fuzzerutil::UpdateModuleIdBound(ir_context, message_.value_synonym_id());
FindInstruction(message_.instruction_descriptor(), ir_context)
    ->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpLoad, value_instruction->type_id(),
        message_.value_synonym_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.variable_id()}}})))
    ->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpStore, 0, 0,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.variable_id()}},
             {SPV_OPERAND_TYPE_ID, {message_.value_id()}}})));

ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

// Adds the fact that |message_.value_synonym_id|
// and |message_.value_id| are synonymous.
transformation_context->GetFactManager()->AddFactDataSynonym(
    MakeDataDescriptor(message_.value_synonym_id(), {}),
    MakeDataDescriptor(message_.value_id(), {}), ir_context);
    */

protobufs::Transformation
TransformationReplaceCopyMemoryWithLoadStore::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_copy_memory_with_load_store() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
