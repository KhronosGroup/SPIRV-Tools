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
  if (load_instruction->result_id() !=
      store_instruction->GetSingleWordOperand(1)) {
    return false;
  }

  // Get storage class of the variable pointed by the source operand in OpLoad
  opt::Instruction* source_id = ir_context->get_def_use_mgr()->GetDef(
      load_instruction->GetSingleWordOperand(2));
  SpvStorageClass storage_class = fuzzerutil::GetStorageClassFromPointerType(
      ir_context, source_id->type_id());
  // Iterate over all instructions between |load_instruction| and
  // |store_instruction|.
  for (auto it = load_instruction; it != store_instruction;
       it = it->NextNode()) {
    //|load_instruction| and |store_instruction| are not in the same block.
    if (it == nullptr) {
      return false;
    }
    // We need to make sure that the value pointed by source of OpLoad
    // hasn't changed by the time we see matching OpStore instruction.
    if (IsInterferingInstruction(it, storage_class)) {
      return false;
    }
  }
  return true;
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

  // Get the ids of the source operand of the OpLoad and the target operand of
  // the OpStore.
  auto source = ir_context->get_def_use_mgr()->GetDef(
      load_instruction->GetSingleWordOperand(2));
  auto target = ir_context->get_def_use_mgr()->GetDef(
      store_instruction->GetSingleWordInOperand(0));

  // Insert the OpCopyMemory instruction before the OpStore instruction.
  FindInstruction(message_.store_instruction_descriptor(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpCopyMemory, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {target->result_id()}},
               {SPV_OPERAND_TYPE_ID, {source->result_id()}}})));

  // Remove the OpStore instruction.
  ir_context->KillInst(store_instruction);

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

bool TransformationReplaceLoadStoreWithCopyMemory::IsInterferingInstruction(
    opt::Instruction* inst, SpvStorageClass storage_class) {
  if (inst->opcode() == SpvOpCopyMemory || inst->opcode() == SpvOpStore ||
      inst->opcode() == SpvOpCopyMemorySized || inst->IsAtomicOp()) {
    return true;
  } else if (inst->opcode() == SpvOpMemoryBarrier ||
             inst->opcode() == SpvOpMemoryNamedBarrier) {
    switch (storage_class) {
      // These storage classes of the source variable of the OpLoad instruction
      // don't invalidate it.
      case SpvStorageClassUniformConstant:
      case SpvStorageClassInput:
      case SpvStorageClassUniform:
      case SpvStorageClassPrivate:
      case SpvStorageClassFunction:
        return false;
      default:
        return true;
    }
  }
  return false;
}

protobufs::Transformation
TransformationReplaceLoadStoreWithCopyMemory::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_load_store_with_copy_memory() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
