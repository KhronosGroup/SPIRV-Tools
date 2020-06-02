// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_copy_memory.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace fuzz {

TransformationAddCopyMemory::TransformationAddCopyMemory(
    const protobufs::TransformationAddCopyMemory& message)
    : message_(message) {}

TransformationAddCopyMemory::TransformationAddCopyMemory(
    const protobufs::InstructionDescriptor& instruction_descriptor,
    uint32_t target_id, uint32_t source_id) {
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
  message_.set_target_id(target_id);
  message_.set_source_id(source_id);
}

bool TransformationAddCopyMemory::IsApplicable(opt::IRContext* ir_context,
    const TransformationContext& /*unused*/) const {
  // Check that instruction descriptor is valid.
  const auto* inst =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  if (!inst) {
    return false;
  }

  auto iter = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(inst->result_id()), inst);
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCopyMemory, iter)) {
    return false;
  }

  // Check that source instruction exists.
  const auto* source_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.source_id());
  if (!source_inst) {
    return false;
  }

  // Check that result type of source instruction exists, OpTypePointer and is not opaque.
  const auto* source_type_inst =
      ir_context->get_def_use_mgr()->GetDef(source_inst->type_id());
  if (!source_type_inst ||
      source_type_inst->opcode() != SpvOpTypePointer ||
      source_type_inst->IsOpaqueType()) {
    return false;
  }

  // Check that target instruction exists.
  const auto* target_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.target_id());
  if (!target_inst) {
    return false;
  }

  // Check that result type of target instruction exists, OpTypePointer and is not opaque.
  const auto* target_type_inst =
      ir_context->get_def_use_mgr()->GetDef(target_inst->type_id());
  if (!target_type_inst ||
      target_type_inst->opcode() != SpvOpTypePointer ||
      target_type_inst->IsOpaqueType()) {
    return false;
  }

  // Check that result types of both source and target instructions point to the same type.
  if (target_type_inst->GetSingleWordInOperand(1) !=
      source_type_inst->GetSingleWordInOperand(1)) {
    return false;
  }

  return true;
}

void TransformationAddCopyMemory::Apply(opt::IRContext* ir_context,
    TransformationContext* /*unused*/) const {
  const auto* insert_before_inst =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  auto insert_before_iter = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(insert_before_inst->result_id()),
      insert_before_inst);

  opt::Instruction::OperandList operands = {
    {SPV_OPERAND_TYPE_RESULT_ID, {message_.target_id()}},
    {SPV_OPERAND_TYPE_RESULT_ID, {message_.source_id()}}
  };

  insert_before_iter.InsertBefore(MakeUnique<opt::Instruction>(
    ir_context, SpvOpCopyMemory, 0, 0, std::move(operands)));

  // Make sure our changes are analyzed
  // TODO: not sure if we need to do this here.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationAddCopyMemory::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_add_copy_memory() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
