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

#include "source/fuzz/transformation_mutate_pointer.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationMutatePointer::TransformationMutatePointer(
    const protobufs::TransformationMutatePointer& message)
    : message_(message) {}

TransformationMutatePointer::TransformationMutatePointer(
    uint32_t pointer_id, uint32_t fresh_id,
    const protobufs::InstructionDescriptor& insert_before) {
  message_.set_pointer_id(pointer_id);
  message_.set_fresh_id(fresh_id);
  *message_.mutable_insert_before() = insert_before;
}

bool TransformationMutatePointer::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Check that |fresh_id| is fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  auto* insert_before_inst =
      FindInstruction(message_.insert_before(), ir_context);

  // Check that |insert_before| is a valid instruction descriptor.
  if (!insert_before_inst) {
    return false;
  }

  // Check that it is possible to insert OpLoad and OpStore before
  // |insert_before_inst|. We are only using OpLoad here since the result does
  // not depend on the opcode.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad,
                                                    insert_before_inst)) {
    return false;
  }

  const auto* pointer_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.pointer_id());

  // Check that |pointer_id| is a result id of a valid pointer instruction.
  if (!pointer_inst || !IsValidPointerInstruction(ir_context, *pointer_inst)) {
    return false;
  }

  // Check that the module contains an irrelevant constant that will be used to
  // mutate |pointer_inst|.
  auto constant_id = fuzzerutil::MaybeGetZeroConstant(
      ir_context, transformation_context,
      fuzzerutil::GetPointeeTypeIdFromPointerType(ir_context,
                                                  pointer_inst->type_id()),
      true);
  if (!constant_id) {
    return false;
  }

  assert(fuzzerutil::IdIsAvailableBeforeInstruction(
             ir_context, insert_before_inst, constant_id) &&
         "Global constant instruction is not available before "
         "|insert_before_inst|");

  // Check that |pointer_inst| is available before |insert_before_inst|.
  return fuzzerutil::IdIsAvailableBeforeInstruction(
      ir_context, insert_before_inst, pointer_inst->result_id());
}

void TransformationMutatePointer::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto* insert_before_inst =
      FindInstruction(message_.insert_before(), ir_context);
  assert(insert_before_inst && "|insert_before| descriptor is invalid");

  auto pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, fuzzerutil::GetTypeId(ir_context, message_.pointer_id()));

  // Back up the original value.
  insert_before_inst->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, SpvOpLoad, pointee_type_id, message_.fresh_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.pointer_id()}}}));

  // Insert a new value.
  insert_before_inst->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, SpvOpStore, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
          {SPV_OPERAND_TYPE_ID,
           {fuzzerutil::MaybeGetZeroConstant(
               ir_context, *transformation_context, pointee_type_id, true)}}}));

  // Restore the original value.
  insert_before_inst->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, SpvOpStore, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
          {SPV_OPERAND_TYPE_ID, {message_.fresh_id()}}}));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Make sure analyses represent the correct state of the module.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationMutatePointer::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_mutate_pointer() = message_;
  return result;
}

bool TransformationMutatePointer::IsValidPointerInstruction(
    opt::IRContext* ir_context, const opt::Instruction& inst) {
  // |inst| must have both result id and type id and it may not cause undefined
  // behaviour.
  if (!inst.result_id() || !inst.type_id() || inst.opcode() == SpvOpUndef ||
      inst.opcode() == SpvOpConstantNull) {
    return false;
  }

  const auto* type = ir_context->get_type_mgr()->GetType(inst.type_id());
  assert(type && "|inst| has invalid type id");

  const auto* pointer_type = type->AsPointer();

  // |inst| must be a pointer.
  if (!pointer_type) {
    return false;
  }

  // |inst| must have a supported storage class.
  if (pointer_type->storage_class() != SpvStorageClassFunction &&
      pointer_type->storage_class() != SpvStorageClassPrivate &&
      pointer_type->storage_class() != SpvStorageClassWorkgroup) {
    return false;
  }

  // |inst|'s pointee must consist of scalars and/or composites.
  return fuzzerutil::CanCreateConstant(*pointer_type->pointee_type());
}

}  // namespace fuzz
}  // namespace spvtools
