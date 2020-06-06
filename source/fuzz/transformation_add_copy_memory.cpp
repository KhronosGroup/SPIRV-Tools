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

#include "source/fuzz/transformation_add_copy_memory.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
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

bool TransformationAddCopyMemory::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that target id is fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.target_id())) {
    return false;
  }

  // Check that instruction descriptor is valid.
  auto* inst = FindInstruction(message_.instruction_descriptor(), ir_context);
  if (!inst) {
    return false;
  }

  // Check that we can insert OpCopyMemory before |instruction_descriptor|.
  auto iter = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(inst), inst);
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCopyMemory, iter)) {
    return false;
  }

  // Check that source instruction exists.
  auto* source_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.source_id());
  if (!source_inst) {
    return false;
  }

  // Check that result type of source instruction exists and can be used with
  // OpCopyMemory.
  const auto* source_type =
      ir_context->get_type_mgr()->GetType(source_inst->type_id());
  if (!source_type || !source_type->AsPointer()) {
    return false;
  }

  if (!CanUsePointeeWithCopyMemory(*source_type->AsPointer()->pointee_type())) {
    return false;
  }

  // OpTypePointer with Private storage class exists.
  if (!fuzzerutil::MaybeGetPointerType(
          ir_context,
          ir_context->get_type_mgr()->GetId(
              source_type->AsPointer()->pointee_type()),
          SpvStorageClassPrivate)) {
    return false;
  }

  // Check that this transformation respects domination rules.
  const auto* source_block = ir_context->get_instr_block(source_inst);
  const auto* target_block = ir_context->get_instr_block(inst);
  assert(source_block && target_block);

  // Check that both source and target instructions are in the same function.
  if (source_block->GetParent() != target_block->GetParent()) {
    return false;
  }

  // Check domination rules.
  return source_inst != inst &&
         ir_context->GetDominatorAnalysis(source_block->GetParent())
             ->Dominates(source_inst, inst);
}

void TransformationAddCopyMemory::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Result id for target variable type.
  // TODO: it would be good to refactor variable creation part into a separate
  //  function (in, say, fuzzerutil). This will reduce boilerplate here, in
  //  TransformationPushIdsThroughVariables and two transformations that create
  //  variables.
  auto type_id = fuzzerutil::MaybeGetPointerType(
      ir_context,
      fuzzerutil::GetPointeeTypeIdFromPointerType(
          ir_context, fuzzerutil::GetTypeId(ir_context, message_.source_id())),
      SpvStorageClassPrivate);

  // Create global target variable.
  opt::Instruction::OperandList variable_operands = {
      {SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassPrivate}}};
  ir_context->AddGlobalValue(MakeUnique<opt::Instruction>(
      ir_context, SpvOpVariable, type_id, message_.target_id(),
      std::move(variable_operands)));

  fuzzerutil::AddVariableIdToEntryPointInterfaces(ir_context,
                                                  message_.target_id());

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.target_id());

  transformation_context->GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
      message_.target_id());

  // Insert OpCopyMemory before |instruction_descriptor|.
  auto* insert_before_inst =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  assert(insert_before_inst);

  auto insert_before_iter = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(insert_before_inst), insert_before_inst);

  opt::Instruction::OperandList copy_operands = {
      {SPV_OPERAND_TYPE_ID, {message_.target_id()}},
      {SPV_OPERAND_TYPE_ID, {message_.source_id()}}};

  insert_before_iter.InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, SpvOpCopyMemory, 0, 0, std::move(copy_operands)));

  // Make sure our changes are analyzed
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationAddCopyMemory::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_copy_memory() = message_;
  return result;
}

bool TransformationAddCopyMemory::CanUsePointeeWithCopyMemory(
    const opt::analysis::Type& type) {
  switch (type.kind()) {
    case opt::analysis::Type::kBool:
    case opt::analysis::Type::kInteger:
    case opt::analysis::Type::kFloat:
    case opt::analysis::Type::kArray:
      return true;
    case opt::analysis::Type::kVector:
      return CanUsePointeeWithCopyMemory(*type.AsVector()->element_type());
    case opt::analysis::Type::kMatrix:
      return CanUsePointeeWithCopyMemory(*type.AsMatrix()->element_type());
    case opt::analysis::Type::kStruct:
      return std::all_of(type.AsStruct()->element_types().begin(),
                         type.AsStruct()->element_types().end(),
                         [](const opt::analysis::Type* element) {
                           return CanUsePointeeWithCopyMemory(*element);
                         });
    case opt::analysis::Type::kPointer:
      return CanUsePointeeWithCopyMemory(*type.AsPointer()->pointee_type());
    default:
      return false;
  }
}

}  // namespace fuzz
}  // namespace spvtools
