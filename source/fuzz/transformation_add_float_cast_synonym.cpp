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

#include <utility>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_float_cast_synonym.h"

namespace spvtools {
namespace fuzz {

TransformationAddFloatCastSynonym::TransformationAddFloatCastSynonym(
    protobufs::TransformationAddFloatCastSynonym message)
    : message_(std::move(message)) {}

TransformationAddFloatCastSynonym::TransformationAddFloatCastSynonym(
    uint32_t synonym_id, uint32_t to_float_fresh_id, uint32_t to_int_fresh_id,
    uint32_t float_type_id) {
  message_.set_synonym_id(synonym_id);
  message_.set_to_float_fresh_id(to_float_fresh_id);
  message_.set_to_int_fresh_id(to_int_fresh_id);
  message_.set_float_type_id(float_type_id);
}

bool TransformationAddFloatCastSynonym::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that |message_.synonym_id| is valid.
  auto* synonym = ir_context->get_def_use_mgr()->GetDef(message_.synonym_id());
  if (!synonym) {
    return false;
  }

  const auto* synonym_type =
      ir_context->get_type_mgr()->GetType(synonym->type_id());
  assert(synonym_type && "Instruction must have a valid type");

  // |message_.synonym_id| must have either scalar or vector type.
  if (!synonym_type->AsInteger() && !synonym_type->AsVector()) {
    return false;
  }

  // Vector's components must be integers.
  if (synonym_type->AsVector() &&
      !synonym_type->AsVector()->element_type()->AsInteger()) {
    return false;
  }

  auto* block = ir_context->get_instr_block(synonym);
  assert(block && "Instruction must have a basic block");

  auto iter = fuzzerutil::GetIteratorForInstruction(block, synonym);
  ++iter;
  assert(iter != block->end() &&
         "Cannot create a synonym to the last instruction in the block");

  // Check that we can insert conversions after |message_.synonym_id|.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpConvertUToF, iter)) {
    return false;
  }

  // Check that result ids for conversion instructions are fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.to_float_fresh_id()) ||
      !fuzzerutil::IsFreshId(ir_context, message_.to_int_fresh_id())) {
    return false;
  }

  const auto* float_type =
      ir_context->get_type_mgr()->GetType(message_.float_type_id());

  // Check that |message_.float_type_id| is valid.
  if (!float_type) {
    return false;
  }

  // Check that |float_type| corresponds to |synonym_type|.
  if (const auto* integer = synonym_type->AsInteger()) {
    // If |synonym_type| is scalar, |float_type| must be scalar too and have the
    // same width.
    if (!float_type->AsFloat() ||
        float_type->AsFloat()->width() != integer->width()) {
      return false;
    }
  } else {
    // Otherwise, |float_type| must be...
    assert(synonym_type->AsVector() &&
           "Only integers or vectors of integers are supported");
    const auto* synonym_vector = synonym_type->AsVector();
    const auto* float_vector = float_type->AsVector();
    // ...a vector...
    if (!float_vector) {
      return false;
    }

    // ...of floats with the same size as |synonym_type| and...
    if (!float_vector->element_type()->AsFloat() ||
        float_vector->element_count() != synonym_vector->element_count()) {
      return false;
    }

    // ...its elements must have the same width.
    if (float_vector->element_type()->AsFloat()->width() !=
        synonym_vector->element_type()->AsInteger()->width()) {
      return false;
    }
  }

  return true;
}

void TransformationAddFloatCastSynonym::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  const auto* synonym_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.synonym_id());
  assert(synonym_inst);

  const auto* synonym_type =
      ir_context->get_type_mgr()->GetType(synonym_inst->type_id());
  assert(synonym_type);

  // Select an instruction to cast based on whether the integers are signed or
  // not.
  SpvOp convert_to_float_opcode, convert_to_int_opcode;
  if (const auto* vector = synonym_type->AsVector()) {
    convert_to_float_opcode = vector->element_type()->AsInteger()->IsSigned()
                                  ? SpvOpConvertSToF
                                  : SpvOpConvertUToF;
    convert_to_int_opcode = vector->element_type()->AsInteger()->IsSigned()
                                ? SpvOpConvertFToS
                                : SpvOpConvertFToU;
  } else {
    assert(synonym_type->AsInteger() &&
           "Instruction must have either a vector or an integer type");

    convert_to_float_opcode = synonym_type->AsInteger()->IsSigned()
                                  ? SpvOpConvertSToF
                                  : SpvOpConvertUToF;
    convert_to_int_opcode = synonym_type->AsInteger()->IsSigned()
                                ? SpvOpConvertFToS
                                : SpvOpConvertFToU;
  }

  auto iter = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(synonym_inst->result_id()), synonym_inst);
  ++iter;
  assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(convert_to_float_opcode,
                                                      iter) &&
         "Cannot insert conversion instruction after the synonym instruction");

  iter.InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, convert_to_float_opcode, message_.float_type_id(),
      message_.to_float_fresh_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.synonym_id()}}}));

  iter.InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, convert_to_int_opcode, synonym_inst->type_id(),
      message_.to_int_fresh_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.to_float_fresh_id()}}}));

  fuzzerutil::UpdateModuleIdBound(
      ir_context,
      std::max(message_.to_float_fresh_id(), message_.to_int_fresh_id()));

  transformation_context->GetFactManager()->AddFactDataSynonym(
      MakeDataDescriptor(message_.synonym_id(), {}),
      MakeDataDescriptor(message_.to_int_fresh_id(), {}), ir_context);

  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationAddFloatCastSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_float_cast_synonym() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
