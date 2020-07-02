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

#include "source/fuzz/transformation_add_synonym.h"

#include <utility>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/instruction_message.h"

namespace spvtools {
namespace fuzz {

TransformationAddSynonym::TransformationAddSynonym(
    protobufs::TransformationAddSynonym message)
    : message_(std::move(message)) {}

TransformationAddSynonym::TransformationAddSynonym(
    uint32_t result_id,
    protobufs::TransformationAddSynonym::SynonymType synonym_type,
    uint32_t synonym_fresh_id,
    const protobufs::InstructionDescriptor& insert_before) {
  message_.set_result_id(result_id);
  message_.set_synonym_type(synonym_type);
  message_.set_synonym_fresh_id(synonym_fresh_id);
  *message_.mutable_insert_before() = insert_before;
}

bool TransformationAddSynonym::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  assert(protobufs::TransformationAddSynonym::SynonymType_IsValid(
             message_.synonym_type()) &&
         "Synonym type is invalid");

  // Check that |message_.result_id| is valid.
  auto* synonym = ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  if (!synonym) {
    return false;
  }

  // Check that we can apply |synonym_type| to |result_id|.
  if (!IsInstructionValid(ir_context, synonym, message_.synonym_type())) {
    return false;
  }

  // Check that |insert_before| is valid.
  auto* insert_before_inst =
      FindInstruction(message_.insert_before(), ir_context);
  if (!insert_before_inst) {
    return false;
  }

  // Check that we can insert |message._synonymous_instruction| before
  // |message_.insert_before| instruction. We use OpIAdd to represent some
  // instruction that can produce a synonym.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpIAdd,
                                                    insert_before_inst)) {
    return false;
  }

  // Domination rules must be satisfied.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, insert_before_inst, message_.result_id())) {
    return false;
  }

  // A constant instruction must be present in the module if required.
  if (IsAdditionalConstantRequired(message_.synonym_type()) &&
      MaybeGetConstantId(ir_context) == 0) {
    return false;
  }

  // |synonym_fresh_id| must be fresh.
  return fuzzerutil::IsFreshId(ir_context, message_.synonym_fresh_id());
}

void TransformationAddSynonym::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  FindInstruction(message_.insert_before(), ir_context)
      ->InsertBefore(MakeSynonymousInstruction(ir_context));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.synonym_fresh_id());

  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);

  transformation_context->GetFactManager()->AddFactDataSynonym(
      MakeDataDescriptor(message_.result_id(), {}),
      MakeDataDescriptor(message_.synonym_fresh_id(), {}), ir_context);
}

protobufs::Transformation TransformationAddSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_synonym() = message_;
  return result;
}

bool TransformationAddSynonym::IsInstructionValid(
    opt::IRContext* ir_context, opt::Instruction* inst,
    protobufs::TransformationAddSynonym::SynonymType synonym_type) {
  // Instruction must have a result id, type id. We skip OpUndef and
  // OpConstantNull.
  if (!inst || !inst->result_id() || !inst->type_id() ||
      inst->opcode() == SpvOpUndef || inst->opcode() == SpvOpConstantNull) {
    return false;
  }

  switch (synonym_type) {
    case protobufs::TransformationAddSynonym::ADD_ZERO:
    case protobufs::TransformationAddSynonym::SUB_ZERO:
    case protobufs::TransformationAddSynonym::MUL_ONE: {
      // The instruction must be either scalar or vector of integers or floats.
      const auto* type = ir_context->get_type_mgr()->GetType(inst->type_id());
      assert(type && "Instruction's result id is invalid");

      if (const auto* vector = type->AsVector()) {
        return vector->element_type()->AsInteger() ||
               vector->element_type()->AsFloat();
      }

      return type->AsInteger() || type->AsFloat();
    }
    case protobufs::TransformationAddSynonym::COPY_OBJECT:
      // Copy object has no constraints on the type of the operand.
      return true;
    case protobufs::TransformationAddSynonym::LOGICAL_AND:
    case protobufs::TransformationAddSynonym::LOGICAL_OR: {
      // The instruction must be either a scalar or a vector of booleans.
      const auto* type = ir_context->get_type_mgr()->GetType(inst->type_id());
      assert(type && "Instruction's result id is invalid");
      return (type->AsVector() && type->AsVector()->element_type()->AsBool()) ||
             type->AsBool();
    }
    default:
      assert(false && "Synonym type is not supported");
      return false;
  }
}

std::unique_ptr<opt::Instruction>
TransformationAddSynonym::MakeSynonymousInstruction(
    opt::IRContext* ir_context) const {
  auto synonym_type_id =
      fuzzerutil::GetTypeId(ir_context, message_.result_id());
  assert(synonym_type_id && "Synonym has invalid type id");

  switch (message_.synonym_type()) {
    case protobufs::TransformationAddSynonym::SUB_ZERO:
    case protobufs::TransformationAddSynonym::MUL_ONE:
    case protobufs::TransformationAddSynonym::ADD_ZERO: {
      const auto* synonym_type =
          ir_context->get_type_mgr()->GetType(synonym_type_id);
      assert(synonym_type && "Synonym has invalid type");

      // Compute instruction's opcode based on the type of the operand.
      // We have already checked that the operand is either a scalar or a vector
      // of either integers or floats.
      auto is_integral =
          (synonym_type->AsVector() &&
           synonym_type->AsVector()->element_type()->AsInteger()) ||
          synonym_type->AsInteger();
      SpvOp opcode;
      switch (message_.synonym_type()) {
        case protobufs::TransformationAddSynonym::SUB_ZERO:
          opcode = is_integral ? SpvOpISub : SpvOpFSub;
          break;
        case protobufs::TransformationAddSynonym::MUL_ONE:
          opcode = is_integral ? SpvOpIMul : SpvOpFMul;
          break;
        case protobufs::TransformationAddSynonym::ADD_ZERO:
          opcode = is_integral ? SpvOpIAdd : SpvOpFAdd;
          break;
        default:
          assert(false && "Unreachable");
      }

      return MakeUnique<opt::Instruction>(
          ir_context, opcode, synonym_type_id, message_.synonym_fresh_id(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {message_.result_id()}},
              {SPV_OPERAND_TYPE_ID, {MaybeGetConstantId(ir_context)}}});
    }
    case protobufs::TransformationAddSynonym::COPY_OBJECT:
      return MakeUnique<opt::Instruction>(
          ir_context, SpvOpCopyObject, synonym_type_id,
          message_.synonym_fresh_id(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {message_.result_id()}}});
    case protobufs::TransformationAddSynonym::LOGICAL_OR:
    case protobufs::TransformationAddSynonym::LOGICAL_AND: {
      auto opcode = message_.synonym_type() ==
                            protobufs::TransformationAddSynonym::LOGICAL_OR
                        ? SpvOpLogicalOr
                        : SpvOpLogicalAnd;
      return MakeUnique<opt::Instruction>(
          ir_context, opcode, synonym_type_id, message_.synonym_fresh_id(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {message_.result_id()}},
              {SPV_OPERAND_TYPE_ID, {MaybeGetConstantId(ir_context)}}});
    }
    default:
      assert(false && "Unhandled synonym type");
      return nullptr;
  }
}

uint32_t TransformationAddSynonym::MaybeGetConstantId(
    opt::IRContext* ir_context) const {
  assert(IsAdditionalConstantRequired(message_.synonym_type()) &&
         "Synonym type doesn't require an additional constant");

  auto synonym_type_id =
      fuzzerutil::GetTypeId(ir_context, message_.result_id());
  assert(synonym_type_id && "Synonym has invalid type id");

  // TODO():
  //  fuzzerutil::MaybeGet* functions will become available when the PR is
  //  merged.
  switch (message_.synonym_type()) {
    case protobufs::TransformationAddSynonym::ADD_ZERO:
    case protobufs::TransformationAddSynonym::SUB_ZERO:
    case protobufs::TransformationAddSynonym::LOGICAL_OR:
      return fuzzerutil::MaybeGetZeroConstant(ir_context, synonym_type_id);
    case protobufs::TransformationAddSynonym::MUL_ONE:
    case protobufs::TransformationAddSynonym::LOGICAL_AND: {
      auto synonym_type = ir_context->get_type_mgr()->GetType(synonym_type_id);
      assert(synonym_type && "Synonym has invalid type");

      if (const auto* vector = synonym_type->AsVector()) {
        auto element_type_id =
            ir_context->get_type_mgr()->GetId(vector->element_type());
        assert(element_type_id && "Vector's element type is invalid");

        auto scalar_one_id = fuzzerutil::MaybeGetScalarConstant(
            ir_context, {1}, element_type_id);
        if (scalar_one_id == 0) {
          return 0;
        }

        return fuzzerutil::MaybeGetCompositeConstant(
            ir_context,
            std::vector<uint32_t>(vector->element_count(), scalar_one_id),
            synonym_type_id);
      } else {
        return fuzzerutil::MaybeGetScalarConstant(ir_context, {1},
                                                  synonym_type_id);
      }
    }
    default:
      // The assertion at the beginning of the function will fail in the debug
      // mode.
      return 0;
  }
}

bool TransformationAddSynonym::IsAdditionalConstantRequired(
    protobufs::TransformationAddSynonym::SynonymType synonym_type) {
  switch (synonym_type) {
    case protobufs::TransformationAddSynonym::ADD_ZERO:
    case protobufs::TransformationAddSynonym::SUB_ZERO:
    case protobufs::TransformationAddSynonym::LOGICAL_OR:
    case protobufs::TransformationAddSynonym::MUL_ONE:
    case protobufs::TransformationAddSynonym::LOGICAL_AND:
      return true;
    default:
      return false;
  }
}

}  // namespace fuzz
}  // namespace spvtools
