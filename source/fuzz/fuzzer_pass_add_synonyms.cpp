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

#include "source/fuzz/fuzzer_pass_add_synonyms.h"

#include <functional>
#include <unordered_map>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/instruction_message.h"
#include "source/fuzz/transformation_add_float_cast_synonym.h"
#include "source/fuzz/transformation_add_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddSynonyms::FuzzerPassAddSynonyms(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddSynonyms::~FuzzerPassAddSynonyms() = default;

void FuzzerPassAddSynonyms::Apply() {
  using TransformationsMap = std::unordered_map<
      SpvOp, std::vector<std::function<void(const opt::Instruction*)>>>;

  const TransformationsMap kTransformationMap = {
      {SpvOpTypeInt,
       {[this](const opt::Instruction* inst) {
          CreateScalarMultiplicationSynonym(inst, SpvOpIMul);
        },
        [this](const opt::Instruction* inst) {
          CreateScalarAdditionSynonym(inst, SpvOpIAdd);
        },
        [this](const opt::Instruction* inst) { CreateCastSynonym(inst); }}},
      {SpvOpTypeFloat,
       {[this](const opt::Instruction* inst) {
          CreateScalarMultiplicationSynonym(inst, SpvOpFMul);
        },
        [this](const opt::Instruction* inst) {
          CreateScalarAdditionSynonym(inst, SpvOpFAdd);
        }}},
      {SpvOpTypeBool,
       {[this](const opt::Instruction* inst) {
          CreateScalarMultiplicationSynonym(inst, SpvOpLogicalAnd);
        },
        [this](const opt::Instruction* inst) {
          CreateScalarAdditionSynonym(inst, SpvOpLogicalOr);
        }}},
      {SpvOpTypeVector,
       {[this](const opt::Instruction* inst) {
          CreateVectorMultiplicationSynonym(inst);
        },
        [this](const opt::Instruction* inst) {
          CreateVectorAdditionSynonym(inst);
        },
        [this](const opt::Instruction* inst) { CreateCastSynonym(inst); }}}};

  for (const auto* type_inst : GetIRContext()->module()->GetTypes()) {
    // Check that |type_inst| is supported.
    if (kTransformationMap.find(type_inst->opcode()) ==
        kTransformationMap.end()) {
      continue;
    }

    // Collect all instructions that will be used to create synonym for. We
    // store these in a separate vector to make sure we don't invalidate
    // iterators by inserting new instructions into the module.
    std::vector<const opt::Instruction*> candidate_instructions;
    GetIRContext()->get_def_use_mgr()->ForEachUser(
        type_inst,
        [this, type_inst, &candidate_instructions](opt::Instruction* inst) {
          if (!fuzzerutil::CanMakeSynonymOf(GetIRContext(), inst)) {
            return;
          }

          if (!inst->result_id() || !inst->type_id() ||
              inst->type_id() != type_inst->type_id() ||
              inst->opcode() == SpvOpUndef) {
            return;
          }

          if (!GetFuzzerContext()->ChoosePercentage(
                  GetFuzzerContext()->GetChanceOfAddingSynonyms())) {
            return;
          }

          candidate_instructions.push_back(inst);
        });

    // Apply transformations to create synonyms.
    for (const auto* inst : candidate_instructions) {
      const auto& transformations = kTransformationMap.at(inst->opcode());
      transformations[GetFuzzerContext()->RandomIndex(transformations)](inst);
    }
  }
}

void FuzzerPassAddSynonyms::CreateScalarMultiplicationSynonym(
    const opt::Instruction* inst, SpvOp opcode) {
  ApplyTransformation(TransformationAddSynonym(
      inst->result_id(),
      MakeInstructionMessage(
          opcode, inst->type_id(), GetFuzzerContext()->GetFreshId(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {inst->result_id()}},
              {SPV_OPERAND_TYPE_ID,
               {FindOrCreateConstant({1}, inst->type_id())}}})));
}

void FuzzerPassAddSynonyms::CreateScalarAdditionSynonym(
    const opt::Instruction* inst, SpvOp opcode) {
  ApplyTransformation(TransformationAddSynonym(
      inst->result_id(),
      MakeInstructionMessage(
          opcode, inst->type_id(), GetFuzzerContext()->GetFreshId(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {inst->result_id()}},
              {SPV_OPERAND_TYPE_ID,
               {FindOrCreateZeroConstant(inst->type_id())}}})));
}

void FuzzerPassAddSynonyms::CreateCastSynonym(const opt::Instruction* inst) {
  const auto* type = GetIRContext()->get_type_mgr()->GetType(inst->type_id());
  assert(type && "Instruction has invalid type");

  uint32_t conversion_target_type_id;
  if (const auto* vector = type->AsVector()) {
    if (!vector->element_type()->AsInteger()) {
      // Ignore vectors with non-integral type.
      return;
    }
    conversion_target_type_id = FindOrCreateVectorType(
        FindOrCreateFloatType(vector->element_type()->AsInteger()->width()),
        vector->element_count());
  } else {
    assert(type->AsInteger() &&
           "Instruction must have either a vector or an integer type");
    conversion_target_type_id =
        FindOrCreateFloatType(type->AsInteger()->width());
  }

  ApplyTransformation(TransformationAddFloatCastSynonym(
      inst->result_id(), GetFuzzerContext()->GetFreshId(),
      GetFuzzerContext()->GetFreshId(), conversion_target_type_id));
}

void FuzzerPassAddSynonyms::CreateVectorMultiplicationSynonym(
    const opt::Instruction* inst) {
  const auto* type = GetIRContext()->get_type_mgr()->GetType(inst->type_id());
  assert(type && type->AsVector() && "Type of vector is invalid");

  SpvOp opcode;
  uint32_t one_id;
  if (const auto* integer = type->AsVector()->element_type()->AsInteger()) {
    opcode = SpvOpIMul;
    one_id =
        FindOrCreateIntegerConstant({1}, integer->width(), integer->IsSigned());
  } else if (const auto* floating =
                 type->AsVector()->element_type()->AsFloat()) {
    opcode = SpvOpFMul;
    one_id = FindOrCreateFloatConstant({1}, floating->width());
  } else {
    assert(type->AsVector()->element_type()->AsBool() &&
           "Vector components' type is not scalar");
    opcode = SpvOpLogicalAnd;
    one_id = FindOrCreateBoolConstant(true);
  }

  ApplyTransformation(TransformationAddSynonym(
      inst->result_id(),
      MakeInstructionMessage(
          opcode, inst->type_id(), GetFuzzerContext()->GetFreshId(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {inst->result_id()}},
              {SPV_OPERAND_TYPE_ID,
               {FindOrCreateCompositeConstant(
                   std::vector<uint32_t>(type->AsVector()->element_count(),
                                         one_id),
                   inst->type_id())}}})));
}

void FuzzerPassAddSynonyms::CreateVectorAdditionSynonym(
    const opt::Instruction* inst) {
  const auto* type = GetIRContext()->get_type_mgr()->GetType(inst->type_id());
  assert(type && type->AsVector() && "Type of vector is invalid");

  SpvOp opcode;
  if (type->AsVector()->element_type()->AsInteger()) {
    opcode = SpvOpIAdd;
  } else if (type->AsVector()->element_type()->AsFloat()) {
    opcode = SpvOpFAdd;
  } else {
    assert(type->AsVector()->element_type()->AsBool() &&
           "Vector components' type is not scalar");
    opcode = SpvOpLogicalOr;
  }

  ApplyTransformation(TransformationAddSynonym(
      inst->result_id(),
      MakeInstructionMessage(
          opcode, inst->type_id(), GetFuzzerContext()->GetFreshId(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {inst->result_id()}},
              {SPV_OPERAND_TYPE_ID,
               {FindOrCreateZeroConstant(inst->type_id())}}})));
}

}  // namespace fuzz
}  // namespace spvtools
