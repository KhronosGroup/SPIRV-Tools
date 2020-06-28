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
      SpvOp,
      std::vector<std::function<void(
          const opt::Instruction*, const protobufs::InstructionDescriptor&)>>>;

  const TransformationsMap kTransformationMap = {
      {SpvOpTypeInt,
       {[this](const opt::Instruction* inst,
               const protobufs::InstructionDescriptor& instruction_descriptor) {
          CreateScalarMultiplicationSynonym(inst, instruction_descriptor,
                                            SpvOpIMul);
        },
        [this](const opt::Instruction* inst,
               const protobufs::InstructionDescriptor& instruction_descriptor) {
          CreateScalarAdditionSynonym(inst, instruction_descriptor, SpvOpIAdd);
        }}},
      {SpvOpTypeFloat,
       {[this](const opt::Instruction* inst,
               const protobufs::InstructionDescriptor& instruction_descriptor) {
          CreateScalarMultiplicationSynonym(inst, instruction_descriptor,
                                            SpvOpFMul);
        },
        [this](const opt::Instruction* inst,
               const protobufs::InstructionDescriptor& instruction_descriptor) {
          CreateScalarAdditionSynonym(inst, instruction_descriptor, SpvOpFAdd);
        }}},
      {SpvOpTypeBool,
       {[this](const opt::Instruction* inst,
               const protobufs::InstructionDescriptor& instruction_descriptor) {
          CreateScalarMultiplicationSynonym(inst, instruction_descriptor,
                                            SpvOpLogicalAnd);
        },
        [this](const opt::Instruction* inst,
               const protobufs::InstructionDescriptor& instruction_descriptor) {
          CreateScalarAdditionSynonym(inst, instruction_descriptor,
                                      SpvOpLogicalOr);
        }}},
      {SpvOpTypeVector,
       {[this](const opt::Instruction* inst,
               const protobufs::InstructionDescriptor& instruction_descriptor) {
          CreateVectorMultiplicationSynonym(inst, instruction_descriptor);
        },
        [this](const opt::Instruction* inst,
               const protobufs::InstructionDescriptor& instruction_descriptor) {
          CreateVectorAdditionSynonym(inst, instruction_descriptor);
        }}}};

  ForEachInstructionWithInstructionDescriptor(
      [this, &kTransformationMap](
          opt::Function* function, opt::BasicBlock* block,
          opt::BasicBlock::iterator inst_it,
          const protobufs::InstructionDescriptor& instruction_descriptor) {
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpIAdd, inst_it)) {
          return;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingSynonyms())) {
          return;
        }

        auto available_instructions = FindAvailableInstructions(
            function, block, inst_it,
            [&kTransformationMap](opt::IRContext* ir_context,
                                  opt::Instruction* inst) {
              if (!inst->result_id() || !inst->type_id() ||
                  inst->opcode() == SpvOpUndef) {
                return false;
              }

              const auto* type_inst =
                  ir_context->get_def_use_mgr()->GetDef(inst->type_id());
              assert(type_inst && "Instruction must have a valid type");
              return kTransformationMap.find(type_inst->opcode()) !=
                     kTransformationMap.end();
            });

        if (available_instructions.empty()) {
          return;
        }

        const auto* candidate_inst =
            available_instructions[GetFuzzerContext()->RandomIndex(
                available_instructions)];
        const auto* candidate_type_inst =
            GetIRContext()->get_def_use_mgr()->GetDef(
                candidate_inst->type_id());
        const auto& transformations =
            kTransformationMap.at(candidate_type_inst->opcode());

        transformations[GetFuzzerContext()->RandomIndex(transformations)](
            candidate_inst, instruction_descriptor);
      });
}

void FuzzerPassAddSynonyms::CreateScalarMultiplicationSynonym(
    const opt::Instruction* inst,
    const protobufs::InstructionDescriptor& instruction_descriptor,
    SpvOp opcode) {
  ApplyTransformation(TransformationAddSynonym(
      inst->result_id(), instruction_descriptor,
      MakeInstructionMessage(
          opcode, inst->type_id(), GetFuzzerContext()->GetFreshId(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {inst->result_id()}},
              {SPV_OPERAND_TYPE_ID,
               {FindOrCreateConstant({1}, inst->type_id())}}})));
}

void FuzzerPassAddSynonyms::CreateScalarAdditionSynonym(
    const opt::Instruction* inst,
    const protobufs::InstructionDescriptor& instruction_descriptor,
    SpvOp opcode) {
  ApplyTransformation(TransformationAddSynonym(
      inst->result_id(), instruction_descriptor,
      MakeInstructionMessage(
          opcode, inst->type_id(), GetFuzzerContext()->GetFreshId(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {inst->result_id()}},
              {SPV_OPERAND_TYPE_ID,
               {FindOrCreateZeroConstant(inst->type_id())}}})));
}

void FuzzerPassAddSynonyms::CreateVectorMultiplicationSynonym(
    const opt::Instruction* inst,
    const protobufs::InstructionDescriptor& instruction_descriptor) {
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

  // Recompute instruction's type if it was invalidated.
  type = GetIRContext()->get_type_mgr()->GetType(inst->type_id());
  assert(type && type->AsVector() && "Instruction must have a vector type");

  ApplyTransformation(TransformationAddSynonym(
      inst->result_id(), instruction_descriptor,
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
    const opt::Instruction* inst,
    const protobufs::InstructionDescriptor& instruction_descriptor) {
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
      inst->result_id(), instruction_descriptor,
      MakeInstructionMessage(
          opcode, inst->type_id(), GetFuzzerContext()->GetFreshId(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {inst->result_id()}},
              {SPV_OPERAND_TYPE_ID,
               {FindOrCreateZeroConstant(inst->type_id())}}})));
}

}  // namespace fuzz
}  // namespace spvtools
