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

#include "source/fuzz/transformation_replace_add_sub_mul_with_carrying_extended.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceAddSubMulWithCarryingExtended::
    TransformationReplaceAddSubMulWithCarryingExtended(
        const spvtools::fuzz::protobufs::
            TransformationReplaceAddSubMulWithCarryingExtended& message)
    : message_(message) {}

TransformationReplaceAddSubMulWithCarryingExtended::
    TransformationReplaceAddSubMulWithCarryingExtended(uint32_t struct_fresh_id,
                                                       uint32_t struct_type_id,
                                                       uint32_t result_id) {
  message_.set_struct_fresh_id(struct_fresh_id);
  message_.set_struct_type_id(struct_type_id);
  message_.set_result_id(result_id);
}

bool TransformationReplaceAddSubMulWithCarryingExtended::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext&) const {
  // |message_.struct_fresh_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.struct_fresh_id())) {
    return false;
  }

  // |message_.result_id| must refer to an OpIAdd or OpISub or OpIMul
  // instruction
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  auto instruction_opcode = instruction->opcode();
  if (instruction_opcode != SpvOpIAdd && instruction_opcode != SpvOpISub &&
      instruction_opcode != SpvOpIMul)
    return false;

  uint32_t operand_1_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(instruction->GetSingleWordOperand(2))
          ->type_id();

  uint32_t operand_2_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(instruction->GetSingleWordOperand(3))
          ->type_id();

  uint32_t operand_1_signedness = ir_context->get_def_use_mgr()
                                      ->GetDef(operand_1_type_id)
                                      ->GetSingleWordOperand(2);
  uint32_t operand_2_signedness = ir_context->get_def_use_mgr()
                                      ->GetDef(operand_2_type_id)
                                      ->GetSingleWordOperand(2);
  switch (instruction_opcode) {
    case SpvOpIAdd:
    case SpvOpISub:
      return operand_1_signedness == 0 && operand_2_signedness == 0;
    default:
      return true;
  }
}

void TransformationReplaceAddSubMulWithCarryingExtended::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // |message_.struct_fresh_id| must be fresh.
  assert(fuzzerutil::IsFreshId(ir_context, message_.struct_fresh_id()) &&
         "|message_.struct_fresh_id| must be fresh");

  // |message_.result_id| must refer to an OpIAdd or OpISub or OpIMul
  // instruction
  auto original_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  auto original_instruction_opcode = original_instruction->opcode();
  assert((original_instruction_opcode == SpvOpIAdd ||
          original_instruction_opcode == SpvOpISub ||
          original_instruction_opcode == SpvOpIMul) &&
         "The instruction must have the opcode: OpIAdd or OpISub or OpIMul");

  uint32_t operand_1_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(original_instruction->GetSingleWordOperand(2))
          ->type_id();

  uint32_t operand_2_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(original_instruction->GetSingleWordOperand(3))
          ->type_id();

  assert(operand_1_type_id == operand_2_type_id &&
         "The type ids of components must be equal");

  uint32_t operand_1_signedness = ir_context->get_def_use_mgr()
                                      ->GetDef(operand_1_type_id)
                                      ->GetSingleWordOperand(2);
  uint32_t operand_2_signedness = ir_context->get_def_use_mgr()
                                      ->GetDef(operand_2_type_id)
                                      ->GetSingleWordOperand(2);

  switch (original_instruction_opcode) {
    case SpvOpIAdd:
    case SpvOpISub:
      assert(operand_1_signedness == 0 && operand_2_signedness == 0 &&
             "Components must be unsigned.");
      break;
    default:
      break;
  }

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.struct_fresh_id());

  // Insert the OpCompositeExtract.
  auto instruction_composite_extract =
      original_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpCompositeExtract, operand_1_type_id,
          message_.result_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.struct_fresh_id()}},
               {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}})));
  switch (original_instruction_opcode) {
    case SpvOpIAdd:
      // Insert the OpIAddCarry before the OpCompositeExtract.
      instruction_composite_extract->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpIAddCarry, message_.struct_type_id(),
          message_.struct_fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID,
                {original_instruction->GetSingleWordOperand(2)}},
               {SPV_OPERAND_TYPE_ID,
                {original_instruction->GetSingleWordOperand(3)}}})));
      break;
    case SpvOpISub:
      // Insert the OpISubBorrow before the OpCompositeExtract.
      instruction_composite_extract->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpISubBorrow, message_.struct_type_id(),
          message_.struct_fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID,
                {original_instruction->GetSingleWordOperand(2)}},
               {SPV_OPERAND_TYPE_ID,
                {original_instruction->GetSingleWordOperand(3)}}})));
      break;
    case SpvOpIMul:
      if (operand_1_signedness == 0) {
        // Insert the OpUMulExtended before the OpCompositeExtract.
        instruction_composite_extract->InsertBefore(
            MakeUnique<opt::Instruction>(
                ir_context, SpvOpUMulExtended, message_.struct_type_id(),
                message_.struct_fresh_id(),
                opt::Instruction::OperandList(
                    {{SPV_OPERAND_TYPE_ID,
                      {original_instruction->GetSingleWordOperand(2)}},
                     {SPV_OPERAND_TYPE_ID,
                      {original_instruction->GetSingleWordOperand(3)}}})));
      } else {
        // Insert the OpSMulExtended before the OpCompositeExtract.
        instruction_composite_extract->InsertBefore(
            MakeUnique<opt::Instruction>(
                ir_context, SpvOpSMulExtended, message_.struct_type_id(),
                message_.struct_fresh_id(),
                opt::Instruction::OperandList(
                    {{SPV_OPERAND_TYPE_ID,
                      {original_instruction->GetSingleWordOperand(2)}},
                     {SPV_OPERAND_TYPE_ID,
                      {original_instruction->GetSingleWordOperand(3)}}})));
      }
      break;
    default:
      break;
  }
  // Remove the original instruction.
  ir_context->KillInst(original_instruction);

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation
TransformationReplaceAddSubMulWithCarryingExtended::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_add_sub_mul_with_carrying_extended() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools