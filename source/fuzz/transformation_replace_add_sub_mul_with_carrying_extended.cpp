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

namespace {
const uint32_t kOpCompositeExtractIndexLowOrderBits = 0;
const uint32_t kArithmeticInstructionIndexLeftInOperand = 0;
const uint32_t kArithmeticInstructionIndexRightInOperand = 1;
const uint32_t kOpTypeIndexSignedness = 2;
}  // namespace

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

  // |message_.result_id| must refer to a suitable OpIAdd, OpISub or OpIMul
  // instruction. The instruction must be defined.
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  if (instruction == nullptr) {
    return false;
  }
  auto instruction_opcode = instruction->opcode();

  switch (instruction_opcode) {
    case SpvOpIAdd:
    case SpvOpISub:
    case SpvOpIMul:
      if (!TransformationReplaceAddSubMulWithCarryingExtended::
              IsInstructionSuitable(ir_context, instruction))
        return false;
      break;
    default:
      return false;
  }
  return true;
}

void TransformationReplaceAddSubMulWithCarryingExtended::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // |message_.struct_fresh_id| must be fresh.
  assert(fuzzerutil::IsFreshId(ir_context, message_.struct_fresh_id()) &&
         "|message_.struct_fresh_id| must be fresh");

  auto original_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  uint32_t operand_signedness =
      ir_context->get_def_use_mgr()
          ->GetDef(original_instruction->type_id())
          ->GetSingleWordOperand(kOpTypeIndexSignedness);

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.struct_fresh_id());

  // Determine the opcode of the new instruction that computes the result into a
  // struct.
  SpvOp new_instruction_opcode;

  switch (original_instruction->opcode()) {
    case SpvOpIAdd:
      new_instruction_opcode = SpvOpIAddCarry;
      break;
    case SpvOpISub:
      new_instruction_opcode = SpvOpISubBorrow;
      break;
    case SpvOpIMul:
      if (operand_signedness == 0) {
        new_instruction_opcode = SpvOpUMulExtended;
      } else {
        new_instruction_opcode = SpvOpSMulExtended;
      }
      break;
    default:
      assert(false);
      return;
  }

  // Insert the new instruction that computes the result into a struct before
  // the  |original_instruction|.
  original_instruction->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, new_instruction_opcode, message_.struct_type_id(),
      message_.struct_fresh_id(),
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID,
            {original_instruction->GetSingleWordInOperand(
                kArithmeticInstructionIndexLeftInOperand)}},
           {SPV_OPERAND_TYPE_ID,
            {original_instruction->GetSingleWordInOperand(
                kArithmeticInstructionIndexRightInOperand)}}})));

  // Insert the OpCompositeExtract after the added instruction. This instruction
  // takes the first component of the struct which represents low-order bits of
  // the operation. This the original result.
  original_instruction->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, SpvOpCompositeExtract, original_instruction->type_id(),
      message_.result_id(),
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.struct_fresh_id()}},
           {SPV_OPERAND_TYPE_LITERAL_INTEGER,
            {kOpCompositeExtractIndexLowOrderBits}}})));

  // Remove the original instruction.
  ir_context->KillInst(original_instruction);

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

bool TransformationReplaceAddSubMulWithCarryingExtended::IsInstructionSuitable(
    opt::IRContext* ir_context, const opt::Instruction* instruction) {
  auto instruction_opcode = instruction->opcode();

  // Only instructions OpIAdd, OpISub, OpIMul are supported.
  switch (instruction_opcode) {
    case SpvOpIAdd:
    case SpvOpISub:
    case SpvOpIMul:
      break;
    default:
      return false;
  }
  uint32_t operand_1_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(instruction->GetSingleWordInOperand(
              kArithmeticInstructionIndexLeftInOperand))
          ->type_id();

  uint32_t operand_2_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(instruction->GetSingleWordInOperand(
              kArithmeticInstructionIndexRightInOperand))
          ->type_id();
  uint32_t result_type_id = instruction->type_id();

  // Both type ids of the operands and the result type ids must be equal.
  if (operand_1_type_id != operand_2_type_id) {
    return false;
  }
  if (operand_2_type_id != result_type_id) {
    return false;
  }

  // In case of OpIAdd and OpISub, the type must be unsigned.
  uint32_t instruction_signedness;
  switch (instruction_opcode) {
    case SpvOpIAdd:
    case SpvOpISub:
      // Both types of the operands and the result type must be unsigned.
      instruction_signedness =
          ir_context->get_def_use_mgr()
              ->GetDef(instruction->type_id())
              ->GetSingleWordOperand(kOpTypeIndexSignedness);
      if (instruction_signedness != 0) {
        return false;
      }
      break;
    default:
      break;
  }
  return true;
}

protobufs::Transformation
TransformationReplaceAddSubMulWithCarryingExtended::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_add_sub_mul_with_carrying_extended() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
