// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_vector_dynamic.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationVectorDynamic::TransformationVectorDynamic(
    const spvtools::fuzz::protobufs::TransformationVectorDynamic& message)
    : message_(message) {}

TransformationVectorDynamic::TransformationVectorDynamic(
    uint32_t instruction_result_id) {
  message_.set_instruction_result_id(instruction_result_id);
}

bool TransformationVectorDynamic::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // |instruction| must be a vector operation.
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_result_id());
  if (!IsVectorOperation(ir_context, instruction)) {
    return false;
  }

  // The |instruction| literal operand must be defined as constant.
  if (!MaybeGetConstantForIndex(ir_context, *instruction,
                                transformation_context)) {
    return false;
  }

  return true;
}

void TransformationVectorDynamic::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_result_id());

  // The OpVectorInsertDynamic instruction has the vector and component operands
  // in reverse order in relation to the OpCompositeInsert corresponding
  // operands.
  if (instruction->opcode() == SpvOpCompositeInsert) {
    std::swap(instruction->GetInOperand(0), instruction->GetInOperand(1));
  }

  // Sets the literal operand to the equivalent constant.
  instruction->SetInOperand(
      instruction->opcode() == SpvOpCompositeExtract ? 1 : 2,
      {MaybeGetConstantForIndex(ir_context, *instruction,
                                *transformation_context)});

  // Sets the |instruction| opcode to the corresponding vector dynamic opcode.
  instruction->SetOpcode(instruction->opcode() == SpvOpCompositeExtract
                             ? SpvOpVectorExtractDynamic
                             : SpvOpVectorInsertDynamic);
}

protobufs::Transformation TransformationVectorDynamic::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_vector_dynamic() = message_;
  return result;
}

bool TransformationVectorDynamic::IsVectorOperation(
    opt::IRContext* ir_context, opt::Instruction* instruction) {
  // |instruction| must be defined and must be an OpCompositeExtract/Insert
  // instruction.
  if (!instruction || (instruction->opcode() != SpvOpCompositeExtract &&
                       instruction->opcode() != SpvOpCompositeInsert)) {
    return false;
  }

  // The composite must be a vector.
  auto composite_instruction =
      ir_context->get_def_use_mgr()->GetDef(instruction->GetSingleWordInOperand(
          instruction->opcode() == SpvOpCompositeExtract ? 0 : 1));
  if (!ir_context->get_type_mgr()
           ->GetType(composite_instruction->type_id())
           ->AsVector()) {
    return false;
  }

  return true;
}

uint32_t TransformationVectorDynamic::MaybeGetConstantForIndex(
    opt::IRContext* ir_context, const opt::Instruction& instruction,
    const TransformationContext& transformation_context) {
  uint32_t literal = instruction.GetSingleWordInOperand(
      instruction.opcode() == SpvOpCompositeExtract ? 1 : 2);
  uint32_t integer_constant_id = fuzzerutil::MaybeGetIntegerConstant(
      ir_context, transformation_context, {literal}, 32, false, false);
  return integer_constant_id ? integer_constant_id
                             : fuzzerutil::MaybeGetIntegerConstant(
                                   ir_context, transformation_context,
                                   {literal}, 32, true, false);
}

}  // namespace fuzz
}  // namespace spvtools
