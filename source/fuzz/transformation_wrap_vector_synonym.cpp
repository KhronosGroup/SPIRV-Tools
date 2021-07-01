// Copyright (c) 2021 Shiyu Liu
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

#include "source/fuzz/transformation_wrap_vector_synonym.h"
#include <algorithm>
#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace fuzz {

TransformationWrapVectorSynonym::TransformationWrapVectorSynonym(
    protobufs::TransformationWrapVectorSynonym message)
    : message_(std::move(message)) {}

TransformationWrapVectorSynonym::TransformationWrapVectorSynonym(
    uint32_t instruction_id, uint32_t vector_operand1, uint32_t vector_operand2,
    uint32_t vec_id, uint32_t pos) {
  message_.set_instruction_id(instruction_id);
  message_.set_vector_operand1(vector_operand1);
  message_.set_vector_operand2(vector_operand2);
  message_.set_vec_id(vec_id);
  message_.set_scalar_position(pos);
}

bool TransformationWrapVectorSynonym::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& transformation_context) const {
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_id());
  auto vec1 = ir_context->get_def_use_mgr()->GetDef(message_.vector_operand1());
  auto vec2 = ir_context->get_def_use_mgr()->GetDef(message_.vector_operand2());

  // |instruction_id| must refer to an existing instruction.
  if (instruction == nullptr) {
    return false;
  }

  std::unordered_set<SpvOp> valid_opcodes{SpvOpIAdd, SpvOpISub, SpvOpIMul,
                                           SpvOpSDiv, SpvOpUDiv, SpvOpFAdd,
                                           SpvOpFSub, SpvOpFMul, SpvOpFDiv};

  if (!valid_opcodes.count(instruction->opcode())) {
    return false;
  }


  // |vector_operand1| and |vector_operand2| must exist.
  if (vec1 == nullptr || vec2 == nullptr) {
    return false;
  }

  // The 2 vectors must be the same valid vector type.
  auto vec1_type_id = vec1->type_id();
  auto vec2_type_id = vec2->type_id();
  auto vec_type = ir_context->get_type_mgr()->GetType(vec1_type_id);

  if (vec1_type_id != vec2_type_id) {
    return false;
  }

  if(vec_type->AsVector() == nullptr) {
    return false;
  }

  // |vec_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.vec_id())) {
    return false;
  }

  std::vector<uint32_t> vec_ids{message_.vector_operand1(), message_.vector_operand2(),
                                message_.vec_id()};

  // |vec_id|, |result_id1| and |result_id2| should be disparate ids.
  if (fuzzerutil::HasDuplicates(vec_ids)) {
    return false;
  }

  // |scalar_position| needs to be a non-negative integer less than the vector
  // length.
  // OpTypeVector instruction has the component count at index 2.
  auto vec_len = ir_context->get_def_use_mgr()
                     ->GetDef(vec1_type_id)
                     ->GetSingleWordOperand(2);
  if (message_.scalar_position() >= vec_len) {
    return false;
  }

  // Check the id in the corresponding position from the result vectors are synonyms with the
  // operands from the original instruction.
  uint32_t pos = message_.scalar_position();
  uint32_t inst_operand1 = instruction->GetSingleWordInOperand(0);
  uint32_t inst_operand2 = instruction->GetSingleWordInOperand(1);
  bool check_synonym_operand1 = transformation_context.GetFactManager()->IsSynonymous(
      MakeDataDescriptor(message_.vector_operand1(), {pos}), MakeDataDescriptor(inst_operand1, {}));
  bool check_synonym_operand2 = transformation_context.GetFactManager()->IsSynonymous(
      MakeDataDescriptor(message_.vector_operand2(), {pos}), MakeDataDescriptor(inst_operand2, {}));

  if(!check_synonym_operand1 || !check_synonym_operand2) {
    return false;
  }

  return true;
}

void TransformationWrapVectorSynonym::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Create an instruction descriptor for the original instruction.
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_id());
  auto destination_block = ir_context->get_instr_block(instruction);

  //  Populate input operand list with two vectors for vector operation.
  opt::Instruction::OperandList in_operands;
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.vector_operand1()}});
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.vector_operand2()}});

  // Make a new arithmetic instruction: %vec_id = OpXX %type_id %result_id1
  // %result_id2.
  auto vec_type_id = ir_context->get_def_use_mgr()->GetDef(message_.vector_operand1())->type_id();
  auto new_instruction = MakeUnique<opt::Instruction>(
      ir_context, instruction->opcode(), vec_type_id,
      message_.vec_id(), in_operands);
  auto new_instruction_ptr = new_instruction.get();
  instruction->InsertBefore(std::move(new_instruction));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
  ir_context->set_instr_block(new_instruction_ptr, destination_block);

  // Add |vec_id| to id bound.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.vec_id());

  // Add synonyms between |vec_id| and |instruction_id|.
  auto result_vec_descriptor =
      MakeDataDescriptor(message_.vec_id(), {message_.scalar_position()});
  auto original_inst_descriptor =
      MakeDataDescriptor(message_.instruction_id(), {});
  transformation_context->GetFactManager()->AddFactDataSynonym(
      result_vec_descriptor, original_inst_descriptor);
}

protobufs::Transformation TransformationWrapVectorSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_wrap_vector_synonym() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationWrapVectorSynonym::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>{message_.vec_id()};
}

}  // namespace fuzz
}  // namespace spvtools
