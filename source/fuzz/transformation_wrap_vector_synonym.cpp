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
#include "source/fuzz/fuzzer_util.h"
#include "source/opt/instruction.h"
#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/instruction_descriptor.h"
#include <algorithm>

namespace spvtools {
namespace fuzz {

TransformationWrapVectorSynonym::TransformationWrapVectorSynonym(
    protobufs::TransformationWrapVectorSynonym message)
    : message_(std::move(message)) {}

TransformationWrapVectorSynonym:: TransformationWrapVectorSynonym(uint32_t instruction_id, uint32_t result_id1, uint32_t result_id2, uint32_t vec_id,  uint32_t vec_type_id, uint32_t pos) {

    message_.set_instruction_id(instruction_id);
    message_.set_result_id1(result_id1);
    message_.set_result_id2(result_id2);
    message_.set_vec_id(vec_id);
    message_.set_vec_type_id(vec_type_id);
    message_.set_scalar_position(pos);
}

bool TransformationWrapVectorSynonym::IsApplicable(
      opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {

    auto instruction = ir_context->get_def_use_mgr()->GetDef(message_.instruction_id());
    auto vector_type = ir_context->get_type_mgr()->GetType(message_.vec_type_id());
    auto vec1 = ir_context->get_def_use_mgr()->GetDef(message_.result_id1());
    auto vec2 = ir_context->get_def_use_mgr()->GetDef(message_.result_id2());

    // |instruction_id| must refer to an existing instruction.
    if(instruction == nullptr) {
      return false;
    }

    std::unordered_set<SpvOp_> valid_opcodes {SpvOpIAdd, SpvOpISub, SpvOpIMul, SpvOpSDiv, SpvOpUDiv, SpvOpFAdd, SpvOpFSub, SpvOpFMul, SpvOpFDiv};

    if(!valid_opcodes.count(instruction->opcode())) {
      return false;
    }

    // |vec_type_id| must refer to a valid type instruction.
    if(vector_type == nullptr) {
      return false;
    }

    // |result_id1| and |result_id2| must exists.
    if(vec1 == nullptr || vec2 == nullptr) {
      return false;
    }

    // |vector_type_id| must correspond to a valid vector type.
    if(vector_type->AsVector() == nullptr) {
      return false;
    }

    // |vec_id| must be fresh.
    if(!fuzzerutil::IsFreshId(ir_context, message_.vec_id())) {
      return false;
    }

    std::vector<uint32_t> vec_ids {message_.result_id1(), message_.result_id2(), message_.vec_id()};

    // |vec_id|, |result_id1| and |result_id2| should be disparate ids.
    if(fuzzerutil::HasDuplicates(vec_ids)) {
      return false;
    }

    // |scalar_position| needs to be a non-negative integer less than the vector length.
    auto vec_len = ir_context->get_def_use_mgr()->GetDef(message_.vec_type_id())->GetSingleWordOperand(1);
    if(message_.scalar_position() >= vec_len) {
      return false;
    }

    // The 2 vectors must have the same type as the result vector type.
    auto vec1_type_id = vec1->GetSingleWordOperand(0);
    auto vec2_type_id = vec2->GetSingleWordOperand(0);

    if(vec1_type_id != message_.vec_type_id() || vec2_type_id!= message_.vec_type_id()) {
      return false;
    }

    return true;
}

void TransformationWrapVectorSynonym::Apply(
    opt::IRContext* ir_context, TransformationContext* transformation_context) const {

    auto instruction = ir_context->get_def_use_mgr()->GetDef(message_.instruction_id());
    // Create an instruction descriptor for the original instruction.
    auto inst_descriptor = MakeInstructionDescriptor(message_.instruction_id(), instruction->opcode(), 0);
    auto insert_before_inst =
            FindInstruction(inst_descriptor, ir_context);
    auto destination_block = ir_context->get_instr_block(insert_before_inst);
    auto insert_before = fuzzerutil::GetIteratorForInstruction(
            destination_block, insert_before_inst);

    //  Populate input operand list with two vectors for vector operation.
    opt::Instruction::OperandList in_operands;
    in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.result_id1()}});
    in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.result_id2()}});

    // Make a new arithmetic instruction: %vec_id = OpXX %type_id %result_id1 %result_id2.
    auto new_instruction = MakeUnique<opt::Instruction>(
            ir_context, instruction->opcode(), message_.vec_type_id(),
            message_.vec_id(), in_operands);
    auto new_instruction_ptr = new_instruction.get();
    insert_before.InsertBefore(std::move(new_instruction));
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
    ir_context->set_instr_block(new_instruction_ptr, destination_block);
    
    // Add |vec_id| to id bound.
    fuzzerutil::UpdateModuleIdBound(ir_context, message_.vec_id());

    // Add synonyms between |vec_id| and |instruction_id|.
    auto result_vec_descriptor = MakeDataDescriptor(message_.vec_id(), {message_.scalar_position()});
    auto original_inst_descriptor = MakeDataDescriptor(message_.instruction_id(), {});
    transformation_context->GetFactManager()->AddFactDataSynonym(result_vec_descriptor, original_inst_descriptor);

}

protobufs::Transformation TransformationWrapVectorSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_wrap_vector_synonym() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationWrapVectorSynonym::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
