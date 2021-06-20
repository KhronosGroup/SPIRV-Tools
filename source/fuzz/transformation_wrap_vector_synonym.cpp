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
#include "source/fuzz/transformation_composite_construct.h"
#include "source/opt/function.h"
#include "source/opt/module.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/opt/instruction.h"
#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"
#include <algorithm>
#include "source/fuzz/fact_manager/fact_manager.h"

namespace spvtools {
namespace fuzz {

TransformationWrapVectorSynonym::TransformationWrapVectorSynonym(
    protobufs::TransformationWrapVectorSynonym message)
    : message_(std::move(message)) {}

TransformationWrapVectorSynonym::TransformationWrapVectorSynonym(uint32_t instruction_id, uint32_t vec_id1, uint32_t vec_id2,
                                                                 uint32_t vec_id3,  uint32_t vec_type_id, uint32_t pos,
                                                                 const std::vector<uint32_t>& vec1_elements, const std::vector<uint32_t>& vec2_elements) {
    message_.set_instruction_id(instruction_id);
    message_.set_vec_id1(vec_id1);
    message_.set_vec_id2(vec_id2);
    message_.set_vec_id3(vec_id3);
    message_.set_vec_type_id(vec_type_id);
    message_.set_scalar_position(pos);
    for(auto id : vec1_elements) {
      message_.add_vec1_elements(id);
    }
    for(auto id : vec2_elements) {
      message_.add_vec2_elements(id);
    }
}

bool TransformationWrapVectorSynonym::IsApplicable(
      opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
    std::unordered_set<SpvOp> valid_arithmetic_types {SpvOpIAdd, SpvOpISub, SpvOpIMul, SpvOpFAdd, SpvOpFSub, SpvOpFMul};
    auto instruction = ir_context->get_def_use_mgr()->GetDef(message_.instruction_id());
    auto vector_type = ir_context->get_def_use_mgr()->GetDef(message_.vec_type_id());

    // |instruction_id| must refer to an existing instruction.
    if(instruction == nullptr) {
      return false;
    }

    // |vec_type_id| must refer to a valid type instruction.
    if(vector_type == nullptr) {
      return false;
    }

    // |instruction_id| must be a valid arithmetic type.
    if(valid_arithmetic_types.count(instruction->opcode())) {
      return false;
    }

    // |vector_type_id| must correspond to a valid vector type.
    if(vector_type->opcode() == SpvOpTypeVector) {
      return false;
    }

    // |vec_id1|, |vec_id2| and |vec_id3| must be fresh.
    if(!fuzzerutil::IsFreshId(ir_context, message_.vec_id1()) || !fuzzerutil::IsFreshId(ir_context, message_.vec_id2())
        || !fuzzerutil::IsFreshId(ir_context, message_.vec_id3())) {
      return false;
    }

    // |vec_id1|, |vec_id2| and |vec_id3| should have disparate ids.
    if(fuzzerutil::HasDuplicates({message_.vec_id1(), message_.vec_id2(), message_.vec_id3()})) {
      return false;
    }

    // |scalar_position| needs to be a non-negative integer less than the vector length.
    auto vec_len = vector_type->GetSingleWordOperand(1);
    if(message_.scalar_position() >= vec_len) {
      return false;
    }

    auto vec1 = message_.vec1_elements();
    auto vec2 = message_.vec2_elements();
    // The vectors being populated must have the same length as specified by vector type.
    if(vec1.size() != (int)vec_len || vec2.size() != (int)vec_len) {
      return false;
    }

    auto type_id = instruction->type_id();
    auto matchType = [&ir_context, &type_id](uint32_t id) { return ir_context->get_def_use_mgr()->GetDef(id)->type_id() == type_id;};

    // All ids should match the type_id specified in the instruction.
    if(!std::all_of(vec1.begin(), vec1.end(), matchType) || !std::all_of(vec2.begin(), vec2.end(), matchType)) {
      return false;
    }

    // Get the OpConstant instruction with id corresponding to index |pos|.
    auto constant1 = ir_context->get_def_use_mgr()->GetDef(vec1[message_.scalar_position()]);
    auto constant2 = ir_context->get_def_use_mgr()->GetDef(vec2[message_.scalar_position()]);
    // The constants at position |pos| of the vectors should be zero constants.
    if(constant1->GetSingleWordOperand(1) || constant2->GetSingleWordOperand(1)) {
      return false;
    }

    return true;
}

void TransformationWrapVectorSynonym::Apply(
    opt::IRContext* ir_context, TransformationContext* transformation_context) const {
    // OpCompositeConstructs are inserted before the original arithmetic type instruction.
    auto instruction = ir_context->get_def_use_mgr()->GetDef(message_.instruction_id());
    auto vec1 = message_.vec1_elements();
    auto vec2 = message_.vec2_elements();

    // Change the target position with variable from the original instruction.
    vec1[message_.scalar_position()] = instruction->GetSingleWordOperand(1);
    vec2[message_.scalar_position()] = instruction->GetSingleWordOperand(2);

    auto inst_descriptor = MakeInstructionDescriptor(message_.instruction_id(), instruction->opcode(), 0);

    // Apply transformation to add two composite construct.
    ApplyTransformation(TransformationCompositeConstruct(
        message_.vec_type_id(), message_.vec1_elements(), inst_descriptor, message_.vec_id1()));
    ApplyTransformation(TransformationCompositeConstruct(
        message_.vec_type_id(), message_.vec1_elements(), inst_descriptor, message_.vec_id2()));

    // Insert an arithmetic operation that combines the two vector into a new vector with id |vec_id3|.
    // New instruction has the same opcode as the original instruction.
    // New instruction has a return type specified by |vec_type_id| and takes
    // two vector as input specified by |vec_id1| and |vec_id2|.
    auto insert_before_inst =
            FindInstruction(inst_descriptor, ir_context);
    auto destination_block = ir_context->get_instr_block(insert_before_inst);
    auto insert_before = fuzzerutil::GetIteratorForInstruction(
            destination_block, insert_before_inst);

    // Make a new arithmetic instruction: %vec_id3 = OpXX %type_id %vec_id1 %vec_id2
    auto new_instruction = MakeUnique<opt::Instruction>(
            ir_context, instruction->opcode(), message_.vec_type_id(),
            message_.vec_id3(), message_.vec_id1(), message_.vec_id2());
    auto new_instruction_ptr = new_instruction.get();
    insert_before.InsertBefore(std::move(new_instruction));
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
    ir_context->set_instr_block(new_instruction_ptr, destination_block);
    
    // Add |vec_id3| to id bound.
    fuzzerutil::UpdateModuleIdBound(ir_context, message_.vec_id3());

    // Add synonyms between |vec_id3| and |instruction_id|.
    auto result_vec_descriptor = MakeDataDescriptor(message_.vec_id3(), {});
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
