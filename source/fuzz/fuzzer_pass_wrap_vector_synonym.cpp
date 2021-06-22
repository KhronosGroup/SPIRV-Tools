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

#include "source/fuzz/fuzzer_pass_wrap_vector_synonym.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/transformation_wrap_vector_synonym.h"
#include "source/fuzz/random_generator.h"

namespace spvtools {
namespace fuzz {

FuzzerPassWrapVectorSynonym::FuzzerPassWrapVectorSynonym(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

void FuzzerPassWrapVectorSynonym::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* /*unused*/, opt::BasicBlock* /*unused*/,
             opt::BasicBlock::iterator instruction_iterator,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        // Only run fuzzer pass on valid arithmetic operation instruction.
        if(!valid_arithmetic_types.count(instruction_iterator->opcode())) return;

        assert(instruction_iterator->opcode() ==
               instruction_descriptor.target_instruction_opcode() &&
               "The opcode of the instruction we might insert before must be "
               "the same as the opcode in the descriptor for the instruction");

        // Randomly decide whether to wrap it to a vector operation.
        if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfWrappingVectorSynonym())) {
          return;
        }

        // It must be valid to insert an OpCompositeConstruct instruction
        // before |instruction_iterator|.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
            SpvOpCompositeConstruct, instruction_iterator)) {
          return;
        }
        // Get the scalar type represented by the targeted instruction id.
        uint32_t scalar_type_id = instruction_iterator->type_id();

        RandomGenerator* random_generator;

        // Get a random vector size from 2 to 4.
        uint32_t component_count = random_generator->RandomUint32(3) + 2;
        // Get the vector type of size range from 2 to 4 of the corresponding scalar type.
        uint32_t vec_type_id =  FindOrCreateVectorType(scalar_type_id, component_count);

        // Randomly choose a position that target ids should be placed at.
        // The position is in range [0, n - 1], where n is the size of the vector.
        uint32_t position = random_generator->RandomUint32(component_count);

        uint32_t target_id1 = instruction_iterator->GetSingleWordOperand(1);
        uint32_t target_id2 = instruction_iterator->GetSingleWordOperand(2);

        // Stores the ids of scalar constants.
        std::vector<uint32_t> vec1_components;
        std::vector<uint32_t> vec2_components;

        // Width is specified in the first index for either OpTypeInt or OpTypeFloat.
        uint32_t width = GetIRContext()->get_def_use_mgr()->GetDef(scalar_type_id)->GetSingleWordOperand(0);

        auto type = GetIRContext()->get_type_mgr()->GetType(scalar_type_id);

        bool is_signed = type->AsInteger() ? type->AsInteger()->IsSigned() : true;

        // Populate components based on vector type and size.
        for(uint32_t i = 0; i < component_count; ++i) {
          if(i == position) {
            vec1_components.emplace_back(target_id1);
            vec2_components.emplace_back(target_id2);
          } else {
            switch (instruction_iterator->opcode()) {
              case SpvOpTypeInt: {
                AddRandomIntConstant(vec1_components, width, is_signed, random_generator);
                AddRandomIntConstant(vec2_components, width, is_signed, random_generator);
                break;
              }
              case SpvOpTypeFloat: {
                AddRandomFloatConstant(vec1_components, width, random_generator);
                AddRandomFloatConstant(vec2_components, width, random_generator);
                break;
              }
              default:
                assert(false && "Instruction opcode must be a valid numeric constant type");
            }
          }
        }
        // Add two OpCompositeConstruct to the module.
        uint32_t result_id1 = AddNewVecNType(scalar_type_id, vec1_components, instruction_descriptor);
        uint32_t result_id2 = AddNewVecNType(scalar_type_id, vec2_components, instruction_descriptor);

        // Apply transformation to do vector operation and add synonym between the result
        // vector id and the id of the original instruction.
        // TransformationWrapVectorSynonym(uint32_t instruction_id, uint32_t result_id1, uint32_t result_id2, uint32_t vec_id, uint32_t vec_type_id, uint32_t pos);

        ApplyTransformation(TransformationWrapVectorSynonym(instruction_iterator->result_id(), result_id1,
                                                            result_id2, GetFuzzerContext()->GetFreshId(), vec_type_id, position));
      });
}

uint32_t FuzzerPassWrapVectorSynonym::AddNewVecNType(uint32_t composite_type_id, std::vector<uint32_t> component,
                                                     const protobufs::InstructionDescriptor& inst_to_insert_before) {
  uint32_t current_fresh_id = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationCompositeConstruct(composite_type_id, component,
                                                       inst_to_insert_before, current_fresh_id));
  return current_fresh_id;
}

void FuzzerPassWrapVectorSynonym::AddRandomFloatConstant(std::vector<uint32_t>& vec, uint32_t width, RandomGenerator* random_generator) {

  float multiplier = random_generator->RandomBool() ? 1 : -1;
  float random_float = multiplier *
      (float)(random_generator->RandomUint32(1000) +
              random_generator->RandomDouble());
  // Make sure the created float is not zero.
  if(random_float == 0) random_float += (float)0.1;
  std::vector<uint32_t> words = {
      fuzzerutil::FloatToWord(random_float)};
  vec.emplace_back(
      FindOrCreateFloatConstant(words, width, false));
}

void FuzzerPassWrapVectorSynonym::AddRandomIntConstant(std::vector<uint32_t>& vec, uint32_t width, bool is_signed, RandomGenerator* random_generator) {
  auto multiplier = is_signed ? (random_generator->RandomBool() ? 1 : -1) : 1;
  // Make sure the random integer is not zero.
  auto random_int = multiplier * (random_generator->RandomUint32(1000) + 1);
  std::vector<uint32_t> words = fuzzerutil::IntToWords(random_int, width, is_signed);
  vec.emplace_back(FindOrCreateIntegerConstant(words,width, is_signed,false));
}

}  // namespace fuzz
}  // namespace spvtools
