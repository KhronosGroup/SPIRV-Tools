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
#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_wrap_vector_synonym.h"

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
        if (!valid_arithmetic_types.count(instruction_iterator->opcode()))
          return;

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
        uint32_t operand_type_id = instruction_iterator->type_id();

        // Get a random vector size from 2 to 4.
        uint32_t component_count = GetFuzzerContext()->GetRandomIntegerFromRange(2,4);

        // Randomly choose a position that target ids should be placed at.
        // The position is in range [0, n - 1], where n is the size of the vector.
        uint32_t position = GetFuzzerContext()->GetRandomIntegerFromRange(0, component_count - 1);

        // Target ids are the two scalar ids from the original instruction.
        uint32_t target_id1 = instruction_iterator->GetSingleWordInOperand(0);
        uint32_t target_id2 = instruction_iterator->GetSingleWordInOperand(1);

        // Stores the ids of scalar constants.
        std::vector<uint32_t> vec1_components;
        std::vector<uint32_t> vec2_components;

        // Width is specified in the first index for either OpTypeInt or
        // OpTypeFloat.
        uint32_t width = GetIRContext()
                             ->get_def_use_mgr()
                             ->GetDef(operand_type_id)
                             ->GetSingleWordOperand(0);
        // Get the scalar type.
        auto type = GetIRContext()->get_type_mgr()->GetType(operand_type_id);
        // Whether the constant is signed, not used for float type so default
        // set to true.
        bool is_signed_constant =
            type->AsInteger() ? type->AsInteger()->IsSigned() : true;

        // Populate components based on vector type and size.
        for (uint32_t i = 0; i < component_count; ++i) {
          if (i == position) {
            vec1_components.emplace_back(target_id1);
            vec2_components.emplace_back(target_id2);
          } else {
            if (type->AsInteger()) {
              // Operands are integers. Add random integers to each vector.
              int sign1 = is_signed_constant ? (GetFuzzerContext()->ChooseEven() ? 1 : -1) : 1;
              int sign2 = is_signed_constant ? (GetFuzzerContext()->ChooseEven() ? 1 : -1) : 1;
              int random_int1 = sign1 * (GetFuzzerContext()->GetRandomIntegerFromRange(1, 100));
              int random_int2 = sign2 * (GetFuzzerContext()->GetRandomIntegerFromRange(1, 100));
              vec1_components.emplace_back(FindOrCreateIntegerConstant(fuzzerutil::IntToWords(random_int1, width, is_signed_constant) ,width, is_signed_constant,false));
              vec2_components.emplace_back(FindOrCreateIntegerConstant(fuzzerutil::IntToWords(random_int2, width, is_signed_constant) ,width, is_signed_constant,false));
            } else if (type->AsFloat()) {
              // Operands are floats. Add random floats to each vector.
              float sign1 = GetFuzzerContext()->ChooseEven() ? 1.0f : -1.0f;
              float sign2 = GetFuzzerContext()->ChooseEven() ? 1.0f : -1.0f;
              float random_float1 = sign1 * GetFuzzerContext()->GetRandomFloatFromRange(0.1f, 10.0f);
              float random_float2 = sign2 * GetFuzzerContext()->GetRandomFloatFromRange(0.1f, 10.0f);
              vec1_components.emplace_back(FindOrCreateFloatConstant({fuzzerutil::FloatToWord(random_float1)}, width, false));
              vec2_components.emplace_back(FindOrCreateFloatConstant({fuzzerutil::FloatToWord(random_float2)}, width, false));
            } else {
              assert(false && "Instruction opcode must be a valid numeric constant type.");
            }
          }
        }
        // Add two OpCompositeConstruct to the module with result id returned.
        // Add the first OpCompositeConstruct that wraps the id of the first operand.
        uint32_t result_id1 = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationCompositeConstruct(operand_type_id, vec1_components,
                                                             instruction_descriptor, result_id1));

        // Add the second OpCompositeConstruct that wraps the id of the second operand.
        uint32_t result_id2 = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationCompositeConstruct(operand_type_id, vec2_components,
                                                             instruction_descriptor, result_id2));

        // Add synonym facts between original operands and id from the |scalar_position| of the vectors created.
        GetTransformationContext()->GetFactManager()->AddFactDataSynonym(
            MakeDataDescriptor(result_id1, {position}),
            MakeDataDescriptor(instruction_iterator->GetSingleWordInOperand(0), {}));

        GetTransformationContext()->GetFactManager()->AddFactDataSynonym(
            MakeDataDescriptor(result_id2, {position}),
            MakeDataDescriptor(instruction_iterator->GetSingleWordInOperand(1), {}));

        // Apply transformation to do vector operation and add synonym between the result
        // vector id and the id of the original instruction.
        ApplyTransformation(TransformationWrapVectorSynonym(instruction_iterator->result_id(), result_id1,
                                                            result_id2, GetFuzzerContext()->GetFreshId(), position));
      });
}

}  // namespace fuzz
}  // namespace spvtools
