// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_vector_shuffle.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationVectorShuffle::TransformationVectorShuffle(
    const spvtools::fuzz::protobufs::TransformationVectorShuffle& message)
    : message_(message) {}

TransformationVectorShuffle::TransformationVectorShuffle(
    const protobufs::InstructionDescriptor& instruction_to_insert_before,
    uint32_t fresh_id, uint32_t vector1, uint32_t vector2,
    const std::vector<uint32_t>& component) {
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
  message_.set_fresh_id(fresh_id);
  message_.set_vector1(vector1);
  message_.set_vector2(vector2);
  for (auto a_component : component) {
    message_.add_component(a_component);
  }
}

bool TransformationVectorShuffle::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  // The fresh id must not already be in use.
  if (!fuzzerutil::IsFreshId(context, message_.fresh_id())) {
    return false;
  }
  // The instruction before which the shuffle will be inserted must exist.
  auto instruction_to_insert_before =
      FindInstruction(message_.instruction_to_insert_before(), context);
  if (!instruction_to_insert_before) {
    return false;
  }
  // The first vector must be an instruction with a type id
  auto vector1_instruction =
      context->get_def_use_mgr()->GetDef(message_.vector1());
  if (!vector1_instruction || !vector1_instruction->type_id()) {
    return false;
  }
  // The second vector must be an instruction with a type id
  auto vector2_instruction =
      context->get_def_use_mgr()->GetDef(message_.vector2());
  if (!vector2_instruction || !vector2_instruction->type_id()) {
    return false;
  }
  auto vector1_type =
      context->get_type_mgr()->GetType(vector1_instruction->type_id());
  // The first vector instruction's type must actually be a vector type.
  if (!vector1_type->AsVector()) {
    return false;
  }
  auto vector2_type =
      context->get_type_mgr()->GetType(vector2_instruction->type_id());
  // The second vector instruction's type must actually be a vector type.
  if (!vector2_type->AsVector()) {
    return false;
  }
  // The element types of the vectors must be the same.
  if (vector1_type->AsVector()->element_type() !=
      vector2_type->AsVector()->element_type()) {
    return false;
  }
  uint32_t combined_size = vector1_type->AsVector()->element_count() +
                           vector2_type->AsVector()->element_count();
  for (auto a_compoment : message_.component()) {
    // 0xFFFFFFFF is used to represent an undefined component.  Unless
    // undefined, a component must be less than the combined size of the
    // vectors.
    if (a_compoment != 0xFFFFFFFF && a_compoment >= combined_size) {
      return false;
    }
  }
  // The module must already declare an appropriate type in which to store the
  // result of the shuffle.
  if (!GetResultTypeId(context, *vector1_type->AsVector()->element_type())) {
    return false;
  }
  // Each of the vectors used in the shuffle must be available at the insertion
  // point.
  for (auto used_instruction : {vector1_instruction, vector2_instruction}) {
    if (auto block = context->get_instr_block(used_instruction)) {
      if (!context->GetDominatorAnalysis(block->GetParent())
               ->Dominates(used_instruction, instruction_to_insert_before)) {
        return false;
      }
    }
  }

  // It must be legitimate to insert an OpVectorShuffle before the identified
  // instruction.
  return fuzzerutil::CanInsertOpcodeBeforeInstruction(
      SpvOpVectorShuffle, instruction_to_insert_before);
}

void TransformationVectorShuffle::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* fact_manager) const {
  // Get the types of the two input vectors.
  auto vector1_type =
      context->get_type_mgr()
          ->GetType(
              context->get_def_use_mgr()->GetDef(message_.vector1())->type_id())
          ->AsVector();
  auto vector2_type =
      context->get_type_mgr()
          ->GetType(
              context->get_def_use_mgr()->GetDef(message_.vector2())->type_id())
          ->AsVector();

  // Make input operands for a shuffle instruction - these comprise the two
  // vectors being shuffled, followed by the integer literal components.
  opt::Instruction::OperandList shuffle_operands = {
      {SPV_OPERAND_TYPE_ID, {message_.vector1()}},
      {SPV_OPERAND_TYPE_ID, {message_.vector2()}}};
  for (auto a_component : message_.component()) {
    shuffle_operands.push_back(
        {SPV_OPERAND_TYPE_LITERAL_INTEGER, {a_component}});
  }

  // Add a shuffle instruction right before the instruction identified by
  // |message_.instruction_to_insert_before|.
  FindInstruction(message_.instruction_to_insert_before(), context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          context, SpvOpVectorShuffle,
          GetResultTypeId(context, *vector1_type->element_type()),
          message_.fresh_id(), shuffle_operands));
  fuzzerutil::UpdateModuleIdBound(context, message_.fresh_id());

  // Add synonym facts relating the defined elements of the shuffle result to
  // the vector components that they come from.
  for (uint32_t component_index = 0;
       component_index < static_cast<uint32_t>(message_.component_size());
       component_index++) {
    uint32_t component = message_.component(component_index);
    if (component == 0xFFFFFFFF) {
      // This component is undefined.
      continue;
    }

    // This describes the element of the result vector associated with
    // |component_index|.
    protobufs::DataDescriptor descriptor_for_result_component =
        MakeDataDescriptor(message_.fresh_id(), {component_index}, 1);

    // Check which of the two input vectors |component| refers to - it is the
    // first input vector if and only if it is smaller than that vector's
    // element count.  In each case, make a synonym fact.
    if (component < vector1_type->element_count()) {
      fact_manager->AddFactDataSynonym(
          descriptor_for_result_component,
          MakeDataDescriptor(message_.vector1(), {component}, 1));
    } else {
      auto index_into_vector_2 = component - vector1_type->element_count();
      assert(index_into_vector_2 < vector2_type->element_count() &&
             "Vector shuffle index is out of bounds.");
      fact_manager->AddFactDataSynonym(
          descriptor_for_result_component,
          MakeDataDescriptor(message_.vector2(), {index_into_vector_2}, 1));
    }
  }

  // If all the components are contiguous and refer to the same input vector,
  // the result vector is synonymous with a slice of the input vector.
  bool slice_possible = true;       // We set this to false if we find that the
                                    // conditions required to make the result
                                    // synonymous with a contiguous slice do not
                                    // hold.
  uint32_t previous_component = 0;  // We use this to determine whether we have
                                    // a contiguous range of components, and
                                    // whether these components cross between
                                    // the two input vectors.

  // Iterate through the components.
  for (uint32_t component_index = 0;
       component_index < static_cast<uint32_t>(message_.component_size());
       component_index++) {
    uint32_t component = message_.component(component_index);
    if (component == 0xFFFFFFFF) {
      // If a component of the shuffle is undefined, the result vector cannot be
      // viewed as a slice into one of the input vectors
      slice_possible = false;
      break;
    }
    // If this is the first component, there is hope that it could be the
    // beginning of a contiguous range.  Otherwise, if the component does not
    // immediately follow the last seen component, or if it crosses between
    // the two input vectors, we know that we do not have a contiguous range
    // confined to a single vector.
    if (component_index != 0 &&
        (component != previous_component + 1 ||
         (previous_component < vector1_type->element_count() &&
          component >= vector1_type->element_count()))) {
      // Either the components are not contiguous, or they cross an input vector
      // boundary.
      slice_possible = false;
      break;
    }
    previous_component = component;
  }
  if (slice_possible) {
    // The conditions to record a synonym fact relating the result vector
    // to a contiguous slice of one of the input vectors hold, so record the
    // fact.
    uint32_t relevant_input_vector =
        message_.component(0) < vector1_type->element_count()
            ? message_.vector1()
            : message_.vector2();
    fact_manager->AddFactDataSynonym(
        MakeDataDescriptor(message_.fresh_id(), {}, 1),
        MakeDataDescriptor(relevant_input_vector, {message_.component(0)},
                           static_cast<uint32_t>(message_.component_size())));
  }
}

protobufs::Transformation TransformationVectorShuffle::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_vector_shuffle() = message_;
  return result;
}

uint32_t TransformationVectorShuffle::GetResultTypeId(
    opt::IRContext* context, const opt::analysis::Type& element_type) const {
  opt::analysis::Vector result_type(
      &element_type, static_cast<uint32_t>(message_.component_size()));
  return context->get_type_mgr()->GetId(&result_type);
}

}  // namespace fuzz
}  // namespace spvtools
