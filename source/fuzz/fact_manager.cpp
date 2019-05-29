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

#include "source/fuzz/fact_manager.h"

#include <utility>

#include "source/fuzz/uniform_buffer_element_descriptor.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// The purpose of this struct is to group the fields and data used to represent
// facts about uniform constants.
struct FactManager::ConstantUniformFacts {
  bool AddFact(const protobufs::ConstantUniformFact& fact,
               opt::IRContext* context);

  std::vector<uint32_t> GetConstantsAvailableFromUniformsForType(
      opt::IRContext* ir_context, uint32_t type_id) const;

  const std::vector<protobufs::UniformBufferElementDescriptor>
  GetUniformDescriptorsForConstant(opt::IRContext* ir_context,
                                   uint32_t constant_id) const;

  uint32_t GetConstantFromUniformDescriptor(
      opt::IRContext* context,
      const protobufs::UniformBufferElementDescriptor& uniform_descriptor)
      const;

  std::vector<uint32_t> GetTypesForWhichUniformValuesAreKnown() const;

  bool DataMatches(
      const opt::Instruction& constant_instruction,
      const protobufs::ConstantUniformFact& constant_uniform_fact) const;

  std::vector<std::pair<protobufs::ConstantUniformFact, uint32_t>>
      facts_and_type_ids;
};

bool FactManager::ConstantUniformFacts::DataMatches(
    const opt::Instruction& constant_instruction,
    const protobufs::ConstantUniformFact& constant_uniform_fact) const {
  assert(constant_instruction.opcode() == SpvOpConstant);
  std::vector<uint32_t> data_in_constant;
  for (uint32_t i = 0; i < constant_instruction.NumInOperands(); i++) {
    data_in_constant.push_back(constant_instruction.GetSingleWordInOperand(i));
  }
  std::vector<uint32_t> data_in_fact;
  for (auto word : constant_uniform_fact.constant_word()) {
    data_in_fact.push_back(word);
  }
  return data_in_constant == data_in_fact;
}

std::vector<uint32_t>
FactManager::ConstantUniformFacts::GetConstantsAvailableFromUniformsForType(
    opt::IRContext* ir_context, uint32_t type_id) const {
  std::vector<uint32_t> result;
  for (auto& type_or_value : ir_context->module()->types_values()) {
    if (type_or_value.opcode() == SpvOpConstant &&
        type_or_value.type_id() == type_id) {
      for (auto& fact_and_type_id : facts_and_type_ids) {
        if (fact_and_type_id.second == type_id) {
          if (DataMatches(type_or_value, fact_and_type_id.first)) {
            result.push_back(type_or_value.result_id());
            break;  // Having added it once, we don't want to add it again.
          }
        }
      }
    }
  }
  return result;
}

const std::vector<protobufs::UniformBufferElementDescriptor>
FactManager::ConstantUniformFacts::GetUniformDescriptorsForConstant(
    opt::IRContext* ir_context, uint32_t constant_id) const {
  std::vector<protobufs::UniformBufferElementDescriptor> result;
  auto constant_inst = ir_context->get_def_use_mgr()->GetDef(constant_id);
  assert(constant_inst->opcode() == SpvOpConstant &&
         "The given id must be that of a constant");
  auto type_id = constant_inst->type_id();
  for (auto& fact_and_type_id : facts_and_type_ids) {
    if (fact_and_type_id.second != type_id) {
      continue;
    }
    if (DataMatches(*constant_inst, fact_and_type_id.first)) {
      result.emplace_back(
          fact_and_type_id.first.uniform_buffer_element_descriptor());
    }
  }
  return result;
}

uint32_t FactManager::ConstantUniformFacts::GetConstantFromUniformDescriptor(
    opt::IRContext* context,
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  for (auto& fact_and_type : facts_and_type_ids) {
    if (UniformBufferElementDescriptorEquals()(
            &uniform_descriptor,
            &fact_and_type.first.uniform_buffer_element_descriptor())) {
      for (auto& type_or_value : context->module()->types_values()) {
        if (type_or_value.opcode() == SpvOpConstant &&
            type_or_value.type_id() == fact_and_type.second) {
          if (DataMatches(type_or_value, fact_and_type.first)) {
            return type_or_value.result_id();
          }
        }
      }
    }
  }
  return 0;
}

std::vector<uint32_t>
FactManager::ConstantUniformFacts::GetTypesForWhichUniformValuesAreKnown()
    const {
  std::vector<uint32_t> result;
  for (auto& fact_and_type : facts_and_type_ids) {
    if (std::find(result.begin(), result.end(), fact_and_type.second) ==
        result.end()) {
      result.push_back(fact_and_type.second);
    }
  }
  return result;
}

bool FactManager::ConstantUniformFacts::AddFact(
    const protobufs::ConstantUniformFact& fact, opt::IRContext* context) {
  auto should_be_uniform_variable = context->get_def_use_mgr()->GetDef(
      fact.uniform_buffer_element_descriptor().uniform_variable_id());
  if (!should_be_uniform_variable) {
    return false;
  }
  if (SpvOpVariable != should_be_uniform_variable->opcode()) {
    return false;
  }
  if (SpvStorageClassUniform !=
      should_be_uniform_variable->GetSingleWordInOperand(0)) {
    return false;
  }
  auto should_be_uniform_pointer_type =
      context->get_type_mgr()->GetType(should_be_uniform_variable->type_id());
  if (!should_be_uniform_pointer_type->AsPointer()) {
    return false;
  }
  if (should_be_uniform_pointer_type->AsPointer()->storage_class() !=
      SpvStorageClassUniform) {
    return false;
  }
  auto should_be_uniform_pointer_instruction =
      context->get_def_use_mgr()->GetDef(should_be_uniform_variable->type_id());
  auto element_type =
      should_be_uniform_pointer_instruction->GetSingleWordInOperand(1);

  for (auto index : fact.uniform_buffer_element_descriptor().index()) {
    auto should_be_composite_type =
        context->get_def_use_mgr()->GetDef(element_type);
    if (SpvOpTypeStruct == should_be_composite_type->opcode()) {
      if (index >= should_be_composite_type->NumInOperands()) {
        return false;
      }
      element_type = should_be_composite_type->GetSingleWordInOperand(index);
    } else if (SpvOpTypeArray == should_be_composite_type->opcode()) {
      auto array_length_constant =
          context->get_constant_mgr()
              ->GetConstantFromInst(context->get_def_use_mgr()->GetDef(
                  should_be_composite_type->GetSingleWordInOperand(1)))
              ->AsIntConstant();
      if (array_length_constant->words().size() != 1) {
        return false;
      }
      auto array_length = array_length_constant->GetU32();
      if (index >= array_length) {
        return false;
      }
      element_type = should_be_composite_type->GetSingleWordInOperand(0);
    } else if (SpvOpTypeVector == should_be_composite_type->opcode()) {
      auto vector_length = should_be_composite_type->GetSingleWordInOperand(1);
      if (index >= vector_length) {
        return false;
      }
      element_type = should_be_composite_type->GetSingleWordInOperand(0);
    } else {
      return false;
    }
  }
  auto final_element_type = context->get_type_mgr()->GetType(element_type);
  if (!(final_element_type->AsFloat() || final_element_type->AsInteger())) {
    return false;
  }
  auto width = final_element_type->AsFloat()
                   ? final_element_type->AsFloat()->width()
                   : final_element_type->AsInteger()->width();
  auto required_words = (width + 32 - 1) / 32;
  if (static_cast<uint32_t>(fact.constant_word().size()) != required_words) {
    return false;
  }
  facts_and_type_ids.emplace_back(
      std::pair<protobufs::ConstantUniformFact, uint32_t>(fact, element_type));
  return true;
}

FactManager::FactManager() {
  uniform_constant_facts_ = MakeUnique<ConstantUniformFacts>();
}

FactManager::~FactManager() = default;

bool FactManager::AddFacts(const protobufs::FactSequence& initial_facts,
                           opt::IRContext* context) {
  for (auto& fact : initial_facts.fact()) {
    if (!AddFact(fact, context)) {
      // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2621) Provide
      //  information about the fact that could not be added.
      return false;
    }
  }
  return true;
}

bool FactManager::AddFact(const spvtools::fuzz::protobufs::Fact& fact,
                          spvtools::opt::IRContext* context) {
  assert(fact.fact_case() == protobufs::Fact::kConstantUniformFact &&
         "Right now this is the only fact.");
  if (!uniform_constant_facts_->AddFact(fact.constant_uniform_fact(),
                                        context)) {
    return false;
  }
  return true;
}

std::vector<uint32_t> FactManager::GetConstantsAvailableFromUniformsForType(
    opt::IRContext* ir_context, uint32_t type_id) const {
  return uniform_constant_facts_->GetConstantsAvailableFromUniformsForType(
      ir_context, type_id);
}

const std::vector<protobufs::UniformBufferElementDescriptor>
FactManager::GetUniformDescriptorsForConstant(opt::IRContext* ir_context,
                                              uint32_t constant_id) const {
  return uniform_constant_facts_->GetUniformDescriptorsForConstant(ir_context,
                                                                   constant_id);
}

uint32_t FactManager::GetConstantFromUniformDescriptor(
    opt::IRContext* context,
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  return uniform_constant_facts_->GetConstantFromUniformDescriptor(
      context, uniform_descriptor);
}

std::vector<uint32_t> FactManager::GetTypesForWhichUniformValuesAreKnown()
    const {
  return uniform_constant_facts_->GetTypesForWhichUniformValuesAreKnown();
}

const std::vector<std::pair<protobufs::ConstantUniformFact, uint32_t>>&
FactManager::GetConstantUniformFacts() const {
  return uniform_constant_facts_->facts_and_type_ids;
}

}  // namespace fuzz
}  // namespace spvtools
