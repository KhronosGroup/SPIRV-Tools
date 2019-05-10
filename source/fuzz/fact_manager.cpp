#include <utility>

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
#include "source/fuzz/uniform_buffer_element_descriptor.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// The purpose of this struct is to group the fields and data used to represent
// facts about uniform constants.
struct FactManager::ConstantUniformFacts {
  const opt::analysis::Type* FindOrRegisterType(
      const opt::analysis::Type* type);

  const opt::analysis::ScalarConstant* FindOrRegisterConstant(
      const opt::analysis::ScalarConstant* constant);

  bool AddFact(const protobufs::ConstantUniformFact& fact,
               opt::IRContext* context);

  void AddUniformConstantFact(
      const opt::analysis::ScalarConstant* constant,
      protobufs::UniformBufferElementDescriptor descriptor);

  void AddUniformFloatValueFact(
      uint32_t width, std::vector<uint32_t>&& data,
      protobufs::UniformBufferElementDescriptor descriptor);

  void AddUniformIntValueFact(
      uint32_t width, bool is_signed, std::vector<uint32_t>&& data,
      protobufs::UniformBufferElementDescriptor descriptor);

  std::vector<const opt::analysis::ScalarConstant*>
  GetConstantsAvailableFromUniformsForType(
      const spvtools::opt::analysis::Type& type) const;

  const std::vector<protobufs::UniformBufferElementDescriptor>*
  GetUniformDescriptorsForConstant(
      const opt::analysis::ScalarConstant& constant) const;

  const opt::analysis::ScalarConstant* GetConstantFromUniformDescriptor(
      const protobufs::UniformBufferElementDescriptor& uniform_descriptor)
      const;

  std::vector<const opt::analysis::Type*>
  GetTypesForWhichUniformValuesAreKnown() const;

  std::unordered_set<const opt::analysis::Type*, opt::analysis::HashTypePointer,
                     opt::analysis::CompareTypePointers>
      type_pool;

  std::vector<std::unique_ptr<opt::analysis::Type>> owned_types;

  std::unordered_set<const opt::analysis::ScalarConstant*,
                     opt::analysis::ConstantHash, opt::analysis::ConstantEqual>
      constant_pool;

  std::vector<std::unique_ptr<opt::analysis::ScalarConstant>> owned_constants;

  std::map<const opt::analysis::ScalarConstant*,
           std::vector<protobufs::UniformBufferElementDescriptor>>
      constant_to_uniform_descriptors;

  std::unordered_map<const protobufs::UniformBufferElementDescriptor*,
                     const opt::analysis::ScalarConstant*,
                     UniformBufferElementDescriptorHash,
                     UniformBufferElementDescriptorEquals>
      uniform_descriptor_to_constant;
};

const opt::analysis::Type*
FactManager::ConstantUniformFacts::FindOrRegisterType(
    const opt::analysis::Type* type) {
  auto type_pool_iterator = type_pool.find(type);
  if (type_pool_iterator != type_pool.end()) {
    return *type_pool_iterator;
  }
  auto cloned_type = type->Clone();
  auto result = cloned_type.get();
  owned_types.push_back(std::move(cloned_type));
  type_pool.insert(result);
  return result;
}

const opt::analysis::ScalarConstant*
FactManager::ConstantUniformFacts::FindOrRegisterConstant(
    const opt::analysis::ScalarConstant* constant) {
  assert(type_pool.find(constant->type()) != type_pool.end());
  auto constant_pool_iterator = constant_pool.find(constant);
  if (constant_pool_iterator != constant_pool.end()) {
    return *constant_pool_iterator;
  }
  auto cloned_constant = std::unique_ptr<opt::analysis::ScalarConstant>(
      constant->Copy().release()->AsScalarConstant());
  auto result = cloned_constant.get();
  owned_constants.push_back(std::move(cloned_constant));
  constant_pool.insert(result);
  return result;
}

void FactManager::ConstantUniformFacts::AddUniformConstantFact(
    const opt::analysis::ScalarConstant* constant,
    protobufs::UniformBufferElementDescriptor descriptor) {
  auto registered_constant = FindOrRegisterConstant(constant);
  if (constant_to_uniform_descriptors.find(registered_constant) ==
      constant_to_uniform_descriptors.end()) {
    constant_to_uniform_descriptors[registered_constant] =
        std::vector<protobufs::UniformBufferElementDescriptor>();
  }
  constant_to_uniform_descriptors.find(registered_constant)
      ->second.push_back(descriptor);
  const protobufs::UniformBufferElementDescriptor* descriptor_ptr =
      &constant_to_uniform_descriptors.find(registered_constant)->second.back();
  assert(uniform_descriptor_to_constant.find(descriptor_ptr) ==
         uniform_descriptor_to_constant.end());
  uniform_descriptor_to_constant[descriptor_ptr] = registered_constant;
}

void FactManager::ConstantUniformFacts::AddUniformFloatValueFact(
    uint32_t width, std::vector<uint32_t>&& data,
    protobufs::UniformBufferElementDescriptor descriptor) {
  opt::analysis::Float float_type = opt::analysis::Float(width);
  opt::analysis::FloatConstant float_constant = opt::analysis::FloatConstant(
      FindOrRegisterType(&float_type)->AsFloat(), data);
  AddUniformConstantFact(&float_constant, std::move(descriptor));
}

void FactManager::ConstantUniformFacts::AddUniformIntValueFact(
    uint32_t width, bool is_signed, std::vector<uint32_t>&& data,
    protobufs::UniformBufferElementDescriptor descriptor) {
  opt::analysis::Integer integer_type =
      opt::analysis::Integer(width, is_signed);
  opt::analysis::IntConstant int_constant = opt::analysis::IntConstant(
      FindOrRegisterType(&integer_type)->AsInteger(), data);
  AddUniformConstantFact(&int_constant, std::move(descriptor));
}

std::vector<const opt::analysis::ScalarConstant*>
FactManager::ConstantUniformFacts::GetConstantsAvailableFromUniformsForType(
    const spvtools::opt::analysis::Type& type) const {
  std::vector<const opt::analysis::ScalarConstant*> result;
  auto iterator = type_pool.find(&type);
  if (iterator == type_pool.end()) {
    return result;
  }
  auto registered_type = *iterator;
  for (auto& constant : owned_constants) {
    if (constant->type() == registered_type) {
      result.push_back(constant.get());
    }
  }
  return result;
}

const std::vector<protobufs::UniformBufferElementDescriptor>*
FactManager::ConstantUniformFacts::GetUniformDescriptorsForConstant(
    const opt::analysis::ScalarConstant& constant) const {
  auto registered_type = type_pool.find(constant.type());
  if (registered_type == type_pool.end()) {
    return nullptr;
  }
  const opt::analysis::ScalarConstant* registered_constant;
  if (constant.AsFloatConstant()) {
    opt::analysis::FloatConstant temp((*registered_type)->AsFloat(),
                                      constant.words());
    auto iterator = constant_pool.find(&temp);
    if (iterator == constant_pool.end()) {
      return nullptr;
    }
    registered_constant = *iterator;
  } else if (constant.AsIntConstant()) {
    opt::analysis::IntConstant temp((*registered_type)->AsInteger(),
                                    constant.words());
    auto iterator = constant_pool.find(&temp);
    if (iterator == constant_pool.end()) {
      return nullptr;
    }
    registered_constant = *iterator;
  } else {
    return nullptr;
  }
  return &constant_to_uniform_descriptors.find(registered_constant)->second;
}

const opt::analysis::ScalarConstant*
FactManager::ConstantUniformFacts::GetConstantFromUniformDescriptor(
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  if (uniform_descriptor_to_constant.find(&uniform_descriptor) ==
      uniform_descriptor_to_constant.end()) {
    return nullptr;
  }
  return uniform_descriptor_to_constant.at(&uniform_descriptor);
}

std::vector<const opt::analysis::Type*>
FactManager::ConstantUniformFacts::GetTypesForWhichUniformValuesAreKnown()
    const {
  std::vector<const opt::analysis::Type*> result;
  // Iterate through the sequence of owned types, rather than the unordered type
  // pool, so that the order of the resulting types is deterministic.
  for (auto& type : owned_types) {
    result.push_back(type.get());
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
  if ((uint32_t)fact.constant_word().size() != required_words) {
    return false;
  }
  std::vector<uint32_t> data;
  for (auto word : fact.constant_word()) {
    data.push_back(word);
  }
  if (final_element_type->AsFloat()) {
    AddUniformFloatValueFact(final_element_type->AsFloat()->width(),
                             std::move(data),
                             fact.uniform_buffer_element_descriptor());
  } else {
    AddUniformIntValueFact(final_element_type->AsInteger()->width(),
                           final_element_type->AsInteger()->IsSigned(),
                           std::move(data),
                           fact.uniform_buffer_element_descriptor());
  }
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

std::vector<const opt::analysis::ScalarConstant*>
FactManager::GetConstantsAvailableFromUniformsForType(
    const spvtools::opt::analysis::Type& type) const {
  return uniform_constant_facts_->GetConstantsAvailableFromUniformsForType(
      type);
}

const std::vector<protobufs::UniformBufferElementDescriptor>*
FactManager::GetUniformDescriptorsForConstant(
    const opt::analysis::ScalarConstant& constant) const {
  return uniform_constant_facts_->GetUniformDescriptorsForConstant(constant);
}

const opt::analysis::ScalarConstant*
FactManager::GetConstantFromUniformDescriptor(
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  return uniform_constant_facts_->GetConstantFromUniformDescriptor(
      uniform_descriptor);
}

std::vector<const opt::analysis::Type*>
FactManager::GetTypesForWhichUniformValuesAreKnown() const {
  return uniform_constant_facts_->GetTypesForWhichUniformValuesAreKnown();
}

}  // namespace fuzz
}  // namespace spvtools
