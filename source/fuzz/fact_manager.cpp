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

namespace spvtools {
namespace fuzz {

struct FactManager::UniformConstantFacts {
  const opt::analysis::Type* FindOrRegisterType(
      const opt::analysis::Type* type);

  const opt::analysis::ScalarConstant* FindOrRegisterConstant(
      const opt::analysis::ScalarConstant* constant);

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
FactManager::UniformConstantFacts::FindOrRegisterType(
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
FactManager::UniformConstantFacts::FindOrRegisterConstant(
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

void FactManager::UniformConstantFacts::AddUniformConstantFact(
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

void FactManager::UniformConstantFacts::AddUniformFloatValueFact(
    uint32_t width, std::vector<uint32_t>&& data,
    protobufs::UniformBufferElementDescriptor descriptor) {
  opt::analysis::Float float_type = opt::analysis::Float(width);
  opt::analysis::FloatConstant float_constant = opt::analysis::FloatConstant(
      FindOrRegisterType(&float_type)->AsFloat(), data);
  AddUniformConstantFact(&float_constant, std::move(descriptor));
}

void FactManager::UniformConstantFacts::AddUniformIntValueFact(
    uint32_t width, bool is_signed, std::vector<uint32_t>&& data,
    protobufs::UniformBufferElementDescriptor descriptor) {
  opt::analysis::Integer integer_type =
      opt::analysis::Integer(width, is_signed);
  opt::analysis::IntConstant int_constant = opt::analysis::IntConstant(
      FindOrRegisterType(&integer_type)->AsInteger(), data);
  AddUniformConstantFact(&int_constant, std::move(descriptor));
}

std::vector<const opt::analysis::ScalarConstant*>
FactManager::UniformConstantFacts::GetConstantsAvailableFromUniformsForType(
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
FactManager::UniformConstantFacts::GetUniformDescriptorsForConstant(
    const opt::analysis::ScalarConstant& constant) const {
  assert(constant_pool.find(&constant) != constant_pool.end());
  if (constant_to_uniform_descriptors.find(&constant) ==
      constant_to_uniform_descriptors.end()) {
    return nullptr;
  }
  return &constant_to_uniform_descriptors.find(&constant)->second;
}

const opt::analysis::ScalarConstant*
FactManager::UniformConstantFacts::GetConstantFromUniformDescriptor(
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  if (uniform_descriptor_to_constant.find(&uniform_descriptor) ==
      uniform_descriptor_to_constant.end()) {
    return nullptr;
  }
  return uniform_descriptor_to_constant.at(&uniform_descriptor);
}

std::vector<const opt::analysis::Type*>
FactManager::UniformConstantFacts::GetTypesForWhichUniformValuesAreKnown()
    const {
  std::vector<const opt::analysis::Type*> result;
  // Iterate through the sequence of owned types, rather than the unordered type
  // pool, so that the order of the resulting types is deterministic.
  for (auto& type : owned_types) {
    result.push_back(type.get());
  }
  return result;
}

FactManager::FactManager() {
  uniform_constant_facts_ = MakeUnique<UniformConstantFacts>();
}

FactManager::~FactManager() = default;

void FactManager::AddUniformFloatValueFact(
    uint32_t width, std::vector<uint32_t>&& data,
    protobufs::UniformBufferElementDescriptor descriptor) {
  uniform_constant_facts_->AddUniformFloatValueFact(width, std::move(data),
                                                    descriptor);
}

void FactManager::AddUniformIntValueFact(
    uint32_t width, bool is_signed, std::vector<uint32_t>&& data,
    protobufs::UniformBufferElementDescriptor descriptor) {
  uniform_constant_facts_->AddUniformIntValueFact(width, is_signed,
                                                  std::move(data), descriptor);
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
