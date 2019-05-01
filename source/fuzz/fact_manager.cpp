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

namespace spvtools {
namespace fuzz {

const opt::analysis::Type* FactManager::FindOrRegisterType(
    const opt::analysis::Type* type) {
  auto type_pool_iterator = type_pool_.find(type);
  if (type_pool_iterator != type_pool_.end()) {
    return *type_pool_iterator;
  }
  auto cloned_type = type->Clone();
  auto result = cloned_type.get();
  owned_types_.push_back(std::move(cloned_type));
  type_pool_.insert(result);
  return result;
}

const opt::analysis::Constant* FactManager::FindOrRegisterConstant(
    const opt::analysis::Constant* constant) {
  assert(type_pool_.find(constant->type()) != type_pool_.end());
  auto constant_pool_iterator = constant_pool_.find(constant);
  if (constant_pool_iterator != constant_pool_.end()) {
    return *constant_pool_iterator;
  }
  auto cloned_constant = constant->Copy();
  auto result = cloned_constant.get();
  owned_constants_.push_back(std::move(cloned_constant));
  constant_pool_.insert(result);
  return result;
}

void FactManager::AddUniformConstantFact(
    const opt::analysis::Constant* constant,
    protobufs::UniformBufferElementDescriptor descriptor) {
  auto registered_constant = FindOrRegisterConstant(constant);
  if (constant_to_uniform_descriptors_.find(registered_constant) ==
      constant_to_uniform_descriptors_.end()) {
    constant_to_uniform_descriptors_[registered_constant] =
        std::vector<protobufs::UniformBufferElementDescriptor>();
  }
  constant_to_uniform_descriptors_.find(registered_constant)
      ->second.push_back(descriptor);
  const protobufs::UniformBufferElementDescriptor* descriptor_ptr =
      &constant_to_uniform_descriptors_.find(registered_constant)
           ->second.back();
  assert(uniform_descriptor_to_constant_.find(descriptor_ptr) ==
         uniform_descriptor_to_constant_.end());
  uniform_descriptor_to_constant_[descriptor_ptr] = registered_constant;
}

void FactManager::AddUniformFloatValueFact(
    uint32_t width, std::vector<uint32_t>&& data,
    protobufs::UniformBufferElementDescriptor descriptor) {
  opt::analysis::Float float_type = opt::analysis::Float(width);
  opt::analysis::FloatConstant float_constant = opt::analysis::FloatConstant(
      FindOrRegisterType(&float_type)->AsFloat(), data);
  AddUniformConstantFact(&float_constant, std::move(descriptor));
}

void FactManager::AddUniformIntValueFact(
    uint32_t width, bool is_signed, std::vector<uint32_t>&& data,
    protobufs::UniformBufferElementDescriptor descriptor) {
  opt::analysis::Integer integer_type =
      opt::analysis::Integer(width, is_signed);
  opt::analysis::IntConstant int_constant = opt::analysis::IntConstant(
      FindOrRegisterType(&integer_type)->AsInteger(), data);
  AddUniformConstantFact(&int_constant, std::move(descriptor));
}

std::vector<const opt::analysis::Constant*>
FactManager::ConstantsAvailableFromUniformsForType(
    const spvtools::opt::analysis::Type& type) {
  std::vector<const opt::analysis::Constant*> result;
  auto iterator = type_pool_.find(&type);
  if (iterator == type_pool_.end()) {
    return result;
  }
  auto registered_type = *iterator;
  for (auto& constant : owned_constants_) {
    if (constant->type() == registered_type) {
      result.push_back(constant.get());
    }
  }
  return result;
}

const std::vector<protobufs::UniformBufferElementDescriptor>*
FactManager::GetUniformDescriptorsForConstant(
    const opt::analysis::Constant& constant) {
  assert(constant_pool_.find(&constant) != constant_pool_.end());
  if (constant_to_uniform_descriptors_.find(&constant) ==
      constant_to_uniform_descriptors_.end()) {
    return nullptr;
  }
  return &constant_to_uniform_descriptors_.find(&constant)->second;
}

const opt::analysis::Constant* FactManager::GetConstantFromUniformDescriptor(
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  if (uniform_descriptor_to_constant_.find(&uniform_descriptor) ==
      uniform_descriptor_to_constant_.end()) {
    return nullptr;
  }
  return uniform_descriptor_to_constant_.at(&uniform_descriptor);
}

protobufs::UniformBufferElementDescriptor
FactManager::MakeUniformBufferElementDescriptor(
    uint32_t uniform_variable_id, std::vector<uint32_t>&& indices) {
  protobufs::UniformBufferElementDescriptor result;
  result.set_uniform_variable_id(uniform_variable_id);
  for (auto index : indices) {
    result.add_index(index);
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
