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

#ifndef SOURCE_FUZZ_FACT_MANAGER_H_
#define SOURCE_FUZZ_FACT_MANAGER_H_

#include "source/fuzz/protobufs/spirvfuzz.pb.h"
#include "source/opt/constants.h"

namespace spvtools {
namespace fuzz {

// Keeps track of facts about the module being transformed on which the fuzzing
// process can depend. Some initial facts can be provided, for example about
// guarantees on the values of inputs to SPIR-V entry points. Transformations
// may then rely on these facts, can add further facts that they establish.
// Facts are intended to be simple properties that either cannot be deduced from
// the module (such as properties that are guaranteed to hold for entry point
// inputs), or that are established by transformations, likely to be useful for
// future transformations, and not completely trivial to deduce straight from
// the module.
class FactManager {
 public:
  static protobufs::UniformBufferElementDescriptor
  MakeUniformBufferElementDescriptor(uint32_t uniform_variable_id,
                                     std::vector<uint32_t>&& indices);

  struct UniformBufferElementDescriptorHash {
    size_t operator()(
        const protobufs::UniformBufferElementDescriptor* descriptor) const {
      std::u32string data;
      data.push_back(descriptor->uniform_variable_id());
      for (auto id : descriptor->indices()) {
        data.push_back(id);
      }
      return std::hash<std::u32string>()(data);
    }
  };

  struct UniformBufferElementDescriptorEquals {
    bool operator()(
        const protobufs::UniformBufferElementDescriptor* first,
        const protobufs::UniformBufferElementDescriptor* second) const {
      return first->uniform_variable_id() == second->uniform_variable_id() &&
             std::equal(first->indices().begin(), first->indices().end(),
                        second->indices().begin());
    }
  };

  FactManager() = default;

  virtual ~FactManager() = default;

  void AddUniformFloatValueFact(
      uint32_t width, std::vector<uint32_t>&& data,
      protobufs::UniformBufferElementDescriptor descriptor);
  void AddUniformIntValueFact(
      uint32_t width, bool is_signed, std::vector<uint32_t>&& data,
      protobufs::UniformBufferElementDescriptor descriptor);

  std::vector<const opt::analysis::Constant*>
  ConstantsAvailableFromUniformsForType(const opt::analysis::Type& type);

  const std::vector<protobufs::UniformBufferElementDescriptor>*
  GetUniformDescriptorsForConstant(const opt::analysis::Constant& constant);

  const opt::analysis::Constant* GetConstantFromUniformDescriptor(
      const protobufs::UniformBufferElementDescriptor& uniform_descriptor)
      const;

 private:
  const opt::analysis::Type* FindOrRegisterType(
      const opt::analysis::Type* type);
  const opt::analysis::Constant* FindOrRegisterConstant(
      const opt::analysis::Constant* constant);
  void AddUniformConstantFact(
      const opt::analysis::Constant* constant,
      protobufs::UniformBufferElementDescriptor descriptor);

  std::unordered_set<const opt::analysis::Type*, opt::analysis::HashTypePointer,
                     opt::analysis::CompareTypePointers>
      type_pool_;
  std::vector<std::unique_ptr<opt::analysis::Type>> owned_types_;

  std::unordered_set<const opt::analysis::Constant*,
                     opt::analysis::ConstantHash, opt::analysis::ConstantEqual>
      constant_pool_;
  std::vector<std::unique_ptr<opt::analysis::Constant>> owned_constants_;

  std::map<const opt::analysis::Constant*,
           std::vector<protobufs::UniformBufferElementDescriptor>>
      constant_to_uniform_descriptors_;

  std::unordered_map<const protobufs::UniformBufferElementDescriptor*,
                     const opt::analysis::Constant*,
                     UniformBufferElementDescriptorHash,
                     UniformBufferElementDescriptorEquals>
      uniform_descriptor_to_constant_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // #define SOURCE_FUZZ_FACT_MANAGER_H_
