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

#ifndef SOURCE_FUZZ_FACT_MANAGER_CONSTANT_UNIFORM_FACTS_H_
#define SOURCE_FUZZ_FACT_MANAGER_CONSTANT_UNIFORM_FACTS_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {
namespace detail {
namespace fact_manager {

// The purpose of this class is to group the fields and data used to represent
// facts about uniform constants.
class ConstantUniformFacts {
 public:
  // See method in FactManager which delegates to this method.
  bool AddFact(const protobufs::FactConstantUniform& fact,
               opt::IRContext* context);

  // See method in FactManager which delegates to this method.
  std::vector<uint32_t> GetConstantsAvailableFromUniformsForType(
      opt::IRContext* ir_context, uint32_t type_id) const;

  // See method in FactManager which delegates to this method.
  std::vector<protobufs::UniformBufferElementDescriptor>
  GetUniformDescriptorsForConstant(opt::IRContext* ir_context,
                                   uint32_t constant_id) const;

  // See method in FactManager which delegates to this method.
  uint32_t GetConstantFromUniformDescriptor(
      opt::IRContext* context,
      const protobufs::UniformBufferElementDescriptor& uniform_descriptor)
      const;

  // See method in FactManager which delegates to this method.
  std::vector<uint32_t> GetTypesForWhichUniformValuesAreKnown() const;

  // See method in FactManager which delegates to this method.
  const std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>&
  GetConstantUniformFactsAndTypes() const;

 private:
  // Returns true if and only if the words associated with
  // |constant_instruction| exactly match the words for the constant associated
  // with |constant_uniform_fact|.
  static bool DataMatches(
      const opt::Instruction& constant_instruction,
      const protobufs::FactConstantUniform& constant_uniform_fact);

  // Yields the constant words associated with |constant_uniform_fact|.
  static std::vector<uint32_t> GetConstantWords(
      const protobufs::FactConstantUniform& constant_uniform_fact);

  // Yields the id of a constant of type |type_id| whose data matches the
  // constant data in |constant_uniform_fact|, or 0 if no such constant is
  // declared.
  static uint32_t GetConstantId(
      opt::IRContext* context,
      const protobufs::FactConstantUniform& constant_uniform_fact,
      uint32_t type_id);

  // Checks that the width of a floating-point constant is supported, and that
  // the constant is finite.
  static bool FloatingPointValueIsSuitable(
      const protobufs::FactConstantUniform& fact, uint32_t width);

  std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>
      facts_and_type_ids_;
};

}  // namespace fact_manager
}  // namespace detail
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FACT_MANAGER_CONSTANT_UNIFORM_FACTS_H_
