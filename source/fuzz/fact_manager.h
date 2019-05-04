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
  FactManager();

  virtual ~FactManager();

  // Adds all the facts from |facts|, checking them for validity with respect to
  // |context|. Returns true if and only if all facts are valid.
  bool AddFacts(const protobufs::FactSequence& facts, opt::IRContext* context);

  // Adds |fact| to the fact manager, checking it for validity with respect to
  // |context|. Returns true if and only if the fact is valid.
  bool AddFact(const protobufs::Fact& fact, opt::IRContext* context);

  //==============================
  // Querying facts about uniform constants

  // Provides a sequence of all types for which at least one "constant ==
  // uniform element" fact is known.
  std::vector<const opt::analysis::Type*>
  GetTypesForWhichUniformValuesAreKnown() const;

  // Provides a sequence of all distinct constants for which an equal uniform
  // element is known.
  std::vector<const opt::analysis::ScalarConstant*>
  GetConstantsAvailableFromUniformsForType(
      const opt::analysis::Type& type) const;

  // Provides details of all uniform elements that are known to be equal to
  // |constant|.
  const std::vector<protobufs::UniformBufferElementDescriptor>*
  GetUniformDescriptorsForConstant(
      const opt::analysis::ScalarConstant& constant) const;

  // Returns the constant known to be equal to the given uniform element, and
  // nullptr if there is no such constant.
  const opt::analysis::ScalarConstant* GetConstantFromUniformDescriptor(
      const protobufs::UniformBufferElementDescriptor& uniform_descriptor)
      const;

  // End of uniform constant facts
  //==============================

 private:
  // Add an opaque struct type for each distinct category of fact to be managed.

  struct UniformConstantFacts;  // Opaque struct for holding data about uniform
                                // buffer elements.
  std::unique_ptr<UniformConstantFacts>
      uniform_constant_facts_;  // Unique pointer to internal data.
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // #define SOURCE_FUZZ_FACT_MANAGER_H_
