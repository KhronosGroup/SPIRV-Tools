// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_LOCAL_VARIABLES_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_LOCAL_VARIABLES_H_

#include "source/fuzz/fuzzer_pass.h"

#include <utility>
#include <vector>

namespace spvtools {
namespace fuzz {

// TODO comment
class FuzzerPassAddLocalVariables : public FuzzerPass {
 public:
  FuzzerPassAddLocalVariables(
      opt::IRContext* ir_context, FactManager* fact_manager,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations);

  ~FuzzerPassAddLocalVariables();

  void Apply() override;

 private:
  using CompositeConstantSupplier =
      std::function<std::unique_ptr<opt::analysis::Constant>(
          const opt::analysis::Type& composite_type,
          const std::vector<const opt::analysis::Constant*>&
              component_constants)>;

  // TODO comment
  uint32_t FindOrCreateZeroConstant(uint32_t scalar_or_composite_type_id);

  // TODO comment; must be array, vector or matrix
  uint32_t GetZeroConstantForHomogeneousComposite(
      const opt::Instruction& composite_type_instruction,
      uint32_t component_type_id, uint32_t num_components,
      const CompositeConstantSupplier& composite_constant_supplier);

  // TODO comment
  uint32_t FindOrCreateCompositeConstant(
      const opt::Instruction& composite_type_instruction,
      const std::vector<const opt::analysis::Constant*>& constants,
      const std::vector<uint32_t>& constant_ids,
      const CompositeConstantSupplier& composite_constant_supplier);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_LOCAL_VARIABLES_H_
