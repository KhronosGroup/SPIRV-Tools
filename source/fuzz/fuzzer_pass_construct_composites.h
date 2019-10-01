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

#ifndef SOURCE_FUZZ_FUZZER_PASS_CONSTRUCT_COMPOSITES_H_
#define SOURCE_FUZZ_FUZZER_PASS_CONSTRUCT_COMPOSITES_H_

#include "source/fuzz/fuzzer_pass.h"

#include <map>
#include <vector>

namespace spvtools {
namespace fuzz {

// A fuzzer pass for constructing composite objects from smaller objects.
class FuzzerPassConstructComposites : public FuzzerPass {
 public:
  FuzzerPassConstructComposites(opt::IRContext* ir_context, FactManager* fact_manager,
                        FuzzerContext* fuzzer_context,
                        protobufs::TransformationSequence* transformations);

  ~FuzzerPassConstructComposites();

  void Apply() override;

 private:

  // TODO comment
  typedef std::map<uint32_t, std::vector<opt::Instruction*>> TypeIdToInstructions;

  // TODO comment
  void RecordAvailableInstruction(opt::Instruction* inst, TypeIdToInstructions* type_id_to_available_instructions);

  // TODO comment
  std::unique_ptr<std::vector<uint32_t>> TryConstructingArrayComposite(const opt::analysis::Array& array_type,
                                                                       const TypeIdToInstructions& type_id_to_available_instructions);

  // TODO comment
  std::unique_ptr<std::vector<uint32_t>> TryConstructingMatrixComposite(const opt::analysis::Matrix& matrix_type,
                                                                        const TypeIdToInstructions& type_id_to_available_instructions);

  // TODO comment
  std::unique_ptr<std::vector<uint32_t>> TryConstructingStructComposite(const opt::analysis::Struct& struct_type,
                                                                        const TypeIdToInstructions& type_id_to_available_instructions);

  // TODO comment
  std::unique_ptr<std::vector<uint32_t>> TryConstructingVectorComposite(const opt::analysis::Vector& vector_type,
                                      const TypeIdToInstructions& type_id_to_available_instructions);

};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_CONSTRUCT_COMPOSITES_H_
