// Copyright (c) 2021 Shiyu Liu
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

#ifndef SOURCE_FUZZ_FUZZER_WRAP_VECTOR_SYNONYM_H_
#define SOURCE_FUZZ_FUZZER_WRAP_VECTOR_SYNONYM_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Randomly wrap a scalar operation into a vector operation.
class FuzzerPassWrapVectorSynonym : public FuzzerPass {
 public:
  FuzzerPassWrapVectorSynonym(opt::IRContext* ir_context,
                          TransformationContext* transformation_context,
                          FuzzerContext* fuzzer_context,
                          protobufs::TransformationSequence* transformations);

  void Apply() override;
 private:
  // Add a vector type of the given size from 2 to 4 to the module and return the result id back.
  uint32_t AddNewVecNType(uint32_t composite_type_id, std::vector<uint32_t> component,
                      const protobufs::InstructionDescriptor& inst_to_insert_before);

  // Randomly add a float constant id with specified width to a vector.
  void AddRandomFloatConstant(std::vector<uint32_t>& vec, uint32_t width, RandomGenerator* random_generator);

  // Randomly add a integer constant id with specified width and sign to a vector.
  void AddRandomIntConstant(std::vector<uint32_t>& vec, uint32_t width, bool is_signed, RandomGenerator* random_generator);

  std::unordered_set<SpvOp> valid_arithmetic_types {SpvOpIAdd, SpvOpISub, SpvOpIMul, SpvOpFAdd, SpvOpFSub, SpvOpFMul};
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_WRAP_VECTOR_SYNONYM_H_
