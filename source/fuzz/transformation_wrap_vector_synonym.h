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

#ifndef SOURCE_FUZZ_TRANSFORMATION_WRAP_VECTOR_SYNONYM_H_
#define SOURCE_FUZZ_TRANSFORMATION_WRAP_VECTOR_SYNONYM_H_

#include <utility>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_composite_construct.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/opt/instruction.h"
#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

class TransformationWrapVectorSynonym : public Transformation {
 public:
  explicit TransformationWrapVectorSynonym(
      protobufs::TransformationWrapVectorSynonym message);

  TransformationWrapVectorSynonym(uint32_t instruction_id, uint32_t vec_id1,
                                  uint32_t vec_id2, uint32_t vec_id3,
                                  uint32_t vec_type_id, uint32_t pos,
                                  std::vector<uint32_t>& vec1_elements,
                                  std::vector<uint32_t>& vec2_elements);
  // - |instruction_id| must be the id of a arithmetic operation.
// - |vec_id1| and |vec_id2| represents the ids of the two added vector.
// - |arith_id| is the id of the arithmetic operation that performs vector arithmetic.
// - |vec_id1|, |vec_id2| and |arith_id3| must be fresh ids.
// - and they should be different from each other.
// - |vec_type_id| must be a valid vector type from vec2 to vec4.
// - |pos| is a 0-indexed position of the component that contains the
// - value of a and b. pos must be smaller than the number of
// - elements that the vector type can has.
// - |vec1_elements_ids| and |vec2_elements_ids| must contain scalar type ids.
// - They should have an id of a zero constant at position
// - |pos|. All ids must be valid ids that has the type as the variables
// - in the original instruction.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;
  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationWrapVectorSynonym message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif //SOURCE_FUZZ_TRANSFORMATION_WRAP_VECTOR_SYNONYM_H_