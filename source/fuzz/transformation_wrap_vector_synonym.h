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

#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_composite_construct.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationWrapVectorSynonym : public Transformation {
 public:
  explicit TransformationWrapVectorSynonym(
      protobufs::TransformationWrapVectorSynonym message);

  TransformationWrapVectorSynonym(uint32_t instruction_id, uint32_t result_id1,
                                  uint32_t result_id2, uint32_t vec_id,
                                  uint32_t vec_type_id, uint32_t pos);
  // - |instruction_id| must be the id of a arithmetic operation.
  // - |result_id1| and |result_id2| represents the result ids of the two added
  // vector.
  // - |vec_id| is a vector type for the result of the transformation.
  // - result vector type must match the type of two vectors being added.
  // - |vec_id| must be fresh.
  // - |result_id1|, |result_id2| and |vec_id| should be different from each
  // other.
  // - They must be of a valid vector type.
  // - |pos| is a 0-indexed position of the component that contains the
  // - value of a and b. pos must be smaller than the number of
  // - elements that the vector type can has.
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

#endif  // SOURCE_FUZZ_TRANSFORMATION_WRAP_VECTOR_SYNONYM_H_