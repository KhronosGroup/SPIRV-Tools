// Copyright (c) 2020 Vasyl Teliman
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_FLOAT_CAST_SYNONYM_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_FLOAT_CAST_SYNONYM_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddFloatCastSynonym : public Transformation {
 public:
  explicit TransformationAddFloatCastSynonym(
      protobufs::TransformationAddFloatCastSynonym message);

  TransformationAddFloatCastSynonym(uint32_t synonym_id,
                                    uint32_t to_float_fresh_id,
                                    uint32_t to_int_fresh_id,
                                    uint32_t float_type_id);

  // - |synonym_id| must be a valid result id of some instruction in the module.
  // - the instruction with |synonym_id| result id must have either an integral
  //   or a vector of integers type.
  // - it should be possible to insert OpConvert* instructions after
  //   |synonym_id|.
  // - |to_float_fresh_id| and |to_int_fresh_id| must be fresh ids.
  // - |float_type_id| must be a result id of OpTypeFloat or OpTypeVector with
  //   OpTypeFloat components.
  // - if |synonym_id|'th type is OpTypeInteger then |float_type_id| must be a
  //   result id of OpTypeFloat with the same width. Otherwise, |float_type_id|
  //   is a result id of an OpTypeVector with OpTypeFloat components. It must
  //   have the same component count and the width of components as |synonym_id|
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an OpConvertSToF or OpConvertUToF with |to_float_fresh_id| after
  // |synonym_id| and OpConvertFToS or OpConvertFToU with |to_int_fresh_id|
  // after |to_float_fresh_id|. Marks |to_int_fresh_id| and |synonym_id| as
  // synonymous.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddFloatCastSynonym message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_FLOAT_CAST_SYNONYM_H_
