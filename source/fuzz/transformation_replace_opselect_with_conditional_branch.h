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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_OPSELECT_WITH_CONDITIONAL_BRANCH_H
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_OPSELECT_WITH_CONDITIONAL_BRANCH_H

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceOpselectWithConditionalBranch
    : public Transformation {
 public:
  explicit TransformationReplaceOpselectWithConditionalBranch(
      const protobufs::TransformationReplaceOpselectWithConditionalBranch&
          message);

  TransformationReplaceOpselectWithConditionalBranch(
      uint32_t select_id, std::pair<uint32_t, uint32_t> new_block_ids);

  // - |message_.select_id| is the result id of an OpSelect instruction.
  // - The block containing the instruction can be split at the position
  //   corresponding to the instruction.
  // - The pair |message_.new_block_ids| contains 2 fresh and distinct ids
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces the OpSelect instruction with id |message_.select_id| with a
  // conditional branch and an OpPhi instruction.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationReplaceOpselectWithConditionalBranch message_;
};
}  // namespace fuzz
}  // namespace spvtools
#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_OPSELECT_WITH_CONDITIONAL_BRANCH_H
