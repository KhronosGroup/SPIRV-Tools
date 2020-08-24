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

#ifndef SPIRV_TOOLS_TRANSFORMATION_DUPLICATE_REGION_WITH_SELECTION_H
#define SPIRV_TOOLS_TRANSFORMATION_DUPLICATE_REGION_WITH_SELECTION_H

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationDuplicateRegionWithSelection : public Transformation {
 public:
  explicit TransformationDuplicateRegionWithSelection(
      const protobufs::TransformationDuplicateRegionWithSelection& message);

  explicit TransformationDuplicateRegionWithSelection(
      uint32_t condition_fresh_id, uint32_t merge_label_fresh_id,
      uint32_t then_label_fresh_id, uint32_t else_label_fresh_id,
      uint32_t entry_block_id, uint32_t exit_block_id,
      std::map<uint32_t, uint32_t> original_label_to_duplicate_label,
      std::map<uint32_t, uint32_t> original_id_to_duplicate_id,
      std::map<uint32_t, uint32_t> original_id_to_phi_id);

  // - |condition_fresh_id|, |merge_label_fresh_id|, |then_label_fresh_id|,
  //   |else_label_fresh_id| must be fresh.
  // - |entry_block_id| and |exit_block_id| must refer to a single-entry,
  //   single-exit region.
  // - |original_label_to_duplicate_label| must at least contain a key for every
  //   block in the original region.
  // - |original_id_to_duplicate_id| must at least contain a key for every
  //   result id available at the end of the original region.
  // - |original_id_to_phi_id| must at least contain a key for every result id
  //   available at the end of the original region.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // A transformation that inserts a conditional statement on a boolean
  // expression of arbitrary value and duplicates a given single-entry,
  // single-exit region, so that it is present in every conditional branch and
  // will be executed regardless of which branch will be taken.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationDuplicateRegionWithSelection message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SPIRV_TOOLS_TRANSFORMATION_DUPLICATE_REGION_WITH_SELECTION_H
