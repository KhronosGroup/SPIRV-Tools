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

#ifndef SOURCE_FUZZ_TRANSFORMATION_WRAP_REGION_IN_SELECTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_WRAP_REGION_IN_SELECTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationWrapRegionInSelection : public Transformation {
 public:
  explicit TransformationWrapRegionInSelection(
      const protobufs::TransformationWrapRegionInSelection& message);

  TransformationWrapRegionInSelection(uint32_t region_entry_block_id,
                                      uint32_t region_exit_block_id,
                                      bool branch_condition);

  // TODO
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // TODO
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if |header_block_candidate_id| can be transformed into a
  // selection header block with |merge_block_candidate_id| as it's merge block
  // without changing the semantics of the module.
  static bool IsApplicableToBlockRange(opt::IRContext* ir_context,
                                       uint32_t header_block_candidate_id,
                                       uint32_t merge_block_candidate_id);

 private:
  protobufs::TransformationWrapRegionInSelection message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_WRAP_REGION_IN_SELECTION_H_
