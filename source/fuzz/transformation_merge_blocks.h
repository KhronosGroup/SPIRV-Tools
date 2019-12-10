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

#ifndef SOURCE_FUZZ_TRANSFORMATION_MERGE_BLOCKS_H_
#define SOURCE_FUZZ_TRANSFORMATION_MERGE_BLOCKS_H_

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationMergeBlocks : public Transformation {
 public:
  explicit TransformationMergeBlocks(
      const protobufs::TransformationMergeBlocks& message);

  TransformationMergeBlocks(uint32_t block_id);

  // - |message_.block_id| must be the id of a block, b
  // - b must have a single predecessor, a
  // - b must be the sole successor of a
  // - b must not be a merge block nor a continue target
  // - b must not start with OpPhi
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // The contents of b are merged into a, and a's terminator is replaced with
  // the terminator of b.  Block b is removed from the module.
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationMergeBlocks message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_MERGE_BLOCKS_H_
