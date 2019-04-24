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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SPLIT_BLOCK_H_
#define SOURCE_FUZZ_TRANSFORMATION_SPLIT_BLOCK_H_

#include "source/fuzz/protobufs/spirvfuzz.pb.h"
#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

// A transformation that splits a basic block into two basic blocks.
class TransformationSplitBlock : public Transformation {
 public:
  // Constructs a transformation from given ids and offset.
  TransformationSplitBlock(uint32_t result_id, uint32_t offset,
                           uint32_t fresh_id)
      : result_id_(result_id), offset_(offset), fresh_id_(fresh_id) {}

  // Constructs a transformation from a protobuf message.
  explicit TransformationSplitBlock(
      const protobufs::TransformationSplitBlock& message)
      : result_id_(message.result_id()),
        offset_(message.offset()),
        fresh_id_(message.fresh_id()) {}

  ~TransformationSplitBlock() override = default;

  // - |result_id_| must be the result id of an instruction 'base' in some
  //   block 'blk'.
  // - 'blk' must contain an instruction 'inst' located |offset_| instructions
  //   after 'inst' (if |offset_| = 0 then 'inst' = 'base').
  // - Splitting 'blk' at 'inst', so that all instructions from 'inst' onwards
  //   appear in a new block that 'blk' directly jumps to must be valid.
  // - |fresh_id_| must not be used by the module.
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) override;

  // - A new block with label |fresh_id_| is inserted right after 'blk' in
  //   program order.
  // - All instructions of 'blk' from 'inst' onwards are moved into the new
  //   block.
  // - 'blk' is made to jump unconditionally to the new block.
  void Apply(opt::IRContext* context, FactManager* fact_manager) override;

  protobufs::Transformation ToMessage() override;

 private:
  // The result id of an instruction.
  const uint32_t result_id_;
  // An offset, such that the block containing |result_id_| should be split
  // right before the instruction |offset_| instructions after |result_id_|.
  const uint32_t offset_;
  // An id that must not yet be used by the module to which this transformation
  // is applied.  Rather than having the transformation choose a suitable id on
  // application, we require the id to be given upfront in order to facilitate
  // reducing fuzzed shaders by removing transformations.  The reason is that
  // future transformations may refer to the fresh id introduced by this
  // transformation, and if we end up changing what that id is, due to removing
  // earlier transformations, it may inhibit later transformations from
  // applying.
  const uint32_t fresh_id_;

  // Returns (true, block.end()) if the relevant instruction is in this block
  // but inapplicable
  //         (true, it) if 'it' is an iterator for the relevant instruction
  //         (false, _) otherwise.
  std::pair<bool, opt::BasicBlock::iterator> FindInstToSplitBefore(
      opt::BasicBlock* block);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SPLIT_BLOCK_H_
