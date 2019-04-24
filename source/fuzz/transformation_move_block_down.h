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

#ifndef SOURCE_FUZZ_TRANSFORMATION_MOVE_BLOCK_DOWN_H_
#define SOURCE_FUZZ_TRANSFORMATION_MOVE_BLOCK_DOWN_H_

#include "source/fuzz/protobufs/spirvfuzz.pb.h"
#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

// A transformation that moves a basic block to be one position lower in program
// order.
class TransformationMoveBlockDown : public Transformation {
 public:
  // Constructs a transformation from a given id.
  explicit TransformationMoveBlockDown(uint32_t block_id)
      : block_id_(block_id) {}

  // Constructs a transformation from a protobuf message.
  explicit TransformationMoveBlockDown(
      const protobufs::TransformationMoveBlockDown& message)
      : block_id_(message.block_id()) {}

  ~TransformationMoveBlockDown() override = default;

  // - |block_id_| must be the id of a block b in the given module.
  // - b must not be the first nor last block appearing, in program order,
  //   in a function.
  // - b must not dominate the block that follows it in program order.
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) override;

  // The block with id |block_id_| is moved down; i.e. the program order
  // between it and the block that follows it is swapped.
  void Apply(opt::IRContext* context, FactManager* fact_manager) override;

  protobufs::Transformation ToMessage() override;

 private:
  // The id of the block to move down.
  const uint32_t block_id_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_MOVE_BLOCK_DOWN_H_
