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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BREAK_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BREAK_H_

#include "source/fuzz/protobufs/spirvfuzz.pb.h"
#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

// A transformation that turns a basic block that unconditionally branches to
// its successor into a block that potentially breaks out of a structured
// control flow construct, but in such a manner that the break cannot actually
// be taken.
class TransformationAddDeadBreak : public Transformation {
 public:
  // Constructs a transformation from given ids and boolean.
  TransformationAddDeadBreak(uint32_t from_block, uint32_t to_block,
                             bool break_condition_value,
                             std::vector<uint32_t>&& phi_ids)
      : from_block_(from_block),
        to_block_(to_block),
        break_condition_value_(break_condition_value),
        phi_ids_(phi_ids) {}

  // Constructs a transformation from a protobuf message.
  explicit TransformationAddDeadBreak(
      const protobufs::TransformationAddDeadBreak& message);

  ~TransformationAddDeadBreak() override = default;

  // - |from_block_| must be the id of a block a in the given module.
  // - |to_block_| must be the id of a block b in the given module.
  // - if |break_condition_value_| holds (does not hold) then OpConstantTrue
  //   (OpConstantFalse) must be present in the module
  // - |phi_ids_| must be a list of ids that are all available at |from_block_|
  // - a and b must be in the same function.
  // - b must be a merge block.
  // - a must end with an unconditional branch to some block c.
  // - replacing this branch with a conditional branch to b or c, with
  //   |bool_id_| as the condition, and the ids in |phi_ids_| used to extend
  //   any OpPhi instructions at b as a result of the edge from a, must
  //   maintain validity of the module.
  bool IsApplicable(opt::IRContext* context) override;

  // Replaces the terminator of a with a conditional branch to b or c.
  // |bool_id_| is used as the condition, and the order of b and c is
  // arranged such that control is guaranteed to jump to c.
  void Apply(opt::IRContext* context) override;

  protobufs::Transformation ToMessage() override;

 private:
  // Helper that retrieves the basic block for |maybe_block_id|, or nullptr if
  // no such block exists.
  opt::BasicBlock* MaybeFindBlock(opt::IRContext* context,
                                  uint32_t maybe_block_id);

  // Helper to check whether the contents of |phi_ids_| are suitable for
  // extending the OpPhi instructions of |to_block_| if an edge
  // |from_block_|->|to_block_| is added. |bb_from| and |bb_to| refer to the
  // basic blocks for |from_block_| and |to_block_|.
  bool PhiIdsOk(opt::IRContext* context, opt::BasicBlock* bb_from,
                opt::BasicBlock* bb_to);

  // Helper to check that adding the dead break would respect the rules of
  // structured control flow.  |bb_From| is the basic block associated with
  // |from_block_|.
  bool AddingBreakRespectsStructuredControlFlow(opt::IRContext* context,
                                                opt::BasicBlock* bb_from);

  // Helper to check whether |from_block_| is part of the continue construct
  // of a loop headed at |maybe_loop_header|.
  bool FromBlockIsInLoopContinueConstruct(opt::IRContext* context,
                                          uint32_t maybe_loop_header);

  // The block to break from
  const uint32_t from_block_;
  // The merge block to break to
  const uint32_t to_block_;
  // Determines whether the break condition is true or false
  const bool break_condition_value_;
  // A sequence of ids suitable for extending OpPhi instructions as a result of
  // the new break edge
  std::vector<uint32_t> phi_ids_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BREAK_H_
