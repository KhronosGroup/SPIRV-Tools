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

#include "transformation.h"

namespace spvtools {
namespace fuzz {

// A transformation that turns a basic block that unconditionally branches to
// its successor into a block that potentially breaks out of a structured
// control flow construct, but in such a manner that the break cannot actually
// be taken.
class TransformationAddDeadBreak : public Transformation {
 public:
  TransformationAddDeadBreak(uint32_t from_block, uint32_t to_block,
                             uint32_t bool_id)
      : from_block_(from_block), to_block_(to_block), bool_id_(bool_id) {}

  ~TransformationAddDeadBreak() override = default;

  // - |from_block_| must be the id of a block a in the given module.
  // - |to_block_| must be the id of a block b in the given module.
  // - |bool_id_| must be the id of a boolean constant (OpConstantTrue or
  //   OpConstantFalse)
  // - b must be a merge block.
  // - a must end with an unconditional branch to some block c.
  // - replacing this branch with a conditional branch to b or c, with
  //   |bool_id_| as the condition,
  //   must maintain validity of the module.
  bool IsApplicable(opt::IRContext* context) override;

  // Replaces the terminator of a with a conditional branch to b or c.
  // |bool_id_| is used as the condition, and the order of b and c is
  // arranged such that control is guaranteed to jump to c.
  void Apply(opt::IRContext* context) override;

 private:
  // The block to break from
  const uint32_t from_block_;
  // The merge block to break to
  const uint32_t to_block_;
  // The id of a boolean constant
  const uint32_t bool_id_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BREAK_H_
