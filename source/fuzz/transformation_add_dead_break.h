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
namespace transformation {

// - |message.from_block| must be the id of a block a in the given module.
// - |message.to_block| must be the id of a block b in the given module.
// - if |message.break_condition_value| holds (does not hold) then
//   OpConstantTrue (OpConstantFalse) must be present in the module
// - |message.phi_ids| must be a list of ids that are all available at
//   |message.from_block|
// - a and b must be in the same function.
// - b must be a merge block.
// - a must end with an unconditional branch to some block c.
// - replacing this branch with a conditional branch to b or c, with
//   the boolean constant associated with |message.break_condition_value| as
//   the condition, and the ids in |message.phi_ids| used to extend
//   any OpPhi instructions at b as a result of the edge from a, must
//   maintain validity of the module.
bool IsApplicable(const protobufs::TransformationAddDeadBreak& message,
                  opt::IRContext* context, const FactManager& fact_manager);

// Replaces the terminator of a with a conditional branch to b or c.
// The boolean constant associated with |message.break_condition_value| is used
// as the condition, and the order of b and c is arranged such that control is
// guaranteed to jump to c.
void Apply(const protobufs::TransformationAddDeadBreak& message,
           opt::IRContext* context, FactManager* fact_manager);

// Helper factory to create a transformation message.
protobufs::TransformationAddDeadBreak MakeTransformationAddDeadBreak(
    uint32_t from_block, uint32_t to_block, bool break_condition_value,
    std::vector<uint32_t> phi_ids);

}  // namespace transformation
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BREAK_H_
