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

#include "source/fuzz/transformation_flatten_conditional_branch.h"

namespace spvtools {
namespace fuzz {

TransformationFlattenConditionalBranch::TransformationFlattenConditionalBranch(
    const protobufs::TransformationFlattenConditionalBranch& message)
    : message_(message) {}

TransformationFlattenConditionalBranch::TransformationFlattenConditionalBranch(
    uint32_t header_block_id,
    std::map<protobufs::InstructionDescriptor, std::vector<uint32_t>>
        instructions_to_fresh_ids) {
  message_.set_header_block_id(header_block_id);
  for (auto const& pair : instructions_to_fresh_ids) {
    protobufs::InstructionUint32ListPair mapping;
    *mapping.mutable_instruction_descriptor() = pair.first;
    for (auto id : pair.second) {
      mapping.add_id(id);
    }
    *message_.mutable_instructions_to_fresh_ids() = mapping;
  }
}

bool TransformationFlattenConditionalBranch::IsApplicable(
    opt::IRContext* /* ir_context */,
    const TransformationContext& /* transformation_context */) const {
  return false;
}

void TransformationFlattenConditionalBranch::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationFlattenConditionalBranch::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_flatten_conditional_branch() = message_;
  return result;
}
}  // namespace fuzz
}  // namespace spvtools
