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

#include "source/fuzz/transformation_replace_opselect_with_conditional_branch.h"

namespace spvtools {
namespace fuzz {
TransformationReplaceOpSelectWithConditionalBranch::
    TransformationReplaceOpSelectWithConditionalBranch(
        const spvtools::fuzz::protobufs::
            TransformationReplaceOpSelectWithConditionalBranch& message)
    : message_(message) {}

TransformationReplaceOpSelectWithConditionalBranch::
    TransformationReplaceOpSelectWithConditionalBranch(
        uint32_t select_id, std::vector<uint32_t> new_block_ids) {
  message_.set_select_id(select_id);
  for (auto id : new_block_ids) {
    message_.add_new_block_id(id);
  }
}
bool TransformationReplaceOpSelectWithConditionalBranch::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* transformation_context */) const {
  return false;
}

void TransformationReplaceOpSelectWithConditionalBranch::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation
TransformationReplaceOpSelectWithConditionalBranch::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_opselect_with_conditional_branch() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
