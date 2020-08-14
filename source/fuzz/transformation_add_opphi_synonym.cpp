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

#include "source/fuzz/transformation_add_opphi_synonym.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {
TransformationAddOpPhiSynonym::TransformationAddOpPhiSynonym(
    const protobufs::TransformationAddOpPhiSynonym& message)
    : message_(message) {}

TransformationAddOpPhiSynonym::TransformationAddOpPhiSynonym(
    uint32_t block_id, std::map<uint32_t, uint32_t>& preds_to_ids,
    uint32_t fresh_id) {
  message_.set_block_id(block_id);
  *message_.mutable_pred_to_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(preds_to_ids);
  message_.set_fresh_id(fresh_id);
}

bool TransformationAddOpPhiSynonym::IsApplicable(
    opt::IRContext* /* ir_context */,
    const TransformationContext& /* transformation_context */) const {
  return false;
}

void TransformationAddOpPhiSynonym::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationAddOpPhiSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_opphi_synonym() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
