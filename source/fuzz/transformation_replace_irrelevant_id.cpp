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

#include "source/fuzz/transformation_replace_irrelevant_id.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceIrrelevantId::TransformationReplaceIrrelevantId(
    const protobufs::TransformationReplaceIrrelevantId& message)
    : message_(message) {}

TransformationReplaceIrrelevantId::TransformationReplaceIrrelevantId(
    protobufs::IdUseDescriptor id_use_descriptor, uint32_t replacement_id) {
  *message_.mutable_id_use_descriptor() = id_use_descriptor;
  message_.set_replacement_id(replacement_id);
}

bool TransformationReplaceIrrelevantId::IsApplicable(
    opt::IRContext* /* ir_context */,
    const TransformationContext& /* transformation_context */) const {
  return false;
}

void TransformationReplaceIrrelevantId::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationReplaceIrrelevantId::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_irrelevant_id() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
