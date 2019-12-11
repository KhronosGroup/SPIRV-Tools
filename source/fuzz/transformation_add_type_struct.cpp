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

#include "source/fuzz/transformation_add_type_struct.h"

namespace spvtools {
namespace fuzz {

TransformationAddTypeStruct::TransformationAddTypeStruct(
    const spvtools::fuzz::protobufs::TransformationAddTypeStruct& message)
    : message_(message) {}

TransformationAddTypeStruct::TransformationAddTypeStruct(
    uint32_t fresh_id, const std::vector<uint32_t>& member_type_ids) {
  (void)(fresh_id);
  (void)(member_type_ids);
  assert(false && "Not implemented yet");
}

bool TransformationAddTypeStruct::IsApplicable(
    opt::IRContext* /*context*/,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  assert(false && "Not implemented yet");
  return false;
}

void TransformationAddTypeStruct::Apply(
    opt::IRContext* /*context*/,
    spvtools::fuzz::FactManager* /*unused*/) const {
  assert(false && "Not implemented yet");
}

protobufs::Transformation TransformationAddTypeStruct::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_type_struct() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
