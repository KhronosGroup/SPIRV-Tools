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

#include "source/fuzz/transformation_add_global_variable.h"

namespace spvtools {
namespace fuzz {

TransformationAddGlobalVariable::TransformationAddGlobalVariable(
    const spvtools::fuzz::protobufs::TransformationAddGlobalVariable& message)
    : message_(message) {}

TransformationAddGlobalVariable::TransformationAddGlobalVariable(
    uint32_t fresh_id, uint32_t type_id, SpvStorageClass storage_class,
    uint32_t initializer_id) {
  (void)(fresh_id);
  (void)(type_id);
  (void)(storage_class);
  (void)(initializer_id);
  assert(false && "Not implemented yet");
}

bool TransformationAddGlobalVariable::IsApplicable(
    opt::IRContext* /*context*/,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  assert(false && "Not implemented yet");
  return false;
}

void TransformationAddGlobalVariable::Apply(
    opt::IRContext* /*context*/,
    spvtools::fuzz::FactManager* /*unused*/) const {
  assert(false && "Not implemented yet");
}

protobufs::Transformation TransformationAddGlobalVariable::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_global_variable() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
