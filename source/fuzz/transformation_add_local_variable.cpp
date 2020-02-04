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

#include "source/fuzz/transformation_add_local_variable.h"

namespace spvtools {
namespace fuzz {

TransformationAddLocalVariable::TransformationAddLocalVariable(
    const spvtools::fuzz::protobufs::TransformationAddLocalVariable& message)
    : message_(message) {}

TransformationAddLocalVariable::TransformationAddLocalVariable(uint32_t fresh_id, uint32_t type_id, uint32_t function_id,
uint32_t initializer_id, bool value_is_arbitrary) {
  message_.set_fresh_id(fresh_id);
  message_.set_type_id(type_id);
  message_.set_function_id(function_id);
  message_.set_initializer_id(initializer_id);
  message_.set_value_is_arbitrary(value_is_arbitrary);
}

bool TransformationAddLocalVariable::IsApplicable(
    opt::IRContext* /*context*/,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  assert(false && "Not implemented yet");
  return false;
}

void TransformationAddLocalVariable::Apply(
    opt::IRContext* /*context*/,
    spvtools::fuzz::FactManager* /*unused*/) const {
  assert(false && "Not implemented yet");
}

protobufs::Transformation TransformationAddLocalVariable::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_local_variable() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
