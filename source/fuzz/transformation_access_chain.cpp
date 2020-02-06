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

#include "source/fuzz/transformation_access_chain.h"

namespace spvtools {
namespace fuzz {

TransformationAccessChain::TransformationAccessChain(
    const spvtools::fuzz::protobufs::TransformationAccessChain& message)
    : message_(message) {}

TransformationAccessChain::TransformationAccessChain(/* TODO */) {
  assert(false && "Not implemented yet");
}

bool TransformationAccessChain::IsApplicable(
    opt::IRContext* /*context*/,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  assert(false && "Not implemented yet");
  return false;
}

void TransformationAccessChain::Apply(
    opt::IRContext* /*context*/,
    spvtools::fuzz::FactManager* /*unused*/) const {
  assert(false && "Not implemented yet");
}

protobufs::Transformation TransformationAccessChain::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_access_chain() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
