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

#include "source/fuzz/transformation_merge_function_returns.h"

namespace spvtools {
namespace fuzz {

TransformationMergeFunctionReturns::TransformationMergeFunctionReturns(
    const protobufs::TransformationMergeFunctionReturns& message)
    : message_(message) {}

TransformationMergeFunctionReturns::TransformationMergeFunctionReturns(
    uint32_t function_id, uint32_t outer_header_id, uint32_t outer_return_id,
    uint32_t return_val_id, uint32_t any_returnable_val_id,
    const std::vector<protobufs::ReturnMergingInfo>& returns_merging_info) {
  message_.set_function_id(function_id);
  message_.set_outer_header_id(outer_header_id);
  message_.set_outer_return_id(outer_return_id);
  message_.set_return_val_id(return_val_id);
  message_.set_any_returnable_val_id(any_returnable_val_id);
  for (const auto& return_merging_info : returns_merging_info) {
    *message_.add_return_merging_info() = return_merging_info;
  }
}

bool TransformationMergeFunctionReturns::IsApplicable(
    opt::IRContext* /* ir_context */,
    const TransformationContext& /* transformation_context */) const {
  return false;
}

void TransformationMergeFunctionReturns::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationMergeFunctionReturns::ToMessage()
    const {
  return protobufs::Transformation();
}

}  // namespace fuzz
}  // namespace spvtools
