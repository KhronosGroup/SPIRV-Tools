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

#include "transformation_add_loop_preheader.h"

namespace spvtools {
namespace fuzz {
TransformationAddLoopPreheader::TransformationAddLoopPreheader(
    const protobufs::TransformationAddLoopPreheader& message)
    : message_(message) {}

TransformationAddLoopPreheader::TransformationAddLoopPreheader(
    uint32_t loop_header_block, uint32_t fresh_id,
    std::vector<uint32_t> phi_id) {
  message_.set_loop_header_block(loop_header_block);
  message_.set_fresh_id(fresh_id);
  for (auto id : phi_id) {
    message_.add_phi_id(id);
  }
}

bool TransformationAddLoopPreheader::IsApplicable(
    opt::IRContext* /* ir_context */,
    const TransformationContext& /* transformation_context */) const {
  return false;
}
void TransformationAddLoopPreheader::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationAddLoopPreheader::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_loop_preheader() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
