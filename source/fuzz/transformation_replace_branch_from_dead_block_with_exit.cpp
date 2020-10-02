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

#include "source/fuzz/transformation_replace_branch_from_dead_block_with_exit.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceBranchFromDeadBlockWithExit::TransformationReplaceBranchFromDeadBlockWithExit(
    const spvtools::fuzz::protobufs::TransformationReplaceBranchFromDeadBlockWithExit& message)
    : message_(message) {}

TransformationReplaceBranchFromDeadBlockWithExit::TransformationReplaceBranchFromDeadBlockWithExit(/* TODO */) {
  assert(false && "Not implemented yet");
}

bool TransformationReplaceBranchFromDeadBlockWithExit::IsApplicable(
    opt::IRContext* /*ir_context*/,
    const TransformationContext& /*unused*/) const {
  assert(false && "Not implemented yet");
  return false;
}

void TransformationReplaceBranchFromDeadBlockWithExit::Apply(
    opt::IRContext* /*ir_context*/, TransformationContext* /*unused*/) const {
  assert(false && "Not implemented yet");
}

std::unordered_set<uint32_t> TransformationReplaceBranchFromDeadBlockWithExit::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

protobufs::Transformation TransformationReplaceBranchFromDeadBlockWithExit::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_branch_from_dead_block_with_exit() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
