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

#include "source/fuzz/transformation_set_memory_operands_mask.h"

namespace spvtools {
namespace fuzz {

TransformationSetMemoryOperandsMask::TransformationSetMemoryOperandsMask(
    const spvtools::fuzz::protobufs::TransformationSetMemoryOperandsMask&
        message)
    : message_(message) {}

TransformationSetMemoryOperandsMask::TransformationSetMemoryOperandsMask(
    const protobufs::InstructionDescriptor& memory_access_instruction,
    uint32_t memory_operands_mask, uint32_t memory_operands_mask_index) {
  *message_.mutable_memory_access_instruction() = memory_access_instruction;
  message_.set_memory_operands_mask(memory_operands_mask);
  message_.set_memory_operands_mask_index(memory_operands_mask_index);
}

bool TransformationSetMemoryOperandsMask::IsApplicable(
    opt::IRContext* /*context*/,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  assert(false && "Not implemented yet");
  return false;
}

void TransformationSetMemoryOperandsMask::Apply(
    opt::IRContext* /*context*/,
    spvtools::fuzz::FactManager* /*unused*/) const {
  assert(false && "Not implemented yet");
}

protobufs::Transformation TransformationSetMemoryOperandsMask::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_set_memory_operands_mask() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
