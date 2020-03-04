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

#include "source/fuzz/transformation_swap_commutable_operands.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationSwapCommutableOperands::TransformationSwapCommutableOperands(
  const spvtools::fuzz::protobufs::TransformationSwapCommutableOperands& message
) : message_(message) {}

TransformationSwapCommutableOperands::TransformationSwapCommutableOperands(
  const protobufs::InstructionDescriptor& instruction_descriptor
) {
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationSwapCommutableOperands::IsApplicable(
  opt::IRContext* /*unused*/,
  const spvtools::fuzz::FactManager& /*unused*/
) const {
  SpvOp opcode = static_cast<SpvOp>(message_.instruction_descriptor().target_instruction_opcode());
  return spvOpcodeIsCommutative(opcode);
}

void TransformationSwapCommutableOperands::Apply(
  opt::IRContext* context, spvtools::fuzz::FactManager* /*unused*/
) const {
  auto instruction = FindInstruction(message_.instruction_descriptor(), context);
  std::swap(instruction->GetOperand(2), instruction->GetOperand(3));
}

protobufs::Transformation TransformationSwapCommutableOperands::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_swap_commutable_operands() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
