// Copyright (c) 2020 Stefano Milizia
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

#include "source/fuzz/transformation_toggle_null_constant.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationToggleNullConstant::TransformationToggleNullConstant(
    const spvtools::fuzz::protobufs::TransformationToggleNullConstant& message)
    : message_(message) {}

TransformationToggleNullConstant::TransformationToggleNullConstant(
    spvtools::fuzz::protobufs::InstructionDescriptor& instruction_descriptor) {
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationToggleNullConstant::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  auto instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);

  // The instruction must exist.
  if (instruction == nullptr) {
    return false;
  }

  const SpvOp opcode = instruction->opcode();

  // The instruction must be one of OpConstant, OpConstantFalse and
  // OpConstantNull.
  if (opcode != SpvOpConstant && opcode != SpvOpConstantFalse &&
      opcode != SpvOpConstantNull) {
    return false;
  }

  // The constant must be of type integer, boolean or floating-point number.
  const uint32_t type_id = instruction->type_id();
  if (type_id != SpvOpTypeInt && type_id != SpvOpTypeBool &&
      type_id != SpvOpTypeFloat) {
    return false;
  }

  // If OpConstant, the literal value must be zero.
  if (opcode == SpvOpConstant) {
    const uint32_t kLiteralOperand = 1;
    return instruction->GetOperand(kLiteralOperand).AsLiteralUint64() == 0;
  }

  return true;
}

}  // namespace fuzz
}  // namespace spvtools
