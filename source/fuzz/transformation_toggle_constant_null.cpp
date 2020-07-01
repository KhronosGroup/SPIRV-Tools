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

#include "source/fuzz/transformation_toggle_constant_null.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace fuzz {

TransformationToggleConstantNull::TransformationToggleConstantNull(
    const spvtools::fuzz::protobufs::TransformationToggleNullConstant& message)
    : message_(message) {}

TransformationToggleConstantNull::TransformationToggleConstantNull(
    uint32_t constant_id) {
  message_.set_constant_id(constant_id);
}

bool TransformationToggleConstantNull::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  auto constant = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.constant_id());

  // The instruction must exist.
  if (constant == nullptr) {
    return false;
  }

  // If it is a null constant, it must be either of integer, boolean or float
  // type.
  if (constant->AsNullConstant()) {
    auto kind = constant->type()->kind();
    return kind == opt::analysis::Type::kInteger ||
           kind == opt::analysis::Type::kBool ||
           kind == opt::analysis::Type::kFloat;
  }

  // If it is a scalar constant, the literal value must be zero.
  if (constant->AsScalarConstant()) {
    return constant->IsZero();
  }

  return false;
}

void TransformationToggleConstantNull::Apply(
    opt::IRContext* ir_context, TransformationContext* /* unused */) const {
  auto constant = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.constant_id());

  auto instruction =
      ir_context->get_constant_mgr()->GetDefiningInstruction(constant);

  // If OpConstantNull, change it to OpConstant or OpConstantFalse
  if (constant->AsNullConstant()) {
    auto kind = constant->type()->kind();

    if (kind == opt::analysis::Type::kBool) {
      // boolean
      instruction->SetOpcode(SpvOpConstantFalse);
    } else {
      // numerical
      instruction->SetOpcode(SpvOpConstant);
      // Add literal operand
      opt::Operand::OperandData operand_data;
      operand_data.push_back(0);
      opt::Operand operand =
          opt::Operand(SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, operand_data);
      instruction->AddOperand(std::move(operand));
      operand =
          opt::Operand(SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, operand_data);
    }
  } else {  // Change it to OpConstantNull
    // If int or float, remove constant literal operand
    const uint32_t kLiteralOperandIndex = 2;
    if (!constant->AsBoolConstant()) {
      instruction->RemoveOperand(kLiteralOperandIndex);
    }

    instruction->SetOpcode(SpvOpConstantNull);
  }

  // Replace the constant with the new one in the constants pool
  ir_context->get_constant_mgr()->RemoveId(message_.constant_id());
  ir_context->get_constant_mgr()->MapInst(instruction);

  // Invalidate analysis of constants since one of the constants changed
  ir_context->InvalidateAnalyses(opt::IRContext::kAnalysisConstants);
}

protobufs::Transformation TransformationToggleConstantNull::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_toggle_null_constant() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
