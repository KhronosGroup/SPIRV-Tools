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

#include "source/fuzz/transformation_equation_instruction.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationEquationInstruction::TransformationEquationInstruction(
    const spvtools::fuzz::protobufs::TransformationEquationInstruction& message)
    : message_(message) {}

TransformationEquationInstruction::TransformationEquationInstruction(
    const std::vector<uint32_t>& fresh_id, SpvOp opcode,
    const std::vector<uint32_t>& in_operand_id,
    const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  for (auto id : fresh_id) {
    message_.add_fresh_id(id);
  }

  message_.set_opcode(opcode);
  for (auto id : in_operand_id) {
    message_.add_in_operand_id(id);
  }
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
}

bool TransformationEquationInstruction::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |fresh_id| must all be fresh.
  if (!std::all_of(message_.fresh_id().begin(), message_.fresh_id().end(),
                   [ir_context](uint32_t id) {
                     return fuzzerutil::IsFreshId(ir_context, id);
                   })) {
    return false;
  }

  // |fresh_id| must not have duplicates.
  if (fuzzerutil::HasDuplicates(std::vector<uint32_t>(
          message_.fresh_id().begin(), message_.fresh_id().end()))) {
    return false;
  }

  // The instruction to insert before must exist.
  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  if (!insert_before) {
    return false;
  }
  // The input ids must all exist, not be OpUndef, and be available before this
  // instruction.
  for (auto id : message_.in_operand_id()) {
    auto inst = ir_context->get_def_use_mgr()->GetDef(id);
    if (!inst) {
      return false;
    }
    if (inst->opcode() == SpvOpUndef) {
      return false;
    }
    if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                    id)) {
      return false;
    }
  }

  // Check that the module remains valid if insert new instruction with
  // |opcode|.
  auto opcode = static_cast<SpvOp>(message_.opcode());
  switch (opcode) {
    case SpvOpConvertUToF:
    case SpvOpConvertSToF: {
      if (message_.in_operand_id_size() != 1 || message_.fresh_id_size() != 3) {
        return false;
      }

      const auto* type = ir_context->get_type_mgr()->GetType(
          fuzzerutil::GetTypeId(ir_context, message_.in_operand_id(0)));
      if (!type) {
        return false;
      }

      if (const auto* vector = type->AsVector()) {
        return vector->element_type()->AsInteger() &&
               vector->element_type()->AsInteger()->IsSigned() ==
                   (opcode == SpvOpConvertSToF);
      } else {
        return type->AsInteger() &&
               type->AsInteger()->IsSigned() == (opcode == SpvOpConvertSToF);
      }
    }
    case SpvOpConvertFToU:
    case SpvOpConvertFToS: {
      if (message_.in_operand_id_size() != 1 || message_.fresh_id_size() != 3) {
        return false;
      }

      const auto* type = ir_context->get_type_mgr()->GetType(
          fuzzerutil::GetTypeId(ir_context, message_.in_operand_id(0)));
      if (!type) {
        return false;
      }

      if (const auto* vector = type->AsVector()) {
        return vector->element_type()->AsFloat();
      } else {
        return type->AsFloat();
      }
    }
    case SpvOpIAdd:
    case SpvOpISub: {
      if (message_.in_operand_id_size() != 2 || message_.fresh_id_size() != 1) {
        return false;
      }
      uint32_t first_operand_width = 0;
      uint32_t first_operand_type_id = 0;
      for (uint32_t index = 0; index < 2; index++) {
        auto operand_inst = ir_context->get_def_use_mgr()->GetDef(
            message_.in_operand_id(index));
        if (!operand_inst || !operand_inst->type_id()) {
          return false;
        }
        auto operand_type =
            ir_context->get_type_mgr()->GetType(operand_inst->type_id());
        if (!(operand_type->AsInteger() ||
              (operand_type->AsVector() &&
               operand_type->AsVector()->element_type()->AsInteger()))) {
          return false;
        }
        uint32_t operand_width =
            operand_type->AsInteger()
                ? 1
                : operand_type->AsVector()->element_count();
        if (index == 0) {
          first_operand_width = operand_width;
          first_operand_type_id = operand_inst->type_id();
        } else {
          assert(first_operand_width != 0 &&
                 "The first operand should have been processed.");
          if (operand_width != first_operand_width) {
            return false;
          }
        }
      }
      assert(first_operand_type_id != 0 &&
             "A type must have been found for the first operand.");
      return true;
    }
    case SpvOpLogicalNot: {
      if (message_.in_operand_id_size() != 1 || message_.fresh_id_size() != 1) {
        return false;
      }
      auto operand_inst =
          ir_context->get_def_use_mgr()->GetDef(message_.in_operand_id(0));
      if (!operand_inst || !operand_inst->type_id()) {
        return false;
      }
      auto operand_type =
          ir_context->get_type_mgr()->GetType(operand_inst->type_id());
      return operand_type->AsBool() ||
             (operand_type->AsVector() &&
              operand_type->AsVector()->element_type()->AsBool());
    }
    case SpvOpSNegate: {
      if (message_.in_operand_id_size() != 1 || message_.fresh_id_size() != 1) {
        return false;
      }
      auto operand_inst =
          ir_context->get_def_use_mgr()->GetDef(message_.in_operand_id(0));
      if (!operand_inst || !operand_inst->type_id()) {
        return false;
      }
      auto operand_type =
          ir_context->get_type_mgr()->GetType(operand_inst->type_id());
      return operand_type->AsInteger() ||
             (operand_type->AsVector() &&
              operand_type->AsVector()->element_type()->AsInteger());
    }
    default:
      assert(false && "Inappropriate opcode for equation instruction.");
      return false;
  }
}

void TransformationEquationInstruction::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  *std::max_element(message_.fresh_id().begin(),
                                                    message_.fresh_id().end()));

  opt::Instruction::OperandList in_operands;
  std::vector<uint32_t> rhs_id;
  for (auto id : message_.in_operand_id()) {
    in_operands.push_back({SPV_OPERAND_TYPE_ID, {id}});
    rhs_id.push_back(id);
  }

  FindInstruction(message_.instruction_to_insert_before(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, static_cast<SpvOp>(message_.opcode()),
          ComputeResultTypeId(ir_context), message_.fresh_id(0),
          std::move(in_operands)));

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  transformation_context->GetFactManager()->AddFactIdEquation(
      message_.fresh_id(0), static_cast<SpvOp>(message_.opcode()), rhs_id,
      ir_context);
}

protobufs::Transformation TransformationEquationInstruction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_equation_instruction() = message_;
  return result;
}

uint32_t TransformationEquationInstruction::ComputeResultTypeId(
    opt::IRContext* ir_context) const {
  auto opcode = static_cast<SpvOp>(message_.opcode());
  switch (opcode) {
    case SpvOpConvertUToF:
    case SpvOpConvertSToF: {
      assert(message_.in_operand_id_size() == 1 &&
             "Instruction has invalid number of operands");
      assert(message_.fresh_id_size() == 3 &&
             "3 fresh ids must be provided for conversion instructions");

      const auto* type = ir_context->get_type_mgr()->GetType(
          fuzzerutil::GetTypeId(ir_context, message_.in_operand_id(0)));
      assert(type && "Operand has invalid type");

      if (const auto* vector = type->AsVector()) {
        assert(vector->element_type()->AsInteger() &&
               "Conversion to float supports only operands of scalar or vector "
               "integral type");

        return fuzzerutil::FindOrCreateVectorType(
            ir_context, message_.fresh_id(1),
            fuzzerutil::FindOrCreateFloatType(
                ir_context, message_.fresh_id(2),
                vector->element_type()->AsInteger()->width()),
            vector->element_count());
      } else {
        assert(type->AsInteger() &&
               "Conversion to float supports only operands of scalar or vector "
               "integral type");

        return fuzzerutil::FindOrCreateFloatType(
            ir_context, message_.fresh_id(1), type->AsInteger()->width());
      }
    }
    case SpvOpConvertFToU:
    case SpvOpConvertFToS: {
      assert(message_.in_operand_id_size() == 1 &&
             "Instruction has invalid number of operands");
      assert(message_.fresh_id_size() == 3 &&
             "3 fresh ids must be provided for conversion instructions");

      const auto* type = ir_context->get_type_mgr()->GetType(
          fuzzerutil::GetTypeId(ir_context, message_.in_operand_id(0)));
      assert(type && "Operand has invalid type");

      if (const auto* vector = type->AsVector()) {
        assert(vector->element_type()->AsFloat() &&
               "Conversion to integer supports only operands of scalar or "
               "vector float type");

        return fuzzerutil::FindOrCreateVectorType(
            ir_context, message_.fresh_id(1),
            fuzzerutil::FindOrCreateIntegerType(
                ir_context, message_.fresh_id(2),
                vector->element_type()->AsFloat()->width(),
                opcode == SpvOpConvertFToS),
            vector->element_count());
      } else {
        assert(type->AsFloat() &&
               "Conversion to integer supports only operands of scalar or "
               "vector float type");

        return fuzzerutil::FindOrCreateIntegerType(
            ir_context, message_.fresh_id(1), type->AsFloat()->width(),
            opcode == SpvOpConvertFToS);
      }
    }
    case SpvOpIAdd:
    case SpvOpISub:
    case SpvOpLogicalNot:
    case SpvOpSNegate:
      // Type id of the instruction is equal to the type id of one of the
      // operands. All the necessary checks have been made in the IsApplicable
      // method.
      return fuzzerutil::GetTypeId(ir_context, message_.in_operand_id(0));
    default:
      assert(false && "Inappropriate opcode for equation instruction.");
      return 0;
  }
}

}  // namespace fuzz
}  // namespace spvtools
