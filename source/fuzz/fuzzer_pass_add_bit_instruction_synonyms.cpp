// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/fuzzer_pass_add_bit_instruction_synonyms.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_bit_instruction_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddBitInstructionSynonyms::FuzzerPassAddBitInstructionSynonyms(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

void FuzzerPassAddBitInstructionSynonyms::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& instruction : block) {
        // This fuzzer pass can add a *lot* of ids.  We bail out early if we hit
        // the recommended id limit.
        if (GetIRContext()->module()->id_bound() >=
            GetFuzzerContext()->GetIdBoundLimit()) {
          return;
        }

        // Randomly decides whether the transformation will be applied.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingBitInstructionSynonym())) {
          continue;
        }

        // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3557):
        //  Right now we only support certain operations. When this issue is
        //  addressed the following conditional can use the function
        //  |spvOpcodeIsBit|.
        if (instruction.opcode() != SpvOpBitwiseOr &&
            instruction.opcode() != SpvOpBitwiseXor &&
            instruction.opcode() != SpvOpBitwiseAnd &&
            instruction.opcode() != SpvOpNot) {
          continue;
        }

        // Make sure fuzzer never applies a transformation to a bitwise
        // instruction with differently signed operands.
        if (instruction.opcode() == SpvOpBitwiseOr ||
            instruction.opcode() == SpvOpBitwiseXor ||
            instruction.opcode() == SpvOpBitwiseAnd ||
            instruction.opcode() == SpvOpNot) {
          auto ir_context = this->GetIRContext();

          if (instruction.opcode() == SpvOpNot) {
            auto operand = instruction.GetInOperand(0).words[0];
            auto operand_inst = ir_context->get_def_use_mgr()->GetDef(operand);
            auto operand_type =
                ir_context->get_type_mgr()->GetType(operand_inst->type_id());
            auto operand_sign = operand_type->AsInteger()->IsSigned();

            auto type_id_sign = ir_context->get_type_mgr()
                                    ->GetType(instruction.type_id())
                                    ->AsInteger()
                                    ->IsSigned();

            if (operand_sign != type_id_sign) {
              continue;
            }
          }

          // Other BitWise operations that takes two operands.
          auto first_operand = instruction.GetInOperand(0).words[0];
          auto first_operand_inst =
              ir_context->get_def_use_mgr()->GetDef(first_operand);
          auto first_operand_type = ir_context->get_type_mgr()->GetType(
              first_operand_inst->type_id());
          auto first_operand_sign = first_operand_type->AsInteger()->IsSigned();

          auto second_operand = instruction.GetInOperand(1).words[0];
          auto second_operand_inst =
              ir_context->get_def_use_mgr()->GetDef(second_operand);
          auto second_operand_type = ir_context->get_type_mgr()->GetType(
              second_operand_inst->type_id());
          auto second_operand_sign =
              second_operand_type->AsInteger()->IsSigned();

          auto type_id_sign = ir_context->get_type_mgr()
                                  ->GetType(instruction.type_id())
                                  ->AsInteger()
                                  ->IsSigned();

          if ((first_operand_sign != second_operand_sign) && type_id_sign) {
            continue;
          }
        }

        // Right now, only integer operands are supported.
        if (GetIRContext()
                ->get_type_mgr()
                ->GetType(instruction.type_id())
                ->AsVector()) {
          continue;
        }

        // Make sure all bit indexes are defined as 32-bit unsigned integers.
        uint32_t width = GetIRContext()
                             ->get_type_mgr()
                             ->GetType(instruction.type_id())
                             ->AsInteger()
                             ->width();
        for (uint32_t i = 0; i < width; i++) {
          FindOrCreateIntegerConstant({i}, 32, false, false);
        }

        // Applies the add bit instruction synonym transformation.
        ApplyTransformation(TransformationAddBitInstructionSynonym(
            instruction.result_id(),
            GetFuzzerContext()->GetFreshIds(
                TransformationAddBitInstructionSynonym::GetRequiredFreshIdCount(
                    GetIRContext(), &instruction))));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
