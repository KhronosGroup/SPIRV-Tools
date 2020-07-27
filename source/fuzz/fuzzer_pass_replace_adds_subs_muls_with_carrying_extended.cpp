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

#include "source/fuzz/fuzzer_pass_replace_adds_subs_muls_with_carrying_extended.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_replace_add_sub_mul_with_carrying_extended.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceAddsSubsMulsWithCarryingExtended::
    FuzzerPassReplaceAddsSubsMulsWithCarryingExtended(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceAddsSubsMulsWithCarryingExtended::
    ~FuzzerPassReplaceAddsSubsMulsWithCarryingExtended() = default;

void FuzzerPassReplaceAddsSubsMulsWithCarryingExtended::Apply() {
  GetIRContext()->module()->ForEachInst([this](opt::Instruction* instruction) {

    // Randomly decide whether to apply the transformation.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfReplacingAddSubMulWithCarryingExtended())) {
      return;
    }

    // Check if the transformation can be applied (opcode, signedness).
    auto instruction_opcode = instruction->opcode();
    if (instruction_opcode != SpvOpIAdd && instruction_opcode != SpvOpISub &&
        instruction_opcode != SpvOpIMul) {
      return;
    }

    uint32_t operand_1_type_id =
        GetIRContext()
            ->get_def_use_mgr()
            ->GetDef(instruction->GetSingleWordOperand(2))
            ->type_id();

    uint32_t operand_2_type_id =
        GetIRContext()
            ->get_def_use_mgr()
            ->GetDef(instruction->GetSingleWordOperand(3))
            ->type_id();

    uint32_t operand_1_signedness = GetIRContext()
                                        ->get_def_use_mgr()
                                        ->GetDef(operand_1_type_id)
                                        ->GetSingleWordOperand(2);
    uint32_t operand_2_signedness = GetIRContext()
                                        ->get_def_use_mgr()
                                        ->GetDef(operand_2_type_id)
                                        ->GetSingleWordOperand(2);
    switch (instruction_opcode) {
      case SpvOpIAdd:
      case SpvOpISub:
        if (operand_1_signedness != 0 || operand_2_signedness != 0) return;
        break;
      default:
        break;
    }

    // Check if the required struct type already exists.
    std::vector<uint32_t> operand_type_ids;
    operand_type_ids.push_back(operand_1_type_id);
    operand_type_ids.push_back(operand_2_type_id);
    uint32_t struct_type_id =
        fuzzerutil::MaybeGetStructType(GetIRContext(), operand_type_ids);
    if (struct_type_id == 0) {
      // If not, get a fresh id and add the type.
      struct_type_id = GetFuzzerContext()->GetFreshId();
      fuzzerutil::AddStructType(GetIRContext(), struct_type_id,
                                operand_type_ids);
    }
    ApplyTransformation(TransformationReplaceAddSubMulWithCarryingExtended(
        GetFuzzerContext()->GetFreshId(), struct_type_id,
        instruction->result_id()));
  });
}
}  // namespace fuzz
}  // namespace spvtools
