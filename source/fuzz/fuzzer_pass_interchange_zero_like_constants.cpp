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

#include "source/fuzz/fuzzer_pass_interchange_zero_like_constants.h"

#include <source/opt/instruction.h>
#include <source/opt/type_manager.h>

#include "source/fuzz/id_use_descriptor.h"
#include "source/fuzz/transformation_add_constant_null.h"
#include "source/fuzz/transformation_record_synonymous_constants.h"
#include "source/fuzz/transformation_replace_id_with_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassInterchangeZeroLikeConstants::FuzzerPassInterchangeZeroLikeConstants(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassInterchangeZeroLikeConstants::
    ~FuzzerPassInterchangeZeroLikeConstants() = default;

inline uint32_t
FuzzerPassInterchangeZeroLikeConstants::FindOrCreateToggledConstant(
    opt::Instruction* declaration) {
  auto constant = GetIRContext()->get_constant_mgr()->FindDeclaredConstant(
      declaration->result_id());

  // If constant is not zero, cannot toggle it
  if (!constant->IsZero()) {
    return 0;
  }

  if (constant->AsScalarConstant()) {
    // Search existing declaration
    opt::analysis::NullConstant null_constant(
        GetIRContext()->get_type_mgr()->GetType(declaration->type_id()));
    auto existing_constant =
        GetIRContext()->get_constant_mgr()->FindConstant(&null_constant);

    if (existing_constant) {
      return GetIRContext()
          ->get_constant_mgr()
          ->GetDefiningInstruction(existing_constant)
          ->result_id();
    }
    // If not found, create ConstantNull
    uint32_t fresh_id = GetFuzzerContext()->GetFreshId();
    ApplyTransformation(
        TransformationAddConstantNull(fresh_id, declaration->type_id()));
    return fresh_id;

  } else if (constant->AsNullConstant()) {
    // Add declaration of equivalent scalar constant
    auto kind = constant->type()->kind();
    if (kind == opt::analysis::Type::kBool ||
        kind == opt::analysis::Type::kInteger ||
        kind == opt::analysis::Type::kFloat) {
      return FindOrCreateZeroConstant(declaration->type_id());
    }
  }

  return 0;
}

void FuzzerPassInterchangeZeroLikeConstants::Apply() {
  // Find the next fresh id

  // Loop through all the zero-like constants
  auto constants = GetIRContext()->GetConstants();
  for (auto constant : constants) {
    uint32_t constant_id = constant->result_id();
    if (uint32_t toggled_id = FindOrCreateToggledConstant(constant)) {
      // Record synonymous constants
      ApplyTransformation(
          TransformationRecordSynonymousConstants(constant_id, toggled_id));

      // Find all the uses of the constant and add them to the vector, because
      // further transformation could invalidate the def_use manager.
      std::vector<protobufs::IdUseDescriptor> uses;
      GetIRContext()->get_def_use_mgr()->ForEachUse(
          constant_id,
          [this, &uses](opt::Instruction* use_inst,
                        uint32_t use_index) -> void {
            auto block_containing_use =
                GetIRContext()->get_instr_block(use_inst);
            // Only consider the use if it actually is in a block.
            if (block_containing_use) {
              // Get index of the operand with respect to just the input
              // operands
              // (|use_index| refers to all the operands, not just the In
              // operands)
              uint32_t in_operand_index = use_index - use_inst->NumOperands() +
                                          use_inst->NumInOperands();
              auto id_use_descriptor = MakeIdUseDescriptorFromUse(
                  GetIRContext(), use_inst, in_operand_index);
              uses.emplace_back(id_use_descriptor);
            }
          });

      for (const auto& id_use_descriptor : uses) {
        // Probabilistically decide whether to apply the transformation
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()
                    ->GetChanceOfInterchangingZeroLikeConstants())) {
          continue;
        }

        // Replace id with synonym
        ApplyTransformation(
            TransformationReplaceIdWithSynonym(id_use_descriptor, toggled_id));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools