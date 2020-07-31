// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/fuzzer_pass_add_parameters.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_local_variable.h"
#include "source/fuzz/transformation_add_parameter.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddParameters::FuzzerPassAddParameters(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddParameters::~FuzzerPassAddParameters() = default;

void FuzzerPassAddParameters::Apply() {
  // Compute type candidates for the new parameter.
  std::vector<uint32_t> type_candidates;
  for (const auto& type_inst : GetIRContext()->module()->GetTypes()) {
    const auto* type =
        GetIRContext()->get_type_mgr()->GetType(type_inst->result_id());
    assert(type && "Type instruction is not registered in the type manager");

    if (TransformationAddParameter::IsParameterTypeSupported(*type)) {
      type_candidates.push_back(type_inst->result_id());
    } else if (type->kind() == opt::analysis::Type::kPointer) {
      // Pointer types with storage class Private and Function are allowed.
      SpvStorageClass storage_class =
          fuzzerutil::GetStorageClassFromPointerType(GetIRContext(),
                                                     type_inst->result_id());
      if (storage_class == SpvStorageClassPrivate ||
          storage_class == SpvStorageClassFunction) {
        type_candidates.push_back(type_inst->result_id());
      }
    }
  }

  if (type_candidates.empty()) {
    // The module contains no suitable types to use in new parameters.
    return;
  }

  // Iterate over all functions in the module.
  for (const auto& function : *GetIRContext()->module()) {
    // Skip all entry-point functions - we don't want to change those.
    if (fuzzerutil::FunctionIsEntryPoint(GetIRContext(),
                                         function.result_id())) {
      continue;
    }

    if (GetNumberOfParameters(function) >=
        GetFuzzerContext()->GetMaximumNumberOfFunctionParameters()) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfAddingParameters())) {
      continue;
    }

    auto num_new_parameters =
        GetFuzzerContext()->GetRandomNumberOfNewParameters(
            GetNumberOfParameters(function));
    for (uint32_t i = 0; i < num_new_parameters; ++i) {
      uint32_t current_type_id =
          type_candidates[GetFuzzerContext()->RandomIndex(type_candidates)];
      auto current_type =
          GetIRContext()->get_type_mgr()->GetType(current_type_id);
      auto current_instr =
          GetIRContext()->get_def_use_mgr()->GetDef(current_type_id);

      if (current_type->kind() == opt::analysis::Type::kPointer) {
        auto current_pointee_type_id =
            fuzzerutil::GetPointeeTypeIdFromPointerType(GetIRContext(),
                                                        current_type_id);
        auto storage_class =
            fuzzerutil::GetStorageClassFromPointerType(current_instr);

        // Look for available variables that have the type |current_type|.
        std::vector<uint32_t> available_variable_ids;
        GetIRContext()->module()->ForEachInst(
            [this, &available_variable_ids,
             current_type_id](opt::Instruction* instruction) {
              if (instruction->opcode() != SpvOpVariable) {
                return;
              }
              if (instruction->type_id() != current_type_id) {
                return;
              }
              available_variable_ids.push_back(instruction->result_id());
            });
        uint32_t initializer_id =
            FindOrCreateZeroConstant(current_pointee_type_id, true);

        // If there are no such variables, then create one. The value is
        // irrelevant.
        if (available_variable_ids.empty()) {
          if (storage_class == SpvStorageClassPrivate) {
            ApplyTransformation(TransformationAddGlobalVariable(
                GetFuzzerContext()->GetFreshId(), current_type_id,
                SpvStorageClassPrivate, initializer_id, true));
          } else if (storage_class == SpvStorageClassFunction) {
            ApplyTransformation(TransformationAddLocalVariable(
                GetFuzzerContext()->GetFreshId(), current_type_id,
                function.result_id(), initializer_id, true));
          }
        }
        // Add a parameter with the created initializer.
        ApplyTransformation(TransformationAddParameter(
            function.result_id(), GetFuzzerContext()->GetFreshId(),
            initializer_id, GetFuzzerContext()->GetFreshId()));

      } else {
        ApplyTransformation(TransformationAddParameter(
            function.result_id(), GetFuzzerContext()->GetFreshId(),
            // We mark the constant as irrelevant so that we can replace it
            // with a more interesting value later.
            FindOrCreateZeroConstant(current_type_id, true),
            GetFuzzerContext()->GetFreshId()));
      }
    }
  }
}

uint32_t FuzzerPassAddParameters::GetNumberOfParameters(
    const opt::Function& function) const {
  const auto* type = GetIRContext()->get_type_mgr()->GetType(
      function.DefInst().GetSingleWordInOperand(1));
  assert(type && type->AsFunction());

  return static_cast<uint32_t>(type->AsFunction()->param_types().size());
}

}  // namespace fuzz
}  // namespace spvtools
