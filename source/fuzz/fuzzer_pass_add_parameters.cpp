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
      auto current_type_id =
          type_candidates[GetFuzzerContext()->RandomIndex(type_candidates)];
      auto current_type =
          GetIRContext()->get_type_mgr()->GetType(current_type_id);

      // Consider pointer types separately.
      if (current_type->kind() == opt::analysis::Type::kPointer) {
        auto storage_class = fuzzerutil::GetStorageClassFromPointerType(
            GetIRContext(), current_type_id);
        uint32_t pointee_type_id =
            (GetIRContext()->get_def_use_mgr()->GetDef(current_type_id))
                ->GetSingleWordInOperand(1);
        uint32_t variable_id = 0;

        // Make the transformation applicable.
        switch (storage_class) {
          case SpvStorageClassFunction: {
            for (auto* instr :
                 fuzzerutil::GetCallers(GetIRContext(), function.result_id())) {
              auto block = GetIRContext()->get_instr_block(instr);
              auto function_id = block->GetParent()->result_id();
              // If there is no available local variable, then create one. The
              // available local variable must be marked with the fact
              // PointeeValueIsIrrelevant.
              variable_id = fuzzerutil::MaybeGetLocalVariable(
                  GetIRContext(), current_type_id, function_id);
              if (!variable_id ||
                  !GetTransformationContext()
                       ->GetFactManager()
                       ->PointeeValueIsIrrelevant(variable_id)) {
                ApplyTransformation(TransformationAddLocalVariable(
                    GetFuzzerContext()->GetFreshId(), current_type_id,
                    function_id,
                    FindOrCreateZeroConstant(pointee_type_id, true), true));
              }
            }
          } break;
          case SpvStorageClassPrivate:
          case SpvStorageClassWorkgroup: {
            // If there is no available global variable, then create one. The
            // available global variable must be marked with the fact
            // PointeeValueIsIrrelevant.
            variable_id = fuzzerutil::MaybeGetGlobalVariable(GetIRContext(),
                                                             current_type_id);
            if (!variable_id || !GetTransformationContext()
                                     ->GetFactManager()
                                     ->PointeeValueIsIrrelevant(variable_id)) {
              ApplyTransformation(TransformationAddGlobalVariable(
                  GetFuzzerContext()->GetFreshId(), current_type_id,
                  storage_class,
                  FindOrCreateZeroConstant(pointee_type_id, true), true));
            }
          } break;
          default:
            break;
        }

        ApplyTransformation(TransformationAddParameter(
            function.result_id(), GetFuzzerContext()->GetFreshId(),
            current_type_id, GetFuzzerContext()->GetFreshId()));

      } else  // Consider non-pointer parameters.
      {
        ApplyTransformation(TransformationAddParameter(
            function.result_id(), GetFuzzerContext()->GetFreshId(),
            // We mark the constant as irrelevant so that we can replace it with
            // a more interesting value later.
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