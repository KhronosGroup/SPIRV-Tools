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
#include "source/fuzz/transformation_add_parameter.h"

#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_local_variable.h"

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
      std::map<uint32_t, uint32_t> call_parameter_id;

      if (current_type->kind() == opt::analysis::Type::kPointer) {
        auto storage_class = fuzzerutil::GetStorageClassFromPointerType(
            GetIRContext(), current_type_id);
        switch (storage_class) {
          case SpvStorageClassFunction: {
            for (auto* instr :
                 fuzzerutil::GetCallers(GetIRContext(), function.result_id())) {
              auto block = GetIRContext()->get_instr_block(instr);
              auto function_id = block->GetParent()->result_id();
              uint32_t variable_id =
                  FindOrCreateLocalVariable(current_type_id, function_id, true);
              call_parameter_id[instr->result_id()] = variable_id;
            }
          } break;
          case SpvStorageClassPrivate:
          case SpvStorageClassWorkgroup: {
            uint32_t variable_id =
                FindOrCreateGlobalVariable(current_type_id, true);
            for (auto* instr :
                 fuzzerutil::GetCallers(GetIRContext(), function.result_id())) {
              call_parameter_id[instr->result_id()] = variable_id;
            }
          } break;
          default:
            break;
        }
      } else {
        uint32_t constant_id = FindOrCreateZeroConstant(current_type_id, true);
        for (auto* instr :
             fuzzerutil::GetCallers(GetIRContext(), function.result_id())) {
          call_parameter_id[instr->result_id()] = constant_id;
        }
        if (call_parameter_id.empty()) {
          call_parameter_id[0] = constant_id;
        }
      }
      // If the function has no callers, and a zero constant of the selected
      // type to the key 0. It is necessary to pass information of the new type
      // to the transformation.
      /*if (call_parameter_id.empty()) {
        uint32_t value_id;
        if (current_type->kind() == opt::analysis::Type::kPointer) {
          uint32_t pointee_type_id =
              fuzzerutil::GetPointeeTypeIdFromPointerType(GetIRContext(),
                                                          current_type_id);
          value_id = FindOrCreateZeroConstant(pointee_type_id, true);
        } else {
          value_id = FindOrCreateZeroConstant(current_type_id, true);
        }
      }*/
      ApplyTransformation(TransformationAddParameter(
          function.result_id(), GetFuzzerContext()->GetFreshId(),
          std::move(call_parameter_id), GetFuzzerContext()->GetFreshId()));
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

uint32_t FuzzerPassAddParameters::FindOrCreateLocalVariable(
    uint32_t pointer_type_id, uint32_t function_id,
    bool pointee_value_is_irrelevant) {
  uint32_t result_id = 0;
  auto pointer_type = GetIRContext()->get_type_mgr()->GetType(pointer_type_id);
  // No unused variables in release mode.
  (void)pointer_type;
  assert(pointer_type->AsPointer() &&
         "The pointer_type_id must refer to a pointer type");
  auto function = fuzzerutil::FindFunction(GetIRContext(), function_id);
  assert(function && "The function must be defined.");
  // All of the local variable declarations are located in the first block.
  auto block = function->begin();
  for (auto& instruction : *block) {
    if (instruction.opcode() != SpvOpVariable) {
      continue;
    }
    if (!instruction.type_id() || instruction.type_id() != pointer_type_id) {
      continue;
    }

    // Check if the found variable is marked with PointeeValueIsIrrelevant.
    if (!GetTransformationContext()->GetFactManager()->PointeeValueIsIrrelevant(
            instruction.result_id())) {
      continue;
    }
    result_id = instruction.result_id();
    break;
  }

  if (!result_id) {
    uint32_t pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
        GetIRContext(), pointer_type_id);
    result_id = GetFuzzerContext()->GetFreshId();
    ApplyTransformation(TransformationAddLocalVariable(
        result_id, pointer_type_id, function_id,
        FindOrCreateZeroConstant(pointee_type_id, true),
        pointee_value_is_irrelevant));
  }
  return result_id;
}

uint32_t FuzzerPassAddParameters::FindOrCreateGlobalVariable(
    uint32_t pointer_type_id, bool pointee_value_is_irrelevant) {
  uint32_t result_id = 0;
  auto pointer_type = GetIRContext()->get_type_mgr()->GetType(pointer_type_id);
  // No unused variables in release mode.
  (void)pointer_type;
  assert(pointer_type->AsPointer() &&
         "The pointer_type_id must refer to a pointer type");
  for (auto& instruction : GetIRContext()->module()->types_values()) {
    if (instruction.opcode() != SpvOpVariable) {
      continue;
    }
    if (!instruction.type_id() || instruction.type_id() != pointer_type_id) {
      continue;
    }

    // Check if the found variable is marked with PointeeValueIsIrrelevant.
    if (!GetTransformationContext()->GetFactManager()->PointeeValueIsIrrelevant(
            instruction.result_id())) {
      continue;
    }
    result_id = instruction.result_id();
    break;
  }
  if (!result_id) {
    uint32_t pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
        GetIRContext(), pointer_type_id);
    auto storage_class = fuzzerutil::GetStorageClassFromPointerType(
        GetIRContext(), pointer_type_id);
    assert((storage_class == SpvStorageClassPrivate ||
            storage_class == SpvStorageClassWorkgroup) &&
           "The storage class must be Private or Workgroup");
    result_id = GetFuzzerContext()->GetFreshId();
    ApplyTransformation(TransformationAddGlobalVariable(
        result_id, pointer_type_id, storage_class,
        FindOrCreateZeroConstant(pointee_type_id, true),
        pointee_value_is_irrelevant));
  }
  return result_id;
}
}  // namespace fuzz

}  // namespace spvtools
