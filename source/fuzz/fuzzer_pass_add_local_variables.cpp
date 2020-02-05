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

#include "source/fuzz/fuzzer_pass_add_local_variables.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_constant_composite.h"
#include "source/fuzz/transformation_add_local_variable.h"
#include "source/fuzz/transformation_add_type_pointer.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddLocalVariables::FuzzerPassAddLocalVariables(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassAddLocalVariables::~FuzzerPassAddLocalVariables() = default;

void FuzzerPassAddLocalVariables::Apply() {
  // Records all of the base types available in the module.
  std::vector<uint32_t> base_type_ids;

  // For each base type, records all the associated pointer types that target
  // that base type and that have the Function storage class.
  std::map<uint32_t, std::vector<uint32_t>> base_type_to_pointer;

  for (auto& inst : GetIRContext()->types_values()) {
    switch (inst.opcode()) {
      case SpvOpTypeArray:
      case SpvOpTypeBool:
      case SpvOpTypeFloat:
      case SpvOpTypeInt:
      case SpvOpTypeMatrix:
      case SpvOpTypeStruct:
      case SpvOpTypeVector:
        // These types are suitable as pointer base types.  Record the type,
        // and the fact that we cannot yet have seen any pointers that use this
        // as its base type.
        base_type_ids.push_back(inst.result_id());
        base_type_to_pointer.insert({inst.result_id(), {}});
        break;
      case SpvOpTypePointer:
        if (inst.GetSingleWordInOperand(0) == SpvStorageClassFunction) {
          // The pointer has Function storage class, so we are interested in it.
          // Associate it with its base type.
          base_type_to_pointer.at(inst.GetSingleWordInOperand(1))
              .push_back(inst.result_id());
        }
        break;
      default:
        break;
    }
  }

  for (auto& function : *GetIRContext()->module()) {
    while (GetFuzzerContext()->ChoosePercentage(
        GetFuzzerContext()->GetChanceOfAddingLocalVariable())) {
      uint32_t base_type_id =
          base_type_ids[GetFuzzerContext()->RandomIndex(base_type_ids)];
      uint32_t pointer_type_id;
      std::vector<uint32_t>& available_pointers =
          base_type_to_pointer.at(base_type_id);
      if (available_pointers.empty()) {
        pointer_type_id = GetFuzzerContext()->GetFreshId();
        available_pointers.push_back(pointer_type_id);
        ApplyTransformation(TransformationAddTypePointer(
            pointer_type_id, SpvStorageClassFunction, base_type_id));
      } else {
        pointer_type_id = available_pointers[GetFuzzerContext()->RandomIndex(
            available_pointers)];
      }
      ApplyTransformation(TransformationAddLocalVariable(
          GetFuzzerContext()->GetFreshId(), pointer_type_id,
          function.result_id(), FindOrCreateZeroConstant(base_type_id), true));
    }
  }
}

uint32_t FuzzerPassAddLocalVariables::FindOrCreateZeroConstant(
    uint32_t scalar_or_composite_type_id) {
  auto type_instruction =
      GetIRContext()->get_def_use_mgr()->GetDef(scalar_or_composite_type_id);
  assert(type_instruction && "The type instruction must exist.");
  switch (type_instruction->opcode()) {
    case SpvOpTypeBool:
      return FindOrCreateBoolConstant(false);
    case SpvOpTypeFloat:
      return FindOrCreate32BitFloatConstant(0);
    case SpvOpTypeInt:
      return FindOrCreate32BitIntegerConstant(
          0, type_instruction->GetSingleWordInOperand(1) != 0);
    case SpvOpTypeArray: {
      return GetZeroConstantForHomogeneousComposite(
          *type_instruction, type_instruction->GetSingleWordInOperand(0),
          fuzzerutil::GetArraySize(*type_instruction, GetIRContext()),
          [](const opt::analysis::Type& composite_type,
             const std::vector<const opt::analysis::Constant*>&
                 component_constants) {
            return MakeUnique<opt::analysis::ArrayConstant>(
                composite_type.AsArray(), component_constants);
          });
    }
    case SpvOpTypeMatrix: {
      return GetZeroConstantForHomogeneousComposite(
          *type_instruction, type_instruction->GetSingleWordInOperand(0),
          type_instruction->GetSingleWordInOperand(1),
          [](const opt::analysis::Type& composite_type,
             const std::vector<const opt::analysis::Constant*>&
                 component_constants) {
            return MakeUnique<opt::analysis::MatrixConstant>(
                composite_type.AsMatrix(), component_constants);
          });
    }
    case SpvOpTypeStruct: {
      std::vector<const opt::analysis::Constant*> field_zero_constants;
      std::vector<uint32_t> field_zero_ids;
      for (uint32_t index = 0; index < type_instruction->NumInOperands();
           index++) {
        uint32_t field_constant_id = FindOrCreateZeroConstant(
            type_instruction->GetSingleWordInOperand(index));
        field_zero_ids.push_back(field_constant_id);
        field_zero_constants.push_back(
            GetIRContext()->get_constant_mgr()->GetConstantFromInst(
                GetIRContext()->get_def_use_mgr()->GetDef(field_constant_id)));
      }
      return FindOrCreateCompositeConstant(
          *type_instruction, field_zero_constants, field_zero_ids,
          [](const opt::analysis::Type& composite_type,
             const std::vector<const opt::analysis::Constant*>&
                 component_constants) {
            return MakeUnique<opt::analysis::StructConstant>(
                composite_type.AsStruct(), component_constants);
          });
    }
    case SpvOpTypeVector: {
      return GetZeroConstantForHomogeneousComposite(
          *type_instruction, type_instruction->GetSingleWordInOperand(0),
          type_instruction->GetSingleWordInOperand(1),
          [](const opt::analysis::Type& composite_type,
             const std::vector<const opt::analysis::Constant*>&
                 component_constants) {
            return MakeUnique<opt::analysis::VectorConstant>(
                composite_type.AsVector(), component_constants);
          });
    }
    default:
      assert(false && "Unknown type.");
      return 0;
  }
}

uint32_t FuzzerPassAddLocalVariables::FindOrCreateCompositeConstant(
    const opt::Instruction& composite_type_instruction,
    const std::vector<const opt::analysis::Constant*>& constants,
    const std::vector<uint32_t>& constant_ids,
    const CompositeConstantSupplier& composite_constant_supplier) {
  uint32_t existing_constant =
      GetIRContext()->get_constant_mgr()->FindDeclaredConstant(
          composite_constant_supplier(
              *GetIRContext()->get_type_mgr()->GetType(
                  composite_type_instruction.result_id()),
              constants)
              .get(),
          composite_type_instruction.result_id());
  if (existing_constant) {
    return existing_constant;
  }
  uint32_t result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddConstantComposite(
      result, composite_type_instruction.result_id(), constant_ids));
  return result;
}

uint32_t FuzzerPassAddLocalVariables::GetZeroConstantForHomogeneousComposite(
    const opt::Instruction& composite_type_instruction,
    uint32_t component_type_id, uint32_t num_components,
    const CompositeConstantSupplier& composite_constant_supplier) {
  std::vector<const opt::analysis::Constant*> zero_constants;
  std::vector<uint32_t> zero_ids;
  uint32_t zero_component = FindOrCreateZeroConstant(component_type_id);
  const opt::analysis::Constant* registered_zero_component =
      GetIRContext()->get_constant_mgr()->GetConstantFromInst(
          GetIRContext()->get_def_use_mgr()->GetDef(zero_component));
  for (uint32_t i = 0; i < num_components; i++) {
    zero_constants.push_back(registered_zero_component);
    zero_ids.push_back(zero_component);
  }
  return FindOrCreateCompositeConstant(composite_type_instruction,
                                       zero_constants, zero_ids,
                                       composite_constant_supplier);
}

}  // namespace fuzz
}  // namespace spvtools
