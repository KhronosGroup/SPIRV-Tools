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

#include "source/fuzz/transformation_add_parameter.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddParameter::TransformationAddParameter(
    const protobufs::TransformationAddParameter& message)
    : message_(message) {}

TransformationAddParameter::TransformationAddParameter(
    uint32_t function_id, uint32_t parameter_fresh_id,
    uint32_t initializer_id_or_pointer_type_id,
    uint32_t function_type_fresh_id) {
  message_.set_function_id(function_id);
  message_.set_parameter_fresh_id(parameter_fresh_id);
  message_.set_initializer_id_or_pointer_type_id(
      initializer_id_or_pointer_type_id);
  message_.set_function_type_fresh_id(function_type_fresh_id);
}

bool TransformationAddParameter::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that function exists.
  const auto* function =
      fuzzerutil::FindFunction(ir_context, message_.function_id());
  if (!function ||
      fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
  }

  // Check that |initializer_id| is valid. Consider also the case where
  // |initalizer_id| is a type id of a pointer.
  const auto* initializer_inst = ir_context->get_def_use_mgr()->GetDef(
      message_.initializer_id_or_pointer_type_id());
  opt::analysis::Type* initializer_type;
  if (!initializer_inst) {
    return false;
  }
  auto type = ir_context->get_type_mgr()->GetType(
      message_.initializer_id_or_pointer_type_id());
  if (type && !type->AsPointer()) {
    return false;
  }
  if (type && type->AsPointer()) {
    initializer_type = type;
  } else {
    initializer_type =
        ir_context->get_type_mgr()->GetType(initializer_inst->type_id());
  }

  // Check that initializer's type is valid.
  if (!initializer_type || !IsParameterTypeSupported(*initializer_type)) {
    return false;
  }

  // Consider pointer types separately.
  bool is_valid = true;
  if (initializer_type->kind() == opt::analysis::Type::kPointer) {
    uint32_t pointer_type_id = message_.initializer_id_or_pointer_type_id();
    auto storage_class =
        fuzzerutil::GetStorageClassFromPointerType(ir_context, pointer_type_id);
    switch (storage_class) {
      case SpvStorageClassFunction:
        for (auto* instr :
             fuzzerutil::GetCallers(ir_context, message_.function_id())) {
          auto block = ir_context->get_instr_block(instr);
          auto function_id = block->GetParent()->result_id();
          // If there is no available local variable in at least one caller,
          // the transformation is invalid.
          if (!fuzzerutil::MaybeGetLocalVariable(ir_context, pointer_type_id,
                                                 function_id)) {
            is_valid = false;
            break;
          }
        }
        break;
      case SpvStorageClassPrivate:
      case SpvStorageClassWorkgroup:
        // If there is no available global variable, the transformation is
        // invalid.
        if (!fuzzerutil::MaybeGetGlobalVariable(ir_context, pointer_type_id)) {
          is_valid = false;
        }
        break;
      default:
        break;
    }
    if (!is_valid) {
      return false;
    }
  }

  return fuzzerutil::IsFreshId(ir_context, message_.parameter_fresh_id()) &&
         fuzzerutil::IsFreshId(ir_context, message_.function_type_fresh_id()) &&
         message_.parameter_fresh_id() != message_.function_type_fresh_id();
}  // namespace fuzz

void TransformationAddParameter::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Find the function that will be transformed.
  auto* function = fuzzerutil::FindFunction(ir_context, message_.function_id());
  assert(function && "Can't find the function");

  uint32_t new_parameter_type_id;

  // If the |message_.initializer_id_or_pointer_type_id| is a type id of a
  // pointer then it is the same as |new_parameter_type_id| since in this case,
  // we don't pass an actual initializer id.
  auto type = ir_context->get_type_mgr()->GetType(
      message_.initializer_id_or_pointer_type_id());
  if (type && type->AsPointer()) {
    new_parameter_type_id = message_.initializer_id_or_pointer_type_id();
  } else {
    new_parameter_type_id = fuzzerutil::GetTypeId(
        ir_context, message_.initializer_id_or_pointer_type_id());
  }
  assert(new_parameter_type_id != 0 && "Initializer has invalid type");

  // Add new parameters to the function.
  function->AddParameter(MakeUnique<opt::Instruction>(
      ir_context, SpvOpFunctionParameter, new_parameter_type_id,
      message_.parameter_fresh_id(), opt::Instruction::OperandList()));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.parameter_fresh_id());

  auto new_parameter_type =
      ir_context->get_type_mgr()->GetType(new_parameter_type_id);

  if (new_parameter_type->kind() == opt::analysis::Type::kPointer) {
    // Add a PointeeValueIsIrrelevant fact if the parameter is a pointer.
    transformation_context->GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
        message_.parameter_fresh_id());

    auto storage_class = fuzzerutil::GetStorageClassFromPointerType(
        ir_context, new_parameter_type_id);
    uint32_t available_variable_id = 0;
    switch (storage_class) {
      case SpvStorageClassFunction:
        // Fix all OpFunctionCall instructions. In each function use the
        // available local variable.
        for (auto* inst :
             fuzzerutil::GetCallers(ir_context, function->result_id())) {
          auto block = ir_context->get_instr_block(inst);
          auto function_id = block->GetParent()->result_id();
          available_variable_id = fuzzerutil::MaybeGetLocalVariable(
              ir_context, new_parameter_type_id, function_id);
          assert(available_variable_id != 0 &&
                 "A local variable must be available for the pointer of "
                 "storage class Function.");
          inst->AddOperand({SPV_OPERAND_TYPE_ID, {available_variable_id}});
        }
        break;
      case SpvStorageClassPrivate:
      case SpvStorageClassWorkgroup:
        // Fix all OpFunctionCall instructions. In each function use the
        // available global variable.
        available_variable_id = fuzzerutil::MaybeGetGlobalVariable(
            ir_context, new_parameter_type_id);
        assert(available_variable_id != 0 &&
               "A global variable must be available for the pointer of storage "
               "class Workgroup or Private.");
        for (auto* inst :
             fuzzerutil::GetCallers(ir_context, function->result_id())) {
          inst->AddOperand({SPV_OPERAND_TYPE_ID, {available_variable_id}});
        }
        break;
      default:
        break;
    }
  } else {
    // Fix all OpFunctionCall instructions.
    for (auto* inst :
         fuzzerutil::GetCallers(ir_context, function->result_id())) {
      inst->AddOperand({SPV_OPERAND_TYPE_ID,
                        {message_.initializer_id_or_pointer_type_id()}});
    }
  }

  // Mark new parameter as irrelevant so that we can replace its use with some
  // other id.
  transformation_context->GetFactManager()->AddFactIdIsIrrelevant(
      message_.parameter_fresh_id());

  // Update function's type.
  {
    // We use a separate scope here since |old_function_type| might become a
    // dangling pointer after the call to the fuzzerutil::UpdateFunctionType.

    const auto* old_function_type =
        fuzzerutil::GetFunctionType(ir_context, function);
    assert(old_function_type && "Function must have a valid type");

    std::vector<uint32_t> parameter_type_ids;
    for (uint32_t i = 1; i < old_function_type->NumInOperands(); ++i) {
      parameter_type_ids.push_back(
          old_function_type->GetSingleWordInOperand(i));
    }

    parameter_type_ids.push_back(new_parameter_type_id);

    fuzzerutil::UpdateFunctionType(
        ir_context, function->result_id(), message_.function_type_fresh_id(),
        old_function_type->GetSingleWordInOperand(0), parameter_type_ids);
  }

  // Make sure our changes are analyzed.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationAddParameter::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_parameter() = message_;
  return result;
}

bool TransformationAddParameter::IsParameterTypeSupported(
    const opt::analysis::Type& type) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3403):
  //  Think about other type instructions we can add here.
  switch (type.kind()) {
    case opt::analysis::Type::kBool:
    case opt::analysis::Type::kInteger:
    case opt::analysis::Type::kFloat:
    case opt::analysis::Type::kArray:
    case opt::analysis::Type::kMatrix:
    case opt::analysis::Type::kVector:
      return true;
    case opt::analysis::Type::kStruct:
      return std::all_of(type.AsStruct()->element_types().begin(),
                         type.AsStruct()->element_types().end(),
                         [](const opt::analysis::Type* element_type) {
                           return IsParameterTypeSupported(*element_type);
                         });
    case opt::analysis::Type::kPointer: {
      auto storage_class = type.AsPointer()->storage_class();
      switch (storage_class) {
        case SpvStorageClassPrivate:
        case SpvStorageClassFunction:
        case SpvStorageClassWorkgroup: {
          auto pointee_type = type.AsPointer()->pointee_type();
          return IsParameterTypeSupported(*pointee_type);
        }
        default:
          return false;
      }
    }
    default:
      return false;
  }
}

}  // namespace fuzz
}  // namespace spvtools
