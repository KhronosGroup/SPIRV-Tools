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

#include "source/fuzz/transformation_replace_parameter_with_global.h"

#include <vector>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {
namespace {

opt::Function* GetFunctionFromParameterId(opt::IRContext* ir_context,
                                          uint32_t param_id) {
  auto* param_inst = ir_context->get_def_use_mgr()->GetDef(param_id);
  assert(param_inst && "Parameter id is invalid");

  for (auto& function : *ir_context->module()) {
    if (fuzzerutil::InstructionIsFunctionParameter(param_inst, &function)) {
      return &function;
    }
  }

  return nullptr;
}

}  // namespace

TransformationReplaceParameterWithGlobal::
    TransformationReplaceParameterWithGlobal(
        const protobufs::TransformationReplaceParameterWithGlobal& message)
    : message_(message) {}

TransformationReplaceParameterWithGlobal::
    TransformationReplaceParameterWithGlobal(uint32_t function_type_fresh_id,
                                             uint32_t parameter_id,
                                             uint32_t global_variable_fresh_id,
                                             uint32_t initializer_id) {
  message_.set_function_type_fresh_id(function_type_fresh_id);
  message_.set_parameter_id(parameter_id);
  message_.set_global_variable_fresh_id(global_variable_fresh_id);
  message_.set_initializer_id(initializer_id);
}

bool TransformationReplaceParameterWithGlobal::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that |parameter_id| is valid.
  const auto* param_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.parameter_id());
  if (!param_inst || param_inst->opcode() != SpvOpFunctionParameter) {
    return false;
  }

  // Check that function exists and is not an entry point.
  const auto* function =
      GetFunctionFromParameterId(ir_context, message_.parameter_id());
  if (!function ||
      fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
  }

  // We already know that the function has at least one parameter -
  // |parameter_id|.

  // Check that replaced parameter has valid type.
  const auto* param_type =
      ir_context->get_type_mgr()->GetType(param_inst->type_id());
  assert(param_type && "Parameter has invalid type");
  if (!CanReplaceFunctionParameterType(*param_type)) {
    return false;
  }

  auto* param_type_inst =
      ir_context->get_def_use_mgr()->GetDef(param_inst->type_id());
  assert(param_type_inst && "Parameter type must exist");

  // Initializer id can be 0 iff parameter is a pointer with Workgroup storage
  // class.
  if (message_.initializer_id() == 0) {
    return param_type->AsPointer() &&
           param_type->AsPointer()->storage_class() == SpvStorageClassWorkgroup;
  }

  // If |initializer_id| is non-zero then parameter can't be a pointer with
  // Workgroup storage class.
  if (param_type->AsPointer() &&
      param_type->AsPointer()->storage_class() == SpvStorageClassWorkgroup) {
    return false;
  }

  auto pointee_type_id =
      param_type->AsPointer()
          ? fuzzerutil::GetPointeeTypeIdFromPointerType(param_type_inst)
          : param_type_inst->result_id();

  // Check that initializer has valid type.
  if (fuzzerutil::GetTypeId(ir_context, message_.initializer_id()) !=
      pointee_type_id) {
    return false;
  }

  // Check that pointer type for a global variable exists.
  if (!fuzzerutil::MaybeGetPointerType(
          ir_context, pointee_type_id,
          GetStorageClassForGlobalVariable(ir_context,
                                           param_inst->type_id()))) {
    return false;
  }

  return fuzzerutil::IsFreshId(ir_context, message_.function_type_fresh_id()) &&
         fuzzerutil::IsFreshId(ir_context,
                               message_.global_variable_fresh_id()) &&
         message_.function_type_fresh_id() !=
             message_.global_variable_fresh_id();
}

void TransformationReplaceParameterWithGlobal::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  const auto* param_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.parameter_id());
  assert(param_inst && "Parameter must exist");

  auto* param_type_inst =
      ir_context->get_def_use_mgr()->GetDef(param_inst->type_id());
  assert(param_type_inst && "Parameter must have a valid type");

  // Get pointee type id for a global variable.
  auto pointee_type_id =
      param_type_inst->opcode() == SpvOpTypePointer
          ? fuzzerutil::GetPointeeTypeIdFromPointerType(param_type_inst)
          : param_inst->type_id();

  auto global_variable_storage_class =
      GetStorageClassForGlobalVariable(ir_context, param_inst->type_id());

  auto is_local_pointer_or_scalar =
      param_type_inst->opcode() != SpvOpTypePointer ||
      fuzzerutil::GetStorageClassFromPointerType(param_type_inst) ==
          SpvStorageClassFunction;

  // If parameter is not a pointer, we use a fresh id for a global variable and
  // insert an OpLoad instruction to load parameter's value. If it's a pointer
  // with Function storage class, we use a fresh id for a global variable and
  // create a local variable to store parameter's value. Otherwise, we can reuse
  // parameter's id for a global variable.
  auto global_variable_result_id = is_local_pointer_or_scalar
                                       ? message_.global_variable_fresh_id()
                                       : message_.parameter_id();

  // Create global variable to store parameter's value.
  fuzzerutil::AddGlobalVariable(
      ir_context, global_variable_result_id,
      fuzzerutil::MaybeGetPointerType(ir_context, pointee_type_id,
                                      global_variable_storage_class),
      global_variable_storage_class, message_.initializer_id());

  auto* function =
      GetFunctionFromParameterId(ir_context, message_.parameter_id());
  assert(function && "Function must exist");

  if (is_local_pointer_or_scalar) {
    // Add a local variable to store parameter's value if it's a pointer with
    // Function storage class.
    if (param_type_inst->opcode() == SpvOpTypePointer) {
      fuzzerutil::AddLocalVariable(ir_context, param_inst->result_id(),
                                   param_inst->type_id(), function->result_id(),
                                   message_.initializer_id());
    }

    // Insert OpLoad or OpCopyMemory instruction right after OpVariable
    // instructions. The decision on which instruction to insert is made based
    // on the parameter's type.
    auto it = function->begin()->begin();
    while (it != function->begin()->end() &&
           !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, it)) {
      ++it;
    }

    assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, it) &&
           "Can't insert OpLoad or OpCopyMemory into the first basic block of "
           "the function");

    if (param_type_inst->opcode() != SpvOpTypePointer) {
      it.InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpLoad, param_inst->type_id(), param_inst->result_id(),
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {global_variable_result_id}}}));
    } else {
      it.InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpCopyMemory, 0, 0,
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {param_inst->result_id()}},
              {SPV_OPERAND_TYPE_ID, {global_variable_result_id}}}));
    }

    // If parameter is not a pointer, the condition will fail. Otherwise, we
    // mark the global variable as irrelevant if parameter's pointee is
    // irrelevant.
    if (transformation_context->GetFactManager()->PointeeValueIsIrrelevant(
            message_.parameter_id())) {
      transformation_context->GetFactManager()
          ->AddFactValueOfPointeeIsIrrelevant(global_variable_result_id);
    }
  }

  // Calculate the index of the replaced parameter (we need to know this to
  // remove operands from OpFunctionCall.
  auto params = fuzzerutil::GetParameters(ir_context, function->result_id());
  auto parameter_index = static_cast<uint32_t>(params.size());
  for (uint32_t i = 0, n = static_cast<uint32_t>(params.size()); i < n; ++i) {
    if (params[i]->result_id() == message_.parameter_id()) {
      parameter_index = i;
      break;
    }
  }

  assert(parameter_index != params.size() &&
         "Parameter must exist in the function");

  // Update all OpFunctionCall.
  ir_context->get_def_use_mgr()->ForEachUser(
      function->result_id(),
      [ir_context, param_type_inst, parameter_index,
       global_variable_result_id](opt::Instruction* inst) {
        if (inst->opcode() != SpvOpFunctionCall) {
          return;
        }

        auto it = fuzzerutil::GetIteratorForInstruction(
            ir_context->get_instr_block(inst), inst);
        assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpStore, it) &&
               "Can't insert OpStore right before the function call");

        // Insert either OpStore or OpCopyMemory depending on the type of the
        // parameter. Neither of those instructions support OpTypeRuntimeArray
        // but it's okay in this case since we check each parameter's type in
        // the IsApplicable method.
        it.InsertBefore(MakeUnique<opt::Instruction>(
            ir_context,
            param_type_inst->opcode() == SpvOpTypePointer ? SpvOpCopyMemory
                                                          : SpvOpStore,
            0, 0,
            opt::Instruction::OperandList{
                {SPV_OPERAND_TYPE_ID, {global_variable_result_id}},
                {SPV_OPERAND_TYPE_ID,
                 {inst->GetSingleWordInOperand(parameter_index + 1)}}}));

        // +1 since the first operand of OpFunctionCall is an id of the
        // function.
        inst->RemoveInOperand(parameter_index + 1);
      });

  // Remove the parameter from the function.
  function->RemoveParameter(message_.parameter_id());

  // Update function's type.
  auto* old_function_type = fuzzerutil::GetFunctionType(ir_context, function);
  assert(old_function_type && "Function has invalid type");

  // Preemptively add function's return type id.
  std::vector<uint32_t> type_ids = {
      old_function_type->GetSingleWordInOperand(0)};

  // +1 and -1 since the first operand is the return type id.
  for (uint32_t i = 1; i < old_function_type->NumInOperands(); ++i) {
    if (i - 1 != parameter_index) {
      type_ids.push_back(old_function_type->GetSingleWordInOperand(i));
    }
  }

  if (ir_context->get_def_use_mgr()->NumUsers(old_function_type) == 1) {
    // Change the old type in place. +1 since the first operand is the result
    // type id of the function.
    old_function_type->RemoveInOperand(parameter_index + 1);
  } else {
    // Find an existing or create a new function type.
    function->DefInst().SetInOperand(
        1, {fuzzerutil::FindOrCreateFunctionType(
               ir_context, message_.function_type_fresh_id(), type_ids)});
  }

  // Make sure our changes are analyzed
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationReplaceParameterWithGlobal::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_replace_parameter_with_global() = message_;
  return result;
}

bool TransformationReplaceParameterWithGlobal::CanReplaceFunctionParameterType(
    const opt::analysis::Type& type) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3403):
  //  Think about other type instructions we can add here.
  switch (type.kind()) {
    case opt::analysis::Type::kBool:
    case opt::analysis::Type::kInteger:
    case opt::analysis::Type::kFloat:
      return true;
    case opt::analysis::Type::kArray:
      return CanReplaceFunctionParameterType(*type.AsArray()->element_type());
    case opt::analysis::Type::kMatrix:
      return CanReplaceFunctionParameterType(*type.AsMatrix()->element_type());
    case opt::analysis::Type::kVector:
      return CanReplaceFunctionParameterType(*type.AsVector()->element_type());
    case opt::analysis::Type::kStruct:
      return std::all_of(
          type.AsStruct()->element_types().begin(),
          type.AsStruct()->element_types().end(),
          [](const opt::analysis::Type* element_type) {
            return CanReplaceFunctionParameterType(*element_type);
          });
    case opt::analysis::Type::kPointer: {
      switch (type.AsPointer()->storage_class()) {
        case SpvStorageClassPrivate:
        case SpvStorageClassFunction:
        case SpvStorageClassWorkgroup:
          return CanReplaceFunctionParameterType(
              *type.AsPointer()->pointee_type());
        default:
          return false;
      }
    }
    default:
      return false;
  }
}

SpvStorageClass
TransformationReplaceParameterWithGlobal::GetStorageClassForGlobalVariable(
    opt::IRContext* ir_context, uint32_t param_type_id) {
  const auto* param_type = ir_context->get_type_mgr()->GetType(param_type_id);
  assert(param_type && "Parameter type is invalid");

  assert(CanReplaceFunctionParameterType(*param_type));

  if (!param_type->AsPointer()) {
    return SpvStorageClassPrivate;
  }

  return param_type->AsPointer()->storage_class() == SpvStorageClassFunction
             ? SpvStorageClassPrivate
             : param_type->AsPointer()->storage_class();
}

}  // namespace fuzz
}  // namespace spvtools
