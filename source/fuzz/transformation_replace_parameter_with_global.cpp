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

TransformationReplaceParameterWithGlobal::
    TransformationReplaceParameterWithGlobal(
        const protobufs::TransformationReplaceParameterWithGlobal& message)
    : message_(message) {}

TransformationReplaceParameterWithGlobal::
    TransformationReplaceParameterWithGlobal(
        uint32_t function_type_fresh_id, uint32_t parameter_id,
        uint32_t global_variable_fresh_id) {
  message_.set_function_type_fresh_id(function_type_fresh_id);
  message_.set_parameter_id(parameter_id);
  message_.set_global_variable_fresh_id(global_variable_fresh_id);
}

bool TransformationReplaceParameterWithGlobal::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Check that |parameter_id| is valid.
  const auto* param_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.parameter_id());
  if (!param_inst || param_inst->opcode() != SpvOpFunctionParameter) {
    return false;
  }

  // Check that function exists and is not an entry point.
  const auto* function = fuzzerutil::GetFunctionFromParameterId(
      ir_context, message_.parameter_id());
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
  if (!IsParameterTypeSupported(*param_type)) {
    return false;
  }

  // Check that initializer for the global variable exists in the module.
  if (fuzzerutil::MaybeGetZeroConstant(ir_context, transformation_context,
                                       param_inst->type_id()) == 0) {
    return false;
  }

  // Check that pointer type for the global variable exists in the module.
  if (!fuzzerutil::MaybeGetPointerType(ir_context, param_inst->type_id(),
                                       SpvStorageClassPrivate)) {
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

  // Create global variable to store parameter's value.
  fuzzerutil::AddGlobalVariable(
      ir_context, message_.global_variable_fresh_id(),
      fuzzerutil::MaybeGetPointerType(ir_context, param_inst->type_id(),
                                      SpvStorageClassPrivate),
      SpvStorageClassPrivate,
      fuzzerutil::MaybeGetZeroConstant(ir_context, *transformation_context,
                                       param_inst->type_id()));

  // Mark the global variable's pointee as irrelevant if replaced parameter is
  // irrelevant.
  if (transformation_context->GetFactManager()->IdIsIrrelevant(
          message_.parameter_id())) {
    transformation_context->GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
        message_.global_variable_fresh_id());
  }

  auto* function = fuzzerutil::GetFunctionFromParameterId(
      ir_context, message_.parameter_id());
  assert(function && "Function must exist");

  // Insert an OpLoad instruction right after OpVariable instructions.
  auto it = function->begin()->begin();
  while (it != function->begin()->end() &&
         !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, it)) {
    ++it;
  }

  assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, it) &&
         "Can't insert OpLoad or OpCopyMemory into the first basic block of "
         "the function");

  it.InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, SpvOpLoad, param_inst->type_id(), param_inst->result_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.global_variable_fresh_id()}}}));

  // Calculate the index of the replaced parameter (we need to know this to
  // remove operands from the OpFunctionCall).
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

  // Update all relevant OpFunctionCall instructions.
  for (auto* inst : fuzzerutil::GetCallers(ir_context, function->result_id())) {
    assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpStore, inst) &&
           "Can't insert OpStore right before the function call");

    // Insert an OpStore before the OpFunctionCall. +1 since the first
    // operand of OpFunctionCall is an id of the function.
    inst->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpStore, 0, 0,
        opt::Instruction::OperandList{
            {SPV_OPERAND_TYPE_ID, {message_.global_variable_fresh_id()}},
            {SPV_OPERAND_TYPE_ID,
             {inst->GetSingleWordInOperand(parameter_index + 1)}}}));

    // +1 since the first operand of OpFunctionCall is an id of the
    // function.
    inst->RemoveInOperand(parameter_index + 1);
  }

  // Remove the parameter from the function.
  function->RemoveParameter(message_.parameter_id());

  // Update function's type.
  {
    // We use a separate scope here since |old_function_type| might become a
    // dangling pointer after the call to the fuzzerutil::UpdateFunctionType.

    auto* old_function_type = fuzzerutil::GetFunctionType(ir_context, function);
    assert(old_function_type && "Function has invalid type");

    // +1 and -1 since the first operand is the return type id.
    std::vector<uint32_t> parameter_type_ids;
    for (uint32_t i = 1; i < old_function_type->NumInOperands(); ++i) {
      if (i - 1 != parameter_index) {
        parameter_type_ids.push_back(
            old_function_type->GetSingleWordInOperand(i));
      }
    }

    fuzzerutil::UpdateFunctionType(
        ir_context, function->result_id(), message_.function_type_fresh_id(),
        old_function_type->GetSingleWordInOperand(0), parameter_type_ids);
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

bool TransformationReplaceParameterWithGlobal::IsParameterTypeSupported(
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
    default:
      return false;
  }
}

}  // namespace fuzz
}  // namespace spvtools
