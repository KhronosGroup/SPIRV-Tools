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
    TransformationReplaceParameterWithGlobal(uint32_t new_type_id,
                                             uint32_t parameter_id,
                                             uint32_t fresh_id,
                                             uint32_t initializer_id) {
  message_.set_new_type_id(new_type_id);
  message_.set_parameter_id(parameter_id);
  message_.set_fresh_id(fresh_id);
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

  auto params = fuzzerutil::GetParameters(ir_context, function->result_id());
  assert(!params.empty() &&
         "The function doesn't have any parameters to replace");

  // Check that new function type is valid.
  const auto* old_type_inst = fuzzerutil::GetFunctionType(ir_context, function);
  assert(old_type_inst && old_type_inst->opcode() == SpvOpTypeFunction &&
         "Function type is invalid");

  const auto* new_type_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.new_type_id());
  if (!new_type_inst || new_type_inst->opcode() != SpvOpTypeFunction) {
    return false;
  }

  // Check that new function type has the same number of operands.
  if (old_type_inst->NumInOperands() != new_type_inst->NumInOperands() + 1) {
    return false;
  }

  // Check that the return type remains the same.
  if (old_type_inst->GetSingleWordInOperand(0) !=
      new_type_inst->GetSingleWordInOperand(0)) {
    return false;
  }

  // Check that new function type has valid parameters' types.
  //
  // We are iterating from 1 since we are no taking return type into account.
  for (uint32_t i = 1, j = 1, n = old_type_inst->NumInOperands(); i < n;) {
    if (params[i - 1]->result_id() == message_.parameter_id()) {
      // Skip replaced parameter in old function's type.
      i++;
      continue;
    }

    if (old_type_inst->GetSingleWordInOperand(i++) !=
        new_type_inst->GetSingleWordInOperand(j++)) {
      return false;
    }
  }

  // Check that replaced parameter has valid type.
  if (!CanReplaceFunctionParameterType(ir_context, param_inst->type_id())) {
    return false;
  }

  auto* param_type_inst =
      ir_context->get_def_use_mgr()->GetDef(param_inst->type_id());
  assert(param_type_inst && "Parameter type must exist");

  auto pointee_type_id =
      param_type_inst->opcode() == SpvOpTypePointer
          ? fuzzerutil::GetPointeeTypeIdFromPointerType(param_type_inst)
          : param_type_inst->result_id();

  // Initializer id can be 0 iff parameter is a pointer with Workgroup storage
  // class.
  if (message_.initializer_id() == 0) {
    return param_type_inst->opcode() == SpvOpTypePointer &&
           GetStorageClassForGlobalVariable(
               ir_context, param_inst->type_id()) == SpvStorageClassWorkgroup;
  }

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

  if (param_type_inst->opcode() != SpvOpTypePointer ||
      fuzzerutil::GetStorageClassFromPointerType(param_type_inst) ==
          SpvStorageClassFunction) {
    return fuzzerutil::IsFreshId(ir_context, message_.fresh_id());
  }

  return message_.fresh_id() == 0;
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

  // If parameter is not a pointer, we use a fresh id for a global variable and
  // insert an OpLoad instruction to load parameter's value. If it's a pointer
  // with Function storage class, we use a fresh id for a global variable and
  // create a local variable to store parameter's value. Otherwise, we can reuse
  // parameter's id for a global variable.
  //
  // |message_.fresh_id| is zero if we are reusing parameter's id for a global
  // variable.
  fuzzerutil::AddGlobalVariable(
      ir_context, message_.fresh_id() ?: message_.parameter_id(),
      fuzzerutil::MaybeGetPointerType(ir_context, pointee_type_id,
                                      global_variable_storage_class),
      global_variable_storage_class, message_.initializer_id());

  auto* function =
      GetFunctionFromParameterId(ir_context, message_.parameter_id());
  assert(function && "Function must exist");

  if (param_type_inst->opcode() != SpvOpTypePointer ||
      fuzzerutil::GetStorageClassFromPointerType(param_type_inst) ==
          SpvStorageClassFunction) {
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
              {SPV_OPERAND_TYPE_ID, {message_.fresh_id()}}}));
    } else {
      it.InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpCopyMemory, 0, 0,
          opt::Instruction::OperandList{
              {SPV_OPERAND_TYPE_ID, {param_inst->result_id()}},
              {SPV_OPERAND_TYPE_ID, {message_.fresh_id()}}}));
    }

    // If parameter is not a pointer, the condition will fail. Otherwise, we
    // mark global variable as irrelevant if parameter's pointee is irrelevant.
    if (transformation_context->GetFactManager()->PointeeValueIsIrrelevant(
            message_.parameter_id())) {
      transformation_context->GetFactManager()
          ->AddFactValueOfPointeeIsIrrelevant(message_.fresh_id());
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
      function->result_id(), [ir_context, this, param_type_inst,
                              parameter_index](opt::Instruction* inst) {
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
                {SPV_OPERAND_TYPE_ID,
                 {message_.fresh_id() ?: message_.parameter_id()}},
                {SPV_OPERAND_TYPE_ID,
                 {inst->GetSingleWordInOperand(parameter_index + 1)}}}));

        // +1 since the first operand of OpFunctionCall is an id of the
        // function.
        inst->RemoveInOperand(parameter_index + 1);
      });

  // Remove the parameter from the function.
  function->RemoveParameter(message_.parameter_id());

  // Update function's type id.
  function->DefInst().SetInOperand(1, {message_.new_type_id()});

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
    opt::IRContext* ir_context, uint32_t param_type_id) {
  auto* param_type_inst = ir_context->get_def_use_mgr()->GetDef(param_type_id);
  assert(param_type_inst && "Parameter type is invalid");

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3403):
  //  Think about other type instructions we can add here.
  switch (param_type_inst->opcode()) {
    case SpvOpTypeBool:
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
    case SpvOpTypeArray:
    case SpvOpTypeMatrix:
    case SpvOpTypeVector:
    case SpvOpTypeStruct:
      return true;
    case SpvOpTypePointer: {
      switch (fuzzerutil::GetStorageClassFromPointerType(param_type_inst)) {
        case SpvStorageClassPrivate:
        case SpvStorageClassFunction:
        case SpvStorageClassWorkgroup:
          return true;
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
  assert(CanReplaceFunctionParameterType(ir_context, param_type_id));

  auto* param_type_inst = ir_context->get_def_use_mgr()->GetDef(param_type_id);
  assert(param_type_inst && "Parameter type is invalid");

  if (param_type_inst->opcode() != SpvOpTypePointer) {
    return SpvStorageClassPrivate;
  }

  auto storage_class =
      fuzzerutil::GetStorageClassFromPointerType(param_type_inst);
  return storage_class == SpvStorageClassFunction ? SpvStorageClassPrivate
                                                  : storage_class;
}

}  // namespace fuzz
}  // namespace spvtools
