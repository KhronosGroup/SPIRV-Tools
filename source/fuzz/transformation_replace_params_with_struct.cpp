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

#include "source/fuzz/transformation_replace_params_with_struct.h"

#include <unordered_set>
#include <vector>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceParamsWithStruct::TransformationReplaceParamsWithStruct(
    const protobufs::TransformationReplaceParamsWithStruct& message)
    : message_(message) {}

TransformationReplaceParamsWithStruct::TransformationReplaceParamsWithStruct(
    uint32_t function_id, const std::vector<uint32_t>& parameter_index,
    uint32_t new_type_id, uint32_t fresh_parameter_id,
    uint32_t fresh_composite_id) {
  message_.set_function_id(function_id);
  message_.set_new_type_id(new_type_id);
  message_.set_fresh_parameter_id(fresh_parameter_id);
  message_.set_fresh_composite_id(fresh_composite_id);

  for (auto index : parameter_index) {
    message_.add_parameter_index(index);
  }
}

bool TransformationReplaceParamsWithStruct::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  const auto* function =
      fuzzerutil::FindFunction(ir_context, message_.function_id());
  if (!function ||
      fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
  }

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3454):
  //  uncomment when the PR is merged.
  // auto params = fuzzerutil::GetParameters(ir_context, function->result_id());
  std::vector<opt::Instruction*> params;

  // Can't replace parameters if there are no any.
  assert(!params.empty() && "A function must have parameters to be replaced");

  std::vector<uint32_t> param_index(message_.parameter_index().begin(),
                                    message_.parameter_index().end());

  assert(!fuzzerutil::HasDuplicates(param_index) &&
         "Indices of replaced parameters may not have duplicates");

  if (param_index.empty() || param_index.size() > params.size()) {
    return false;
  }

  // Check that all parameters' indices are valid.
  if (!std::all_of(
          param_index.begin(), param_index.end(),
          [&params](uint32_t index) { return index < params.size(); })) {
    return false;
  }

  const auto* old_type_inst = fuzzerutil::GetFunctionType(ir_context, function);
  assert(old_type_inst && old_type_inst->opcode() == SpvOpTypeFunction &&
         "Function's type is invalid");
  const auto* new_type_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.new_type_id());

  // Check that new function's type is valid.
  if (!new_type_inst || new_type_inst->opcode() != SpvOpTypeFunction) {
    return false;
  }

  // Check that new function's type has correct number of parameters. +1 since
  // the last parameter of the new type should be the struct type used to
  // replace parameters.
  if (new_type_inst->NumInOperands() !=
      old_type_inst->NumInOperands() - param_index.size() + 1) {
    return false;
  }

  // Check that new parameter's types are correct and their order is preserved.
  // We are adding a return type id as the first element of the vector.
  std::vector<uint32_t> new_type_ids = {
      old_type_inst->GetSingleWordInOperand(0)};
  for (uint32_t i = 0, n = static_cast<uint32_t>(params.size()); i < n; ++i) {
    if (std::find(param_index.begin(), param_index.end(), i) ==
        param_index.end()) {
      new_type_ids.push_back(params[i]->type_id());
    }
  }

  // Check that the return type and the remaining parameters are correct and
  // their order is preserved. We don't check the last operand that contains the
  // result id of the struct type which will be used to replaced parameters.
  for (uint32_t i = 0, n = new_type_inst->NumInOperands() - 1; i < n; ++i) {
    if (new_type_inst->GetSingleWordInOperand(i) != new_type_ids[i]) {
      return false;
    }
  }

  // Check that the last operand (i.e. the struct type) is valid.
  const auto* struct_type_inst = ir_context->get_def_use_mgr()->GetDef(
      new_type_inst->GetSingleWordInOperand(new_type_inst->NumInOperands() -
                                            1));
  if (!struct_type_inst || struct_type_inst->opcode() != SpvOpTypeStruct) {
    return false;
  }

  // Check that all replaced parameters are present in the type.
  if (struct_type_inst->NumInOperands() != param_index.size()) {
    return false;
  }

  // Check that indices of removed parameters correspond to the components'
  // types.
  for (size_t i = 0, n = param_index.size(); i < n; ++i) {
    if (params[param_index[i]]->type_id() !=
        struct_type_inst->GetSingleWordInOperand(static_cast<uint32_t>(i))) {
      return false;
    }
  }

  // Check that the result ids for the new parameter and its value are fresh.
  return fuzzerutil::IsFreshId(ir_context, message_.fresh_parameter_id()) &&
         fuzzerutil::IsFreshId(ir_context, message_.fresh_composite_id());
}

void TransformationReplaceParamsWithStruct::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* function = fuzzerutil::FindFunction(ir_context, message_.function_id());
  assert(function);

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3454):
  //  uncomment when the PR is merged.
  // auto params = fuzzerutil::GetParameters(ir_context,
  // message_.function_id());
  std::vector<opt::Instruction*> params;

  // Add OpCompositeExtract instructions to extract values of replaced
  // parameters from the added struct parameter.
  for (int i = 0, n = message_.parameter_index_size(); i < n; ++i) {
    auto* param_inst = params[message_.parameter_index(i)];

    // Skip all OpVariable and OpPhi instructions.
    auto iter = function->begin()->begin();
    while (iter != function->begin()->end() &&
           !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCompositeExtract,
                                                         iter)) {
      ++iter;
    }

    assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCompositeExtract,
                                                        iter) &&
           "Can't extract parameter's value from the structure");

    // Insert OpCompositeExtract with the result id of the replaced parameter.
    iter.InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpCompositeExtract, param_inst->type_id(),
        param_inst->result_id(),
        opt::Instruction::OperandList{
            {SPV_OPERAND_TYPE_ID, {message_.fresh_parameter_id()}},
            {SPV_OPERAND_TYPE_LITERAL_INTEGER, {static_cast<uint32_t>(i)}}}));

    function->RemoveParameter(param_inst->result_id());
  }

  const auto* new_type_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.new_type_id());
  assert(new_type_inst && "New function's type must exist");

  auto composite_type_id =
      new_type_inst->GetSingleWordInOperand(new_type_inst->NumInOperands() - 1);

  // Add new parameter to the function.
  function->AddParameter(MakeUnique<opt::Instruction>(
      ir_context, SpvOpFunctionParameter, composite_type_id,
      message_.fresh_parameter_id(), opt::Instruction::OperandList{}));

  // Update function's type.
  function->DefInst().SetInOperand(1, {message_.new_type_id()});

  // Update all function calls.
  ir_context->get_def_use_mgr()->ForEachUser(
      function->result_id(),
      [this, ir_context, composite_type_id](opt::Instruction* inst) {
        if (inst->opcode() != SpvOpFunctionCall) {
          return;
        }

        // Create a list of operands for the OpCompositeConstruct instruction.
        opt::Instruction::OperandList composite_components;
        for (auto index : message_.parameter_index()) {
          // +1 since the first in operand to OpFunctionCall is the result id of
          // the function.
          composite_components.emplace_back(
              std::move(inst->GetInOperand(index + 1)));
          inst->RemoveInOperand(index + 1);
        }

        // Insert OpCompositeConstruct before the function call.
        inst->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, SpvOpCompositeConstruct, composite_type_id,
            message_.fresh_composite_id(), std::move(composite_components)));

        // Add a new operand to the OpFunctionCall instruction.
        inst->AddOperand(
            {SPV_OPERAND_TYPE_ID, {message_.fresh_composite_id()}});
      });

  fuzzerutil::UpdateModuleIdBound(
      ir_context,
      std::max(message_.fresh_composite_id(), message_.fresh_parameter_id()));

  // Make sure our changes are analyzed
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationReplaceParamsWithStruct::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_replace_params_with_struct() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
