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

#include <vector>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceParamsWithStruct::TransformationReplaceParamsWithStruct(
    const protobufs::TransformationReplaceParamsWithStruct& message)
    : message_(message) {}

TransformationReplaceParamsWithStruct::TransformationReplaceParamsWithStruct(
    const std::vector<uint32_t>& parameter_id, uint32_t fresh_function_type_id,
    uint32_t fresh_parameter_id,
    const std::unordered_map<uint32_t, uint32_t>&
        caller_id_to_fresh_composite_id) {
  message_.set_fresh_function_type_id(fresh_function_type_id);
  message_.set_fresh_parameter_id(fresh_parameter_id);

  for (auto id : parameter_id) {
    message_.add_parameter_id(id);
  }

  message_.mutable_caller_id_to_fresh_composite_id()->insert(
      caller_id_to_fresh_composite_id.begin(),
      caller_id_to_fresh_composite_id.end());
}

bool TransformationReplaceParamsWithStruct::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  std::vector<uint32_t> parameter_id(message_.parameter_id().begin(),
                                     message_.parameter_id().end());

  // Check that |parameter_id| is neither empty nor it has duplicates.
  if (parameter_id.empty() || fuzzerutil::HasDuplicates(parameter_id)) {
    return false;
  }

  // All ids must correspond to valid parameters of the same function.
  // The function can't be an entry-point function.

  // fuzzerutil::GetFunctionFromParameterId requires a valid id.
  if (!ir_context->get_def_use_mgr()->GetDef(parameter_id[0])) {
    return false;
  }

  const auto* function =
      fuzzerutil::GetFunctionFromParameterId(ir_context, parameter_id[0]);
  if (!function ||
      fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
  }

  // Compute all ids of the function's parameters.
  std::unordered_set<uint32_t> all_parameter_ids;
  for (const auto* param :
       fuzzerutil::GetParameters(ir_context, function->result_id())) {
    all_parameter_ids.insert(param->result_id());
  }

  // Check that all elements in |parameter_id| are valid.
  for (auto id : parameter_id) {
    // fuzzerutil::GetFunctionFromParameterId requires a valid id.
    if (!ir_context->get_def_use_mgr()->GetDef(id)) {
      return false;
    }

    // Check that |id| is a result id of one of the |function|'s parameters.
    if (!all_parameter_ids.count(id)) {
      return false;
    }

    // Check that the parameter with result id |id| has supported type.
    const auto* type = ir_context->get_type_mgr()->GetType(
        fuzzerutil::GetTypeId(ir_context, id));
    assert(type && "Parameter has invalid type");
    if (!IsParameterTypeSupported(*type)) {
      return false;
    }
  }

  // We already know that the function has at least |parameter_id.size()|
  // parameters.

  // Check that a relevant OpTypeStruct exists in the module.
  if (!MaybeGetRequiredStructType(ir_context)) {
    return false;
  }

  // Check that |callee_id_to_fresh_composite_id| is valid.
  for (const auto* inst :
       fuzzerutil::GetCallers(ir_context, function->result_id())) {
    // Check that the callee is present in the map. It's ok if the map contains
    // more ids that there are callees (those ids will not be used).
    if (!message_.caller_id_to_fresh_composite_id().contains(
            inst->result_id())) {
      return false;
    }
  }

  // Check that all fresh ids are unique and fresh.
  std::vector<uint32_t> fresh_ids = {message_.fresh_function_type_id(),
                                     message_.fresh_parameter_id()};

  for (const auto& entry : message_.caller_id_to_fresh_composite_id()) {
    fresh_ids.push_back(entry.second);
  }

  return !fuzzerutil::HasDuplicates(fresh_ids) &&
         std::all_of(fresh_ids.begin(), fresh_ids.end(),
                     [ir_context](uint32_t id) {
                       return fuzzerutil::IsFreshId(ir_context, id);
                     });
}

void TransformationReplaceParamsWithStruct::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* function = fuzzerutil::GetFunctionFromParameterId(
      ir_context, message_.parameter_id(0));
  assert(function &&
         "All parameters' ids should've been checked in the IsApplicable");

  // Get a type id of the OpTypeStruct used as a type id of the new parameter.
  auto struct_type_id = MaybeGetRequiredStructType(ir_context);
  assert(struct_type_id &&
         "IsApplicable should've guaranteed that this value isn't equal to 0");

  // Add new parameter to the function.
  function->AddParameter(MakeUnique<opt::Instruction>(
      ir_context, SpvOpFunctionParameter, struct_type_id,
      message_.fresh_parameter_id(), opt::Instruction::OperandList()));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_parameter_id());

  // Compute indices of replaced parameters. This will be used to adjust
  // OpFunctionCall instructions and create OpCompositeConstruct instructions at
  // every call site.
  std::vector<uint32_t> indices_of_replaced_params;
  {
    // We want to destroy |params| after the loop because it will contain
    // dangling pointers when we remove parameters from the function.
    auto params = fuzzerutil::GetParameters(ir_context, function->result_id());
    for (auto id : message_.parameter_id()) {
      auto it = std::find_if(params.begin(), params.end(),
                             [id](const opt::Instruction* param) {
                               return param->result_id() == id;
                             });
      assert(it != params.end() && "Parameter's id is invalid");
      indices_of_replaced_params.push_back(
          static_cast<uint32_t>(it - params.begin()));
    }
  }

  // Update all function calls.
  for (auto* inst : fuzzerutil::GetCallers(ir_context, function->result_id())) {
    // Create a list of operands for the OpCompositeConstruct instruction.
    opt::Instruction::OperandList composite_components;
    for (auto index : indices_of_replaced_params) {
      // +1 since the first in operand to OpFunctionCall is the result id of
      // the function.
      composite_components.emplace_back(
          std::move(inst->GetInOperand(index + 1)));
    }

    // Remove arguments from the function call. We do it in a separate loop
    // and in reverse order to make sure we have removed correct operands.
    for (auto it = indices_of_replaced_params.rbegin();
         it != indices_of_replaced_params.rend(); ++it) {
      // +1 since the first in operand to OpFunctionCall is the result id of
      // the function.
      inst->RemoveInOperand(*it + 1);
    }

    // Insert OpCompositeConstruct before the function call.
    auto fresh_composite_id =
        message_.caller_id_to_fresh_composite_id().at(inst->result_id());
    inst->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpCompositeConstruct, struct_type_id, fresh_composite_id,
        std::move(composite_components)));

    // Add a new operand to the OpFunctionCall instruction.
    inst->AddOperand({SPV_OPERAND_TYPE_ID, {fresh_composite_id}});
    fuzzerutil::UpdateModuleIdBound(ir_context, fresh_composite_id);
  }

  // Insert OpCompositeExtract instructions into the entry point block of the
  // function and remove replaced parameters.
  for (int i = 0; i < message_.parameter_id_size(); ++i) {
    const auto* param_inst =
        ir_context->get_def_use_mgr()->GetDef(message_.parameter_id(i));
    assert(param_inst && "Parameter id is invalid");

    // Skip all OpVariable instructions.
    auto iter = function->begin()->begin();
    while (iter != function->begin()->end() &&
           !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCompositeExtract,
                                                         iter)) {
      ++iter;
    }

    assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCompositeExtract,
                                                        iter) &&
           "Can't extract parameter's value from the structure");

    // Insert OpCompositeExtract instructions to unpack parameters' values from
    // the struct type.
    iter.InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpCompositeExtract, param_inst->type_id(),
        param_inst->result_id(),
        opt::Instruction::OperandList{
            {SPV_OPERAND_TYPE_ID, {message_.fresh_parameter_id()}},
            {SPV_OPERAND_TYPE_LITERAL_INTEGER, {static_cast<uint32_t>(i)}}}));

    function->RemoveParameter(param_inst->result_id());
  }

  // Update function's type.
  auto* old_function_type = fuzzerutil::GetFunctionType(ir_context, function);
  assert(old_function_type && "Function has invalid type");

  std::vector<uint32_t> type_ids = {
      // Result type of the function.
      old_function_type->GetSingleWordInOperand(0)};

  // +1 since the first in operand to OpTypeFunction is the result type id
  // of the function.
  for (uint32_t i = 1; i < old_function_type->NumInOperands(); ++i) {
    if (std::find(indices_of_replaced_params.begin(),
                  indices_of_replaced_params.end(),
                  i - 1) == indices_of_replaced_params.end()) {
      type_ids.push_back(old_function_type->GetSingleWordInOperand(i));
    }
  }

  type_ids.push_back(struct_type_id);

  fuzzerutil::MaybeReuseFunctionType(ir_context, function->result_id(),
                                     message_.fresh_function_type_id(),
                                     type_ids);

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

bool TransformationReplaceParamsWithStruct::IsParameterTypeSupported(
    const opt::analysis::Type& param_type) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3403):
  //  Consider adding support for more types of parameters.
  switch (param_type.kind()) {
    case opt::analysis::Type::kBool:
    case opt::analysis::Type::kInteger:
    case opt::analysis::Type::kFloat:
    case opt::analysis::Type::kArray:
    case opt::analysis::Type::kVector:
    case opt::analysis::Type::kMatrix:
      return true;
    case opt::analysis::Type::kStruct:
      return std::all_of(param_type.AsStruct()->element_types().begin(),
                         param_type.AsStruct()->element_types().end(),
                         [](const opt::analysis::Type* type) {
                           return IsParameterTypeSupported(*type);
                         });
    default:
      return false;
  }
}

uint32_t TransformationReplaceParamsWithStruct::MaybeGetRequiredStructType(
    opt::IRContext* ir_context) const {
  std::vector<uint32_t> component_type_ids;
  for (auto id : message_.parameter_id()) {
    component_type_ids.push_back(fuzzerutil::GetTypeId(ir_context, id));
  }

  return fuzzerutil::MaybeGetStructType(ir_context, component_type_ids);
}

}  // namespace fuzz
}  // namespace spvtools
