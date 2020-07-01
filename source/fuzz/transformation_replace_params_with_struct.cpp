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
    const std::vector<uint32_t>& parameter_id, uint32_t fresh_function_type_id,
    uint32_t fresh_parameter_id,
    const std::vector<uint32_t>& fresh_composite_id,
    uint32_t fresh_struct_type_id) {
  message_.set_fresh_function_type_id(fresh_function_type_id);
  message_.set_fresh_parameter_id(fresh_parameter_id);
  message_.set_fresh_struct_type_id(fresh_struct_type_id);

  for (auto id : parameter_id) {
    message_.add_parameter_id(id);
  }

  for (auto id : fresh_composite_id) {
    message_.add_fresh_composite_id(id);
  }
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
  //
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3434):
  //  GetFunctionFromParameterId is available when the PR is merged.
  const auto* function =
      fuzzerutil::GetFunctionFromParameterId(ir_context, parameter_id[0]);
  if (!function ||
      fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
  }

  for (size_t i = 1; i < parameter_id.size(); ++i) {
    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3434):
    //  GetFunctionFromParameterId is available when the PR is merged.
    if (fuzzerutil::GetFunctionFromParameterId(ir_context, parameter_id[i]) !=
        function) {
      return false;
    }
  }

  // We already know that the function has at least |parameter_id.size()|
  // parameters.

  // Check that all parameters have supported types.
  if (!std::all_of(parameter_id.begin(), parameter_id.end(),
                   [ir_context](uint32_t id) {
                     const auto* type = ir_context->get_type_mgr()->GetType(
                         fuzzerutil::GetTypeId(ir_context, id));
                     return type && IsParameterTypeSupported(*type);
                   })) {
    return false;
  }

  // Check that |fresh_composite_id| has valid size.
  if (static_cast<uint32_t>(message_.fresh_composite_id_size()) !=
      GetNumberOfCallees(ir_context, function->result_id())) {
    return false;
  }

  // Check that all fresh ids are unique and fresh.
  std::vector<uint32_t> fresh_ids(message_.fresh_composite_id().begin(),
                                  message_.fresh_composite_id().end());
  fresh_ids.insert(fresh_ids.end(), {message_.fresh_function_type_id(),
                                     message_.fresh_parameter_id(),
                                     message_.fresh_struct_type_id()});

  // Check that the result ids for the new parameter and its value are fresh.
  return !fuzzerutil::HasDuplicates(fresh_ids) &&
         std::all_of(fresh_ids.begin(), fresh_ids.end(),
                     [ir_context](uint32_t id) {
                       return fuzzerutil::IsFreshId(ir_context, id);
                     });
}

void TransformationReplaceParamsWithStruct::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3434):
  //  GetFunctionFromParameterId is available when the PR is merged.
  auto* function = fuzzerutil::GetFunctionFromParameterId(
      ir_context, message_.parameter_id(0));
  assert(function);

  // Create a new struct type.
  std::vector<uint32_t> struct_components_ids;
  for (auto id : message_.parameter_id()) {
    struct_components_ids.push_back(fuzzerutil::GetTypeId(ir_context, id));
  }

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3479):
  //  fuzzerutil::FindOrCreateStructType is available when the PR is merged.
  auto struct_type_id = fuzzerutil::FindOrCreateStructType(
      ir_context, message_.fresh_struct_type_id(), struct_components_ids);

  // Add new parameter to the function.
  function->AddParameter(MakeUnique<opt::Instruction>(
      ir_context, SpvOpFunctionParameter, struct_type_id,
      message_.fresh_parameter_id(), opt::Instruction::OperandList()));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_parameter_id());

  // Compute indices of parameters.
  std::vector<uint32_t> param_indices;
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
      param_indices.push_back(static_cast<uint32_t>(it - params.begin()));
    }
  }

  // Update all function calls.
  std::vector<uint32_t> fresh_composite_id(
      message_.fresh_composite_id().begin(),
      message_.fresh_composite_id().end());

  ir_context->get_def_use_mgr()->ForEachUser(
      function->result_id(), [&fresh_composite_id, ir_context, struct_type_id,
                              &param_indices](opt::Instruction* inst) {
        if (inst->opcode() != SpvOpFunctionCall) {
          return;
        }

        // Create a list of operands for the OpCompositeConstruct instruction.
        opt::Instruction::OperandList composite_components;
        for (auto index : param_indices) {
          // +1 since the first in operand to OpFunctionCall is the result id of
          // the function.
          composite_components.emplace_back(
              std::move(inst->GetInOperand(index + 1)));
        }

        // Remove arguments from the function call. We do it in a separate loop
        // because otherwise we would have moved invalid operands into
        // |composite_components|.
        for (auto index : param_indices) {
          // +1 since the first in operand to OpFunctionCall is the result id of
          // the function.
          inst->RemoveInOperand(index + 1);
        }

        // Insert OpCompositeConstruct before the function call.
        inst->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, SpvOpCompositeConstruct, struct_type_id,
            fresh_composite_id.back(), std::move(composite_components)));

        // Add a new operand to the OpFunctionCall instruction.
        inst->AddOperand({SPV_OPERAND_TYPE_ID, {fresh_composite_id.back()}});

        fuzzerutil::UpdateModuleIdBound(ir_context, fresh_composite_id.back());
        fresh_composite_id.pop_back();
      });

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

  if (ir_context->get_def_use_mgr()->NumUsers(old_function_type) == 1) {
    // Update |old_function_type| in place.
    old_function_type->AddOperand({SPV_OPERAND_TYPE_ID, {struct_type_id}});

    for (auto index : param_indices) {
      // +1 since the first in operand to OpTypeFunction is the result type id
      // of the function.
      old_function_type->RemoveInOperand(index + 1);
    }

    // Make sure domination rules are satisfied.
    old_function_type->RemoveFromList();
    ir_context->AddType(std::unique_ptr<opt::Instruction>(old_function_type));
  } else {
    std::vector<uint32_t> type_ids;

    // +1 since the first in operand to OpTypeFunction is the result type id
    // of the function.
    for (uint32_t i = 1; i < old_function_type->NumInOperands(); ++i) {
      if (std::find(param_indices.begin(), param_indices.end(), i - 1) !=
          param_indices.end()) {
        type_ids.push_back(old_function_type->GetSingleWordInOperand(i));
      }
    }

    type_ids.push_back(struct_type_id);

    // Create a new function type or use an existing one.
    function->DefInst().SetInOperand(
        1, {fuzzerutil::FindOrCreateFunctionType(
               ir_context, message_.fresh_function_type_id(), type_ids)});
  }

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

uint32_t TransformationReplaceParamsWithStruct::GetNumberOfCallees(
    opt::IRContext* ir_context, uint32_t function_id) {
  assert(fuzzerutil::FindFunction(ir_context, function_id) &&
         "|function_id| is invalid");

  uint32_t result = 0;
  ir_context->get_def_use_mgr()->ForEachUser(
      function_id, [&result, function_id](const opt::Instruction* user) {
        if (user->opcode() != SpvOpFunctionCall ||
            user->GetSingleWordInOperand(0) != function_id) {
          return;
        }

        ++result;
      });

  return result;
}

bool TransformationReplaceParamsWithStruct::IsParameterTypeSupported(
    const opt::analysis::Type& param_type) {
  switch (param_type.kind()) {
    case opt::analysis::Type::kBool:
    case opt::analysis::Type::kInteger:
    case opt::analysis::Type::kFloat:
      return true;
    case opt::analysis::Type::kVector:
      return IsParameterTypeSupported(*param_type.AsVector()->element_type());
    case opt::analysis::Type::kMatrix:
      return IsParameterTypeSupported(*param_type.AsMatrix()->element_type());
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

}  // namespace fuzz
}  // namespace spvtools
