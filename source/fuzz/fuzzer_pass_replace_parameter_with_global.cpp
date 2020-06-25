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

#include "source/fuzz/fuzzer_pass_replace_parameter_with_global.h"

#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_replace_parameter_with_global.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceParameterWithGlobal::FuzzerPassReplaceParameterWithGlobal(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceParameterWithGlobal::~FuzzerPassReplaceParameterWithGlobal() =
    default;

void FuzzerPassReplaceParameterWithGlobal::Apply() {
  for (const auto& function : *GetIRContext()->module()) {
    auto params =
        fuzzerutil::GetParameters(GetIRContext(), function.result_id());

    if (params.empty() || fuzzerutil::FunctionIsEntryPoint(
                              GetIRContext(), function.result_id())) {
      continue;
    }

    // Make sure at least one parameter can be replaced.
    if (std::none_of(params.begin(), params.end(),
                     [this](const opt::Instruction* param) {
                       return TransformationReplaceParameterWithGlobal::
                           CanReplaceFunctionParameterType(GetIRContext(),
                                                           param->type_id());
                     })) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfReplacingParametersWithGlobals())) {
      continue;
    }

    // Select id of a parameter to replace.
    const opt::Instruction* replaced_param;
    do {
      replaced_param = params[GetFuzzerContext()->RandomIndex(params)];
    } while (!TransformationReplaceParameterWithGlobal::
                 CanReplaceFunctionParameterType(GetIRContext(),
                                                 replaced_param->type_id()));

    auto* param_type_inst =
        GetIRContext()->get_def_use_mgr()->GetDef(replaced_param->type_id());
    assert(param_type_inst && "Parameter's type does not exist");

    auto global_variable_storage_class =
        TransformationReplaceParameterWithGlobal::
            GetStorageClassForGlobalVariable(GetIRContext(),
                                             replaced_param->type_id());

    // Make sure type ids for global variables exist in the module.
    FindOrCreatePointerType(
        param_type_inst->opcode() == SpvOpTypePointer
            ? fuzzerutil::GetPointeeTypeIdFromPointerType(param_type_inst)
            : param_type_inst->result_id(),
        global_variable_storage_class);

    // fuzzerutil::AddGlobalVariable requires initializer to be 0 if variable's
    // storage class is Workgroup.
    auto initializer_id =
        global_variable_storage_class == SpvStorageClassWorkgroup
            ? 0
            : FindOrCreateZeroConstant(
                  param_type_inst->opcode() == SpvOpTypePointer
                      ? fuzzerutil::GetPointeeTypeIdFromPointerType(
                            param_type_inst)
                      : param_type_inst->result_id());

    // Compute type ids for the remaining arguments.
    std::vector<uint32_t> argument_ids;
    for (const auto* param : params) {
      if (param->result_id() != replaced_param->result_id()) {
        argument_ids.push_back(param->type_id());
      }
    }

    auto new_type_id =
        FindOrCreateFunctionType(function.type_id(), argument_ids);

    auto fresh_id = param_type_inst->opcode() != SpvOpTypePointer ||
                            fuzzerutil::GetStorageClassFromPointerType(
                                param_type_inst) == SpvStorageClassFunction
                        ? GetFuzzerContext()->GetFreshId()
                        : 0;

    ApplyTransformation(TransformationReplaceParameterWithGlobal(
        new_type_id, replaced_param->result_id(), fresh_id, initializer_id));
  }
}

}  // namespace fuzz
}  // namespace spvtools
