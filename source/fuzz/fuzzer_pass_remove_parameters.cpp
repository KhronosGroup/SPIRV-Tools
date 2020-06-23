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

#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_pass_remove_parameters.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_remove_parameters.h"

namespace spvtools {
namespace fuzz {
namespace {

bool CanRemoveFunctionParameter(const opt::Instruction& type_inst) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3403):
  //  Add OpTypePointer below when the issue is resolved.
  switch (type_inst.opcode()) {
    case SpvOpTypeBool:
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
    case SpvOpTypeArray:
    case SpvOpTypeMatrix:
    case SpvOpTypeVector:
    case SpvOpTypeStruct:
      return true;
    default:
      return false;
  }
}

}  // namespace

FuzzerPassRemoveParameters::FuzzerPassRemoveParameters(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassRemoveParameters::~FuzzerPassRemoveParameters() = default;

void FuzzerPassRemoveParameters::Apply() {
  for (const auto& function : *GetIRContext()->module()) {
    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3454):
    //  uncomment when the PR is merged
    // auto params = fuzzerutil::GetParameters(function);
    std::vector<opt::Instruction*> params;

    if (params.empty() || fuzzerutil::FunctionIsEntryPoint(
                              GetIRContext(), function.result_id())) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfRemovingParameters())) {
      continue;
    }

    auto num_to_remove = GetFuzzerContext()->GetRandomNumberOfRemovedParameters(
        static_cast<uint32_t>(params.size()));

    // Select |num_to_remove| random parameters' indices to remove from the
    // |function|.
    std::vector<uint32_t> parameter_index(params.size());
    std::iota(parameter_index.begin(), parameter_index.end(), 0);

    // Remove parameters that can't be used in variable initialization.
    parameter_index.erase(
        std::remove_if(parameter_index.begin(), parameter_index.end(),
                       [&params, this](uint32_t index) {
                         const auto* type_inst =
                             GetIRContext()->get_def_use_mgr()->GetDef(
                                 params[index]->type_id());
                         assert(type_inst && "Parameter's type is invalid");

                         return !CanRemoveFunctionParameter(*type_inst);
                       }),
        parameter_index.end());

    if (parameter_index.empty()) {
      continue;
    }

    // Select parameters to remove at random.
    GetFuzzerContext()->Shuffle(&parameter_index);
    parameter_index.resize(
        std::min<size_t>(num_to_remove, parameter_index.size()));

    // Compute initializers for global variables that will be used to pass
    // arguments to the function.
    std::vector<uint32_t> initializer_ids;

    for (auto index : parameter_index) {
      const auto* type_inst =
          GetIRContext()->get_def_use_mgr()->GetDef(params[index]->type_id());
      assert(type_inst && "Type of function parameter is invalid");

      // Make sure type ids for global variables exist in the module.
      FindOrCreatePointerType(type_inst->result_id(), SpvStorageClassPrivate);

      // Create initializer for the global variable.
      //
      // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3403):
      //  Uncomment when the PR is merged.
      // initializer_ids.push_back(
      //     FindOrCreateVariableInitializer(type_inst->result_id()));
      initializer_ids.push_back(
          FindOrCreateZeroConstant(type_inst->result_id()));
    }

    // Compute type ids for the remaining arguments.
    std::vector<uint32_t> argument_ids;
    for (size_t i = 0, n = params.size(); i < n; ++i) {
      if (std::find(parameter_index.begin(), parameter_index.end(), i) ==
          parameter_index.end()) {
        argument_ids.push_back(params[i]->type_id());
      }
    }

    // Create new type for the function.
    auto new_type =
        FindOrCreateFunctionType(function.type_id(), std::move(argument_ids));

    // Create fresh ids for global variables.
    std::vector<uint32_t> var_ids(num_to_remove);
    std::generate(var_ids.begin(), var_ids.end(),
                  [this] { return GetFuzzerContext()->GetFreshId(); });

    ApplyTransformation(TransformationRemoveParameters(
        function.result_id(), new_type, std::move(parameter_index),
        std::move(var_ids), std::move(initializer_ids)));
  }
}

}  // namespace fuzz
}  // namespace spvtools
