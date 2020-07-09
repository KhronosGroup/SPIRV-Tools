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

#include "source/fuzz/fuzzer_pass_replace_params_with_struct.h"

#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_replace_params_with_struct.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceParamsWithStruct::FuzzerPassReplaceParamsWithStruct(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceParamsWithStruct::~FuzzerPassReplaceParamsWithStruct() =
    default;

void FuzzerPassReplaceParamsWithStruct::Apply() {
  for (const auto& function : *GetIRContext()->module()) {
    auto params =
        fuzzerutil::GetParameters(GetIRContext(), function.result_id());

    if (params.empty() || fuzzerutil::FunctionIsEntryPoint(
                              GetIRContext(), function.result_id())) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfReplacingParametersWithStruct())) {
      continue;
    }

    std::vector<uint32_t> parameter_id(params.size());
    std::iota(parameter_id.begin(), parameter_id.end(), 0);

    // Remove unsupported parameters.
    auto new_end = std::remove_if(
        parameter_id.begin(), parameter_id.end(),
        [this, &params](uint32_t index) {
          const auto* type =
              GetIRContext()->get_type_mgr()->GetType(params[index]->type_id());
          return !type || !TransformationReplaceParamsWithStruct::
                              IsParameterTypeSupported(*type);
        });

    parameter_id.erase(new_end, parameter_id.end());

    if (parameter_id.empty()) {
      continue;
    }

    auto num_replaced_params = std::min<size_t>(
        parameter_id.size(),
        GetFuzzerContext()->GetRandomNumberOfReplacedParameters(
            static_cast<uint32_t>(params.size())));

    GetFuzzerContext()->Shuffle(&parameter_id);
    parameter_id.resize(num_replaced_params);

    // Make sure OpTypeStruct exists in the module.
    auto component_type_ids = parameter_id;
    for (auto& id : component_type_ids) {
      id = params[id]->type_id();
    }

    FindOrCreateStructType(component_type_ids);

    // Map parameters' indices to parameters' ids.
    for (auto& id : parameter_id) {
      id = params[id]->result_id();
    }

    ApplyTransformation(TransformationReplaceParamsWithStruct(
        parameter_id,
        /*fresh_function_type_id*/ GetFuzzerContext()->GetFreshId(),
        /*fresh_parameter_id*/ GetFuzzerContext()->GetFreshId(),
        /*fresh_composite_id*/
        GetFuzzerContext()->GetFreshIds(
            TransformationReplaceParamsWithStruct::GetNumberOfCallees(
                GetIRContext(), function.result_id()))));
  }
}

}  // namespace fuzz
}  // namespace spvtools
