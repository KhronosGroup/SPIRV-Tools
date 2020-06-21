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
    // TODO():
    //  uncomment when the PR is merged.
    // auto params = fuzzerutil::GetParameters(GetIRContext(),
    //                                         function.result_id());
    std::vector<opt::Instruction*> params;

    if (params.empty() || fuzzerutil::FunctionIsEntryPoint(
                              GetIRContext(), function.result_id())) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfReplacingParametersWithStruct())) {
      continue;
    }

    std::vector<uint32_t> parameter_index(params.size());
    std::iota(parameter_index.begin(), parameter_index.end(), 0);
    GetFuzzerContext()->Shuffle(&parameter_index);
    parameter_index.resize(
        GetFuzzerContext()->GetRandomNumberOfReplacedParameters(
            static_cast<uint32_t>(params.size())));

    std::vector<uint32_t> new_argument_types;
    for (uint32_t i = 0, n = static_cast<uint32_t>(params.size()); i < n; ++i) {
      if (std::find(parameter_index.begin(), parameter_index.end(), i) ==
          parameter_index.end()) {
        new_argument_types.push_back(params[i]->type_id());
      }
    }

    auto component_type_ids = parameter_index;
    for (auto& id : component_type_ids) {
      id = params[id]->type_id();
    }

    new_argument_types.push_back(FindOrCreateStructType(component_type_ids));

    ApplyTransformation(TransformationReplaceParamsWithStruct(
        function.result_id(), parameter_index,
        FindOrCreateFunctionType(
            fuzzerutil::GetFunctionType(GetIRContext(), &function)
                ->GetSingleWordInOperand(0),
            new_argument_types),
        GetFuzzerContext()->GetFreshId(), GetFuzzerContext()->GetFreshId()));
  }
}

}  // namespace fuzz
}  // namespace spvtools
