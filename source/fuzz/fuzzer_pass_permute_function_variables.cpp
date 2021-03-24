// Copyright (c) 2021 Mostafa Ashraf
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

#include "source/fuzz/fuzzer_pass_permute_function_variables.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteFunctionVariables::FuzzerPassPermuteFunctionVariables(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {} // here call parent constructor

// use default destructor
FuzzerPassPermuteFunctionVariables::~FuzzerPassPermuteFunctionVariables() = default;

void FuzzerPassPermuteFunctionVariables::Apply() {
for (const auto& function : *GetIRContext()->module()) {
    uint32_t function_id = function.result_id();

    // Enty point mean something like e.g. main(), so skip it.
    // Because FunctionIsEntryPoint Returns |true| if one of entry points has function id |function_id|
    if (fuzzerutil::FunctionIsEntryPoint(GetIRContext(), function_id)) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfPermuteFunctionVariables())) {
      continue;
    }



    ApplyTransformation(TransformationSwapFunctionVariables(,,function_id));

}

}

}  // namespace fuzz
}  // namespace spvtools
