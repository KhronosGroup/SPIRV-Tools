// Copyright (c) 2020 Google LLC
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
#include "source/fuzz/fuzzer_pass_permute_function_parameters.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_permute_function_parameters.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteFunctionParameters::FuzzerPassPermuteFunctionParameters(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassPermuteFunctionParameters::~FuzzerPassPermuteFunctionParameters() = default;

void FuzzerPassPermuteFunctionParameters::Apply() {
  for (const auto& function : *GetIRContext()->module()) {
    uint32_t function_id = function.result_id();

    // Skip the function if it is an entry point
    if (fuzzerutil::FunctionIsEntryPoint(GetIRContext(), function_id)) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
        GetFuzzerContext()->GetChanceOfPermutingParameters())) {
      continue;
    }

    // Compute permutation for parameters
    auto* function_type = fuzzerutil::GetFunctionType(GetIRContext(), &function);
    assert(function_type && "Function type is null");

    std::vector<uint32_t> permutation(function_type->NumInOperands());
    std::iota(permutation.begin(), permutation.end(), 0);
    // Return type always remains the same
    GetFuzzerContext()->Shuffle(&permutation, 1, permutation.size() - 1);

    uint32_t fresh_type_id = GetFuzzerContext()->GetFreshId();

    std::vector<protobufs::InstructionDescriptor> call_sites;
    GetIRContext()->get_def_use_mgr()->ForEachUser(&function.DefInst(),
        [this, &call_sites, function_id](opt::Instruction* instruction) {
          if (instruction->opcode() != SpvOpFunctionCall) {
            return;
          }

          assert(instruction->GetSingleWordInOperand(0) == function_id &&
              "OpFunctionCall has wrong function id");

          call_sites.push_back(MakeInstructionDescriptor(GetIRContext(), instruction));
        });

    ApplyTransformation(TransformationPermuteFunctionParameters(
        function_id, fresh_type_id, permutation, call_sites));
  }
}

} // namespace fuzz
} // namespace spvtools
