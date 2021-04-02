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

#include <algorithm>
#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteFunctionVariables::FuzzerPassPermuteFunctionVariables(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}  // Here we call parent constructor

void FuzzerPassPermuteFunctionVariables::Apply() {
  // Permuting OpVariable instructions in each function
  for (auto& function : *GetIRContext()->module()) {
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfPermuteFunctionVariables())) {
      continue;
    }
    // Ids storage section
    auto first_block = function->entry().get();

    std::vector<opt::Instruction*> variables;
    for (auto& instruction : first_block) {
      if (instruction.opcode() == SpvOpVariable) {
        variables.push_back(&instruction);
      }
    }

    uint32_t vars_size = variables.size();

    // Permutation section
    // Create a vector, fill it with [0, n-1] values and shuffle it
    std::vector<uint32_t> permutation(vars_size);
    std::iota(permutation.begin(), permutation.end(), 0);
    GetFuzzerContext()->Shuffle(&permutation);

    std::vector<std::pair<uint32_t, uint32_t>> variables_pair_id;
    /*
    // Cycle notation, Apply product of transpositions because
    // Every Permutation can be written as a product of transpositions and
    // Transpositions Is special case of Permutation but between two numbers.
    // Mathematical formula I've followed
    // (a1,a2,...,as)=(as,as−1)∘(as,as−2)∘...∘(as,a2)∘(as,a1)
    */
    for (uint32_t changed_index = vars_size - 2; changed_index > 0;
         changed_index--) {
      variables_pair_id.push_back(std::make_pair(vars_size - 1, changed_index));
    }

    //  Apply Transformation
    for (const auto& pair_id : variables_pair_id) {
      ApplyTransformation(
          TransformationSwapFunctionVariables(pair_id.first, pair_id.second));
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
