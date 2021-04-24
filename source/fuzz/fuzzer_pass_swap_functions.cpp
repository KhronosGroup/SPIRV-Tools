// Copyright (c) 2021 Emiljano Gjiriti
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

#include "source/fuzz/fuzzer_pass_swap_functions.h"

#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_swap_functions.h"

namespace spvtools {
namespace fuzz {

FuzzerPassSwapFunctions::FuzzerPassSwapFunctions(
    opt::IRContext *ir_context, TransformationContext *transformation_context,
    FuzzerContext *fuzzer_context,
    protobufs::TransformationSequence *transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

void FuzzerPassSwapFunctions::Apply() {
  // This function iterates over the set of all result_ids
  // and it swaps two functions with 0.1 <= probability <=0.5
  // After each transformation it decides with a random probability
  // whether to perform another transformation or exit.
  std::vector<uint32_t> result_ids;
  for (auto &function : *GetIRContext()->module()) {
    result_ids.push_back(function.result_id());
  }
  for (auto id1 : result_ids) {
    for (auto id2 : result_ids) {
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfSwappingFunctions())) {
        continue;
      }
      ApplyTransformation(
          TransformationSwapFunctions(result_ids[id1], result_ids[id2]));
      if (!GetFuzzerContext()->GetChanceOfContinuingSwappingFunctions()) {
        break;
      }
    }
  }
}

} // namespace fuzz
} // namespace spvtools
