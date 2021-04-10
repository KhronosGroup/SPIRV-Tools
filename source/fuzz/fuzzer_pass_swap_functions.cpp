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

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_swap_functions.h"

namespace spvtools {
namespace fuzz {

FuzzerPassSwapFunctions::FuzzerPassSwapFunctions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

void FuzzerPassSwapFunctions::Apply() {
  uint32_t func1_id;
  uint32_t func2_id;
  uint32_t func_size = (uint32_t)(GetIRContext()->module()->end() -
                                  GetIRContext()->module()->begin());
  srand((unsigned)time(0));

  for (int i = 0; i < NUM_SWAPS; i++) {
    func1_id = rand() % func_size;
    func2_id = rand() % func_size;

    ApplyTransformation(TransformationSwapFunctions(func1_id, func2_id));
  }
}

}  // namespace fuzz
}  // namespace spvtools
