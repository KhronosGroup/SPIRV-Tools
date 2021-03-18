// Copyright (c) 2021 Vasyl Teliman
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

#include "source/fuzz/fuzzer_context_wgsl.h"

namespace spvtools {
namespace fuzz {

FuzzerContextWgsl::FuzzerContextWgsl(
    std::unique_ptr<RandomGenerator> random_generator, uint32_t min_fresh_id)
    : FuzzerContext(std::move(random_generator), min_fresh_id) {
  chance_of_interchanging_signedness_of_integer_operands_ = 0;
}

}  // namespace fuzz
}  // namespace spvtools
