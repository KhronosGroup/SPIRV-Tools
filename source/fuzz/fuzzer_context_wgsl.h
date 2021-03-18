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

#ifndef SOURCE_FUZZ_FUZZER_CONTEXT_WGSL_H_
#define SOURCE_FUZZ_FUZZER_CONTEXT_WGSL_H_

#include "source/fuzz/fuzzer_context.h"

namespace spvtools {
namespace fuzz {

// This context should be used when fuzzing SPIR-V for WGSL shaders since the
// latter has some limitations (e.g. signed and unsigned types are not
// interchangeable).
class FuzzerContextWgsl : public FuzzerContext {
 public:
  // See docs in the |FuzzerContext| on what this constructor does.
  FuzzerContextWgsl(std::unique_ptr<RandomGenerator> random_generator,
                    uint32_t min_fresh_id);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_CONTEXT_WGSL_H_
