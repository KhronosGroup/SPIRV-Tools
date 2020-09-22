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

#ifndef SOURCE_FUZZ_FUZZER_PASS_SPLIT_LOOPS_H_
#define SOURCE_FUZZ_FUZZER_PASS_SPLIT_LOOPS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass that iterates over the whole module. For each function it finds
// loop headers. For each loop it randomly decides whether to apply the
// corresponding transformation SplitLoops.
class FuzzerPassSplitLoops : public FuzzerPass {
 public:
  FuzzerPassSplitLoops(opt::IRContext* ir_context,
                       TransformationContext* transformation_context,
                       FuzzerContext* fuzzer_context,
                       protobufs::TransformationSequence* transformations);

  ~FuzzerPassSplitLoops() override;

  void Apply() override;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_SPLIT_LOOPS_H_
