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

#ifndef SOURCE_FUZZ_FUZZER_PASS_FLATTEN_CONDITIONAL_BRANCHES_H
#define SOURCE_FUZZ_FUZZER_PASS_FLATTEN_CONDITIONAL_BRANCHES_H

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

class FuzzerPassFlattenConditionalBranches : public FuzzerPass {
 public:
  FuzzerPassFlattenConditionalBranches(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations);

  ~FuzzerPassFlattenConditionalBranches() override;

  void Apply() override;

  // Comparator for blocks, comparing them based on how deep they are nested
  // inside conditionals. Deeper blocks are considered less than ones that are
  // not as deep.
  class LessIfNestedMoreDeeply {
   public:
    explicit LessIfNestedMoreDeeply(opt::IRContext* ir_context)
        : ir_context_(ir_context) {}

    bool operator()(const opt::BasicBlock* bb1, opt::BasicBlock* bb2) const {
      return ir_context_->GetStructuredCFGAnalysis()->NestingDepth(bb1->id()) >
             ir_context_->GetStructuredCFGAnalysis()->NestingDepth(bb2->id());
    }

   private:
    opt::IRContext* ir_context_;
  };
};
}  // namespace fuzz
}  // namespace spvtools
#endif  // SOURCE_FUZZ_FUZZER_PASS_FLATTEN_CONDITIONAL_BRANCHES_H
