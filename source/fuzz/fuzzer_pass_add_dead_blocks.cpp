// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/fuzzer_pass_add_dead_blocks.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_dead_block.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddDeadBlocks::FuzzerPassAddDeadBlocks(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassAddDeadBlocks::~FuzzerPassAddDeadBlocks() = default;

void FuzzerPassAddDeadBlocks::Apply() {
  std::vector<opt::BasicBlock*> candidate_blocks;
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfAddingDeadBlock())) {
        continue;
      }
      if (block.IsLoopHeader()) {
        continue;
      }
      if (block.terminator()->opcode() != SpvOpBranch) {
        continue;
      }
      if (fuzzerutil::IsMergeOrContinue(
              GetIRContext(), block.terminator()->GetSingleWordInOperand(0))) {
        continue;
      }
      // TODO think about OpPhi here
      candidate_blocks.push_back(&block);
    }
  }
  while (!candidate_blocks.empty()) {
    uint32_t index = GetFuzzerContext()->RandomIndex(candidate_blocks);
    auto block = candidate_blocks.at(index);
    candidate_blocks.erase(candidate_blocks.begin() + index);
    // TODO: address OpPhi situation
    TransformationAddDeadBlock transformation(
        GetFuzzerContext()->GetFreshId(), block->id(),
        GetFuzzerContext()->ChooseEven(), {});
    if (transformation.IsApplicable(GetIRContext(), *GetFactManager())) {
      transformation.Apply(GetIRContext(), GetFactManager());
      *GetTransformations()->add_transformation() = transformation.ToMessage();
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
