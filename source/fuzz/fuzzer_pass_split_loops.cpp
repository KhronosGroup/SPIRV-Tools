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

#include "source/fuzz/fuzzer_pass_split_loops.h"
#include "source/fuzz/transformation_split_loop.h"

namespace spvtools {
namespace fuzz {

FuzzerPassSplitLoops::FuzzerPassSplitLoops(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassSplitLoops::~FuzzerPassSplitLoops() = default;

void FuzzerPassSplitLoops::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    std::vector<opt::BasicBlock*> function_blocks;
    for (auto& block : function) {
      if (&block == &*function.begin()) {
        continue;
      }
      if (block.begin() == --block.end()) {
        continue;
      }
      function_blocks.push_back(&block);
    }
    for (auto& block : function_blocks) {
      if (block->GetLoopMergeInst()) {
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfSplittingLoop())) {
          continue;
        }
        auto merge_block = GetIRContext()->cfg()->block(block->MergeBlockId());
        auto loop_blocks = TransformationSplitLoop::GetRegionBlocks(
            GetIRContext(), block, merge_block);
        std::map<uint32_t, uint32_t> original_label_to_duplicate_label;
        std::map<uint32_t, uint32_t> original_id_to_duplicate_id;
        for (auto& loop_block : loop_blocks) {
          original_label_to_duplicate_label[loop_block->id()] =
              GetFuzzerContext()->GetFreshId();
          for (auto& instr : *loop_block) {
            if (instr.HasResultId()) {
              original_id_to_duplicate_id[instr.result_id()] =
                  GetFuzzerContext()->GetFreshId();
            }
          }
        }
        FindOrCreateIntegerConstant({0}, 32, false, false);
        FindOrCreateIntegerConstant({1}, 32, false, false);
        FindOrCreateBoolConstant(true, false);
        FindOrCreateBoolConstant(false, false);
        uint32_t constant_limit_id = FindOrCreateIntegerConstant(
            {GetFuzzerContext()->GetRandomLimitOfIterationsWhenSplittingLoop()},
            32, false, false);

        uint32_t local_unsigned_int_type = FindOrCreatePointerType(
            FindOrCreateIntegerType(32, false), SpvStorageClassFunction);
        uint32_t variable_counter_id = FindOrCreateLocalVariable(
            local_unsigned_int_type, function.result_id(), false);

        uint32_t local_bool_type = FindOrCreatePointerType(
            FindOrCreateBoolType(), SpvStorageClassFunction);
        uint32_t variable_run_second_id = FindOrCreateLocalVariable(
            local_bool_type, function.result_id(), false);
        TransformationSplitLoop transformation = TransformationSplitLoop(
            block->id(), variable_counter_id, variable_run_second_id,
            constant_limit_id, GetFuzzerContext()->GetFreshId(),
            GetFuzzerContext()->GetFreshId(), GetFuzzerContext()->GetFreshId(),
            GetFuzzerContext()->GetFreshId(), GetFuzzerContext()->GetFreshId(),
            GetFuzzerContext()->GetFreshId(), GetFuzzerContext()->GetFreshId(),
            std::move(original_label_to_duplicate_label),
            std::move(original_id_to_duplicate_id));
        MaybeApplyTransformation(transformation);
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
