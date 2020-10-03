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

#include "source/fuzz/fuzzer_pass_replace_branches_from_dead_blocks_with_exits.h"

#include <algorithm>
#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_replace_branch_from_dead_block_with_exit.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceBranchesFromDeadBlocksWithExits::
    FuzzerPassReplaceBranchesFromDeadBlocksWithExits(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceBranchesFromDeadBlocksWithExits::
    ~FuzzerPassReplaceBranchesFromDeadBlocksWithExits() = default;

void FuzzerPassReplaceBranchesFromDeadBlocksWithExits::Apply() {
  // TODO comment this method

  auto fragment_execution_model_guaranteed =
      std::all_of(GetIRContext()->module()->entry_points().begin(),
                  GetIRContext()->module()->entry_points().begin(),
                  [](opt::Instruction* entry_point) -> bool {
                    return entry_point->GetSingleWordInOperand(0) ==
                           SpvExecutionModelFragment;
                  });

  std::vector<TransformationReplaceBranchFromDeadBlockWithExit>
      candidate_transformations;

  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      if (GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()
                  ->GetChanceOfReplacingBranchFromDeadBlockWithExit())) {
        continue;
      }
      if (!TransformationReplaceBranchFromDeadBlockWithExit::BlockIsSuitable(
              GetIRContext(), *GetTransformationContext(), block)) {
        continue;
      }
      std::vector<SpvOp> opcodes = {SpvOpUnreachable};
      if (fragment_execution_model_guaranteed) {
        opcodes.emplace_back(SpvOpKill);
      }
      auto function_return_type =
          GetIRContext()->get_type_mgr()->GetType(function.type_id());
      if (function_return_type->AsVoid()) {
        opcodes.emplace_back(SpvOpReturn);
      } else if (fuzzerutil::CanCreateConstant(*function_return_type)) {
        opcodes.emplace_back(SpvOpReturnValue);
      }
      auto opcode = opcodes[GetFuzzerContext()->RandomIndex(opcodes)];
      candidate_transformations.emplace_back(
          TransformationReplaceBranchFromDeadBlockWithExit(
              block.id(), opcode,
              opcode == SpvOpReturnValue
                  ? FindOrCreateZeroConstant(function.type_id(), true)
                  : 0));
    }
  }

  while (!candidate_transformations.empty()) {
    // TODO comment that they can disable each other
    MaybeApplyTransformation(
        GetFuzzerContext()->RemoveAtRandomIndex(&candidate_transformations));
  }
}

}  // namespace fuzz
}  // namespace spvtools
