// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/fuzzer_pass_move_instructions.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_move_instruction.h"

namespace spvtools {
namespace fuzz {

FuzzerPassMoveInstructions::FuzzerPassMoveInstructions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassMoveInstructions::~FuzzerPassMoveInstructions() = default;

void FuzzerPassMoveInstructions::Apply() {
  // We are iterating over all instructions in all basic blocks.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto insert_before_it = block.begin();
           insert_before_it != block.end(); ++insert_before_it) {
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfPermutingInstructions())) {
          continue;
        }

        // Compute a set of instructions that can be moved before
        // |insert_before_it|.
        auto move_candidates = ComputeMoveCandidates(&block, insert_before_it);
        if (move_candidates.empty()) {
          continue;
        }

        ApplyTransformation(TransformationMoveInstruction(
            MakeInstructionDescriptor(GetIRContext(), &*insert_before_it),
            MakeInstructionDescriptor(
                GetIRContext(), move_candidates[GetFuzzerContext()->RandomIndex(
                                    move_candidates)])));
      }
    }
  }
}

std::vector<opt::Instruction*>
FuzzerPassMoveInstructions::ComputeMoveCandidates(
    opt::BasicBlock* block, opt::BasicBlock::iterator insert_before_it) {
  std::vector<opt::Instruction*> result;

  for (auto it = block->begin(); it != block->end(); ++it) {
    if (TransformationMoveInstruction::CanMoveInstruction(
            GetIRContext(), insert_before_it, it)) {
      result.push_back(&*it);
    }
  }

  for (auto predecessor_id : GetIRContext()->cfg()->preds(block->id())) {
    auto* predecessor = GetIRContext()->cfg()->block(predecessor_id);
    assert(predecessor && "|predecessor_id| is invalid");
    for (auto it = predecessor->begin(); it != predecessor->end(); ++it) {
      if (TransformationMoveInstruction::CanMoveInstruction(
              GetIRContext(), insert_before_it, it)) {
        result.push_back(&*it);
      }
    }
  }

  block->ForEachSuccessorLabel(
      [this, &result, insert_before_it](uint32_t successor_id) {
        auto* successor = GetIRContext()->cfg()->block(successor_id);
        assert(successor && "|successor_id| is invalid");
        for (auto it = successor->begin(); it != successor->end(); ++it) {
          if (TransformationMoveInstruction::CanMoveInstruction(
                  GetIRContext(), insert_before_it, it)) {
            result.push_back(&*it);
          }
        }
      });

  return result;
}

}  // namespace fuzz
}  // namespace spvtools
