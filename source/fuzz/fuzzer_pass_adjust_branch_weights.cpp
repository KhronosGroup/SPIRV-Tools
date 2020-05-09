// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/fuzzer_pass_adjust_branch_weights.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_adjust_branch_weights.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAdjustBranchWeights::FuzzerPassAdjustBranchWeights(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAdjustBranchWeights::~FuzzerPassAdjustBranchWeights() = default;

void FuzzerPassAdjustBranchWeights::Apply() {
  auto ir_context = GetIRContext();
  auto fuzzer_context = GetFuzzerContext();
  // For all OpBranchConditional instructions with branch weights,
  // randomly applies the transformation.
  ir_context->module()->ForEachInst([this, ir_context, fuzzer_context](
                                        opt::Instruction* instruction) {
    if (instruction->opcode() == SpvOpBranchConditional &&
        instruction->NumOperands() == 5 &&
        fuzzer_context->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfAdjustingBranchWeights())) {
      std::pair<uint32_t, uint32_t> branch_weights = {0, 0};
      while (branch_weights.first == 0 && branch_weights.second == 0) {
        branch_weights.first = fuzzer_context->GetRandomUint32(INT32_MAX);
        branch_weights.second =
            fuzzer_context->GetRandomUint32(INT32_MAX - branch_weights.first);
      }

      auto instruction_descriptor =
          MakeInstructionDescriptor(ir_context, instruction);
      auto transformation = TransformationAdjustBranchWeights(
          instruction_descriptor, branch_weights);
      ApplyTransformation(transformation);
    }
  });
}

}  // namespace fuzz
}  // namespace spvtools
