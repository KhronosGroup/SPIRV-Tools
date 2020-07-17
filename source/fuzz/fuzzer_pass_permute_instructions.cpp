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

#include "source/fuzz/fuzzer_pass_permute_instructions.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_move_instruction_down.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteInstructions::FuzzerPassPermuteInstructions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassPermuteInstructions::~FuzzerPassPermuteInstructions() = default;

void FuzzerPassPermuteInstructions::Apply() {
  // We are iterating over all instructions in all basic blocks.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      std::vector<opt::Instruction*> instructions;
      for (auto& instruction : block) {
        instructions.push_back(&instruction);
      }

      for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfPermutingInstructions())) {
          continue;
        }

        // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3458):
        //  Consider moving instructions between different basic blocks.
        TransformationMoveInstructionDown transformation(
            MakeInstructionDescriptor(GetIRContext(), *it));
        while (transformation.IsApplicable(GetIRContext(),
                                           *GetTransformationContext())) {
          transformation.Apply(GetIRContext(), GetTransformationContext());
          transformation = TransformationMoveInstructionDown(
              MakeInstructionDescriptor(GetIRContext(), *it));
        }
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
