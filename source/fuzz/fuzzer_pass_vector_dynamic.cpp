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

#include "source/fuzz/fuzzer_pass_vector_dynamic.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_vector_dynamic.h"

namespace spvtools {
namespace fuzz {

FuzzerPassVectorDynamic::FuzzerPassVectorDynamic(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassVectorDynamic::~FuzzerPassVectorDynamic() = default;

void FuzzerPassVectorDynamic::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& instruction : block) {
        // |instruction| must be an OpCompositeExtract or an OpCompositeInsert
        // instruction to consider applying the transformation.
        if (instruction.opcode() != SpvOpCompositeExtract &&
            instruction.opcode() != SpvOpCompositeInsert) {
          continue;
        }

        // The composite must be a vector.
        auto composite_instruction = GetIRContext()->get_def_use_mgr()->GetDef(
            instruction.GetSingleWordInOperand(
                instruction.opcode() == SpvOpCompositeExtract ? 0 : 1));
        if (!GetIRContext()
                 ->get_type_mgr()
                 ->GetType(composite_instruction->type_id())
                 ->AsVector()) {
          continue;
        }

        // Randomly decide whether to try applying the transformation.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfVectoringDynamic())) {
          continue;
        }

        // Make sure the |instruction| literal operand is defined as constant.
        // It will be used as operand of the vector dynamic instruction.
        FindOrCreateIntegerConstant(
            {instruction.GetSingleWordInOperand(
                instruction.opcode() == SpvOpCompositeExtract ? 1 : 2)},
            32, false, false);

        // Applies the vector dynamic transformation.
        ApplyTransformation(
            TransformationVectorDynamic(instruction.result_id()));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
