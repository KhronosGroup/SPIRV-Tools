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
        // Randomly decide whether to try applying the transformation.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()
                    ->GetChanceOfMakingVectorOperationDynamic())) {
          continue;
        }

        // |instruction| must be a vector operation.
        if (!TransformationVectorDynamic::IsVectorOperation(GetIRContext(),
                                                            &instruction)) {
          continue;
        }

        // Make sure |instruction| has only one indexing operand.
        assert(instruction.NumInOperands() ==
                   (instruction.opcode() == SpvOpCompositeExtract ? 2 : 3) &&
               "FuzzerPassVectorDynamic: the composite instruction must have "
               "only one indexing operand.");

        // Make sure the |instruction| literal operand is defined as constant.
        // It will be used as operand of the vector dynamic instruction.
        // If it is necessary to create the constant, then its signedness is
        // choosen randomly.
        if (!TransformationVectorDynamic::MaybeGetConstantForIndex(
                GetIRContext(), instruction, *GetTransformationContext())) {
          FindOrCreateIntegerConstant(
              {instruction.GetSingleWordInOperand(
                  instruction.opcode() == SpvOpCompositeExtract ? 1 : 2)},
              32, GetFuzzerContext()->ChooseEven() ? true : false, false);
        }

        // Applies the vector dynamic transformation.
        ApplyTransformation(
            TransformationVectorDynamic(instruction.result_id()));
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
