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

#include "source/fuzz/fuzzer_pass_inline_functions.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_inline_function.h"

namespace spvtools {
namespace fuzz {

FuzzerPassInlineFunctions::FuzzerPassInlineFunctions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassInlineFunctions::~FuzzerPassInlineFunctions() = default;

void FuzzerPassInlineFunctions::Apply() {
  std::vector<opt::Instruction*> function_call_instructions;

  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      for (auto& instruction : block) {
        // |instruction| must be OpFunctionCall to consider applying the
        // transformation.
        if (instruction.opcode() != SpvOpFunctionCall) {
          continue;
        }

        // |called_function| must not have an early return.
        auto called_function = fuzzerutil::FindFunction(
            GetIRContext(), instruction.GetSingleWordInOperand(0));
        if (called_function->HasEarlyReturn()) {
          continue;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfInliningFunction())) {
          continue;
        }

        function_call_instructions.push_back(&instruction);
        // Erases the function call instruction from the caller function.
      }
    }
  }

  for (auto& instruction : function_call_instructions) {
    auto called_function = fuzzerutil::FindFunction(
        GetIRContext(), instruction->GetSingleWordInOperand(0));

    // Mapping the called function instructions.
    std::map<uint32_t, uint32_t> result_id_map;
    for (auto& called_function_block : *called_function) {
      // The called function entry block label will not be inlined.
      if (&called_function_block != &*called_function->entry()) {
        result_id_map[called_function_block.GetLabelInst()->result_id()] =
            GetFuzzerContext()->GetFreshId();
      }

      for (auto& instruction_to_inline : called_function_block) {
        // The instructions are mapped to fresh ids.
        if (instruction_to_inline.HasResultId()) {
          result_id_map[instruction_to_inline.result_id()] =
              GetFuzzerContext()->GetFreshId();
        }
      }
    }

    // Applies the inline function transformation.
    ApplyTransformation(
        TransformationInlineFunction(result_id_map, instruction->result_id()));
  }
}

}  // namespace fuzz
}  // namespace spvtools
