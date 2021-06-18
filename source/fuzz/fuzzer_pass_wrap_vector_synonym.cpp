// Copyright (c) 2021 Shiyu Liu
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

#include "source/fuzz/fuzzer_pass_wrap_vector_synonym.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/transformation_wrap_vector_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassWrapVectorSynonym::FuzzerPassWrapVectorSynonym(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

void FuzzerPassWrapVectorSynonym::Apply() {
//  ForEachInstructionWithInstructionDescriptor(
//      [this](opt::Function* function, opt::BasicBlock* block,
//             opt::BasicBlock::iterator instruction_iterator,
//             const protobufs::InstructionDescriptor& instruction_descriptor)
//          -> void {
//        assert(instruction_iterator->opcode() ==
//               instruction_descriptor.target_instruction_opcode() &&
//               "The opcode of the instruction we might insert before must be "
//               "the same as the opcode in the descriptor for the instruction");
//
//        // Randomly decide whether to try adding an OpVectorShuffle instruction.
//        if (!GetFuzzerContext()->ChoosePercentage(
//            GetFuzzerContext()->GetChanceOfWrappingVectorSynonym())) {
//          return;
//        }
//
//        // It must be valid to insert an OpVectorShuffle instruction
//        // before |instruction_iterator|.
//        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
//            SpvOpCompositeConstruct, instruction_iterator)) {
//          return;
//        }
//
//      });
}

}  // namespace fuzz
}  // namespace spvtools
