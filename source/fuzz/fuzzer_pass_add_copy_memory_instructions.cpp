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

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_pass_add_copy_memory_instructions.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_copy_memory.h"
#include "source/fuzz/transformation_add_local_variable.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddCopyMemoryInstructions::FuzzerPassAddCopyMemoryInstructions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddCopyMemoryInstructions::~FuzzerPassAddCopyMemoryInstructions() =
    default;

void FuzzerPassAddCopyMemoryInstructions::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* /*unused*/,
             opt::BasicBlock::iterator instr_it,
             const protobufs::InstructionDescriptor& /*unused*/) {
        const auto& instr = *instr_it;

        // Omit the instruction if it doesn't have either the result id or
        // the type id.
        if (!instr.result_id() || !instr.type_id()) {
          return;
        }

        const auto* type_instr = GetIRContext()->get_def_use_mgr()->GetDef(instr.type_id());
        if (!type_instr || type_instr->IsOpaqueType() || type_instr->opcode() != SpvOpTypePointer) {
          // Abort if result type is invalid, opaque or not a pointer.
          // TODO: we could probably introduce a new pointer type in the last case.
          return;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfAddingCopyMemoryInstructions())) {
          return;
        }

        // Make sure there exists a pointer type with Function storage class.
        auto pointee_type_id = type_instr->GetSingleWordInOperand(1);
        auto pointer_type_id = FindOrCreatePointerType(pointee_type_id, SpvStorageClassFunction);
        auto variable_id = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationAddLocalVariable(variable_id, pointer_type_id,
            function->result_id(), FindOrCreateZeroConstant(pointee_type_id), true));

        // We are creating a new instruction descriptor since we need to insert OpCopyMemory
        // after the original instruction.
        auto next_inst_iter = ++instr_it;
        ApplyTransformation(TransformationAddCopyMemory(
            MakeInstructionDescriptor(GetIRContext(), &*next_inst_iter),
            variable_id, instr.result_id()));
      });
}

}  // namespace fuzz
}  // namespace spvtools
