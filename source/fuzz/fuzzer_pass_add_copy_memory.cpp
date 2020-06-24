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

#include "source/fuzz/fuzzer_pass_add_copy_memory.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_copy_memory.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddCopyMemory::FuzzerPassAddCopyMemory(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddCopyMemory::~FuzzerPassAddCopyMemory() = default;

void FuzzerPassAddCopyMemory::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor) {
        // Check that we can insert an OpCopyMemory before this instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCopyMemory,
                                                          inst_it)) {
          return;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingCopyMemory())) {
          return;
        }

        // Get all instructions available before |inst_it| according to the
        // domination rules.
        auto instructions = FindAvailableInstructions(
            function, block, inst_it,
            [](opt::IRContext* context, opt::Instruction* inst) {
              // Instruction to apply OpCopyMemory to should have both result id
              // and type id.
              if (!inst->result_id() || !inst->type_id()) {
                return false;
              }

              const auto* type =
                  context->get_type_mgr()->GetType(inst->type_id());
              assert(type && "Type is nullptr for non-zero type id");

              // Instructions type should be OpTypePointer and the pointee
              // should be compatible with OpCopyMemory.
              return type->AsPointer() &&
                     TransformationAddCopyMemory::CanUsePointeeWithCopyMemory(
                         *type->AsPointer()->pointee_type());
            });

        if (instructions.empty()) {
          return;
        }

        const auto* inst =
            instructions[GetFuzzerContext()->RandomIndex(instructions)];

        // Decide whether to create global or local variable.
        auto storage_class = GetFuzzerContext()->ChooseEven()
                                 ? SpvStorageClassPrivate
                                 : SpvStorageClassFunction;

        auto pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
            GetIRContext(), inst->type_id());

        // Create a pointer type with |storage_class| if needed.
        FindOrCreatePointerType(pointee_type_id, storage_class);

        ApplyTransformation(TransformationAddCopyMemory(
            instruction_descriptor, GetFuzzerContext()->GetFreshId(),
            inst->result_id(), storage_class,
            FindOrCreateZeroConstant(pointee_type_id)));
      });
}

}  // namespace fuzz
}  // namespace spvtools
