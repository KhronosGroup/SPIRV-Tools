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
      [this](opt::Function* /*unused*/, opt::BasicBlock* /*unused*/,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& /*unused*/) {
        auto& inst = *inst_it;

        // Skip the instruction if it doesn't have either the result id or
        // the type id.
        if (!inst.result_id() || !inst.type_id()) {
          return;
        }

        const auto* type =
            GetIRContext()->get_type_mgr()->GetType(inst.type_id());
        assert(type && "Type is nullptr for non-zero type id");

        if (!type->AsPointer() ||
            !TransformationAddCopyMemory::CanUsePointeeWithCopyMemory(
                *type->AsPointer()->pointee_type())) {
          // Abort if result type is invalid, opaque or not a pointer.
          return;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingCopyMemory())) {
          return;
        }

        auto next_iter = inst_it;
        ++next_iter;
        // Abort if can't insert OpCopyMemory before next instruction (i.e.
        // after the current one).
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCopyMemory,
                                                          next_iter)) {
          return;
        }

        // Decide whether to create global or local variable. If the copied
        // object is a pointer, create a global variable. Otherwise, decide at
        // random.
        //
        // TODO():
        //  We could choose the storage class completely at random if we were to
        //  initialize global variables that point to pointers.
        auto storage_class = type->AsPointer()->pointee_type()->AsPointer() ||
                                     GetFuzzerContext()->ChooseEven()
                                 ? SpvStorageClassPrivate
                                 : SpvStorageClassFunction;

        auto pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
            GetIRContext(), inst.type_id());

        // Create initializer for the variable.
        //
        // TODO():
        //  We are leaving the variable uninitialized if the copied type is a
        //  pointer. Fix this.
        auto initializer_id = type->AsPointer()->pointee_type()->AsPointer()
                                  ? 0
                                  : FindOrCreateZeroConstant(pointee_type_id);

        // Create a pointer type with |storage_class| if needed.
        FindOrCreatePointerType(pointee_type_id, storage_class);

        // We need to create a new instruction descriptor for the next
        // instruction in the block. It will be used to insert OpCopyMemory
        // above the instruction it points to (i.e. below |inst|).

        ApplyTransformation(TransformationAddCopyMemory(
            MakeInstructionDescriptor(GetIRContext(), &*next_iter),
            GetFuzzerContext()->GetFreshId(), inst.result_id(), storage_class,
            initializer_id));
      });
}

}  // namespace fuzz
}  // namespace spvtools
