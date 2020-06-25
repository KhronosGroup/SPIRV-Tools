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

#include <utility>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/instruction_message.h"
#include "source/fuzz/transformation_add_synonym.h"

namespace spvtools {
namespace fuzz {

TransformationAddSynonym::TransformationAddSynonym(
    protobufs::TransformationAddSynonym message)
    : message_(std::move(message)) {}

TransformationAddSynonym::TransformationAddSynonym(
    uint32_t result_id, const protobufs::Instruction& synonymous_instruction) {
  message_.set_result_id(result_id);
  *message_.mutable_synonymous_instruction() = synonymous_instruction;
}

bool TransformationAddSynonym::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Check that |message_.synonym_id| is valid.
  auto* synonym = ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  if (!synonym) {
    return false;
  }

  auto* block = ir_context->get_instr_block(synonym);
  assert(block && "Instruction must have a basic block");

  auto iter = fuzzerutil::GetIteratorForInstruction(block, synonym);
  ++iter;
  assert(iter != block->end() &&
         "Cannot create a synonym to the last instruction in the block");

  // Check that we can insert |message._synonymous_instruction| after
  // |message_.result_id|.
  // instruction.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
          static_cast<SpvOp>(message_.synonymous_instruction().opcode()),
          iter)) {
    return false;
  }

  // Check that synonymous instruction has fresh id.
  if (!fuzzerutil::IsFreshId(ir_context,
                             message_.synonymous_instruction().result_id())) {
    return false;
  }

  // Make sure that the instruction in |message_.synonymous_instruction| is
  // valid and the domination rules are satisfied.
  //
  // TODO(review): InstructionFromMessage updates module's id bound. Is this the
  //  desired behaviour?
  auto clone = fuzzerutil::CloneIRContext(ir_context);
  ApplyImpl(clone.get());
  return fuzzerutil::IsValid(clone.get(),
                             transformation_context.GetValidatorOptions());
}

void TransformationAddSynonym::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  ApplyImpl(ir_context);
  transformation_context->GetFactManager()->AddFactDataSynonym(
      MakeDataDescriptor(message_.result_id(), {}),
      MakeDataDescriptor(message_.synonymous_instruction().result_id(), {}),
      ir_context);
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

void TransformationAddSynonym::ApplyImpl(opt::IRContext* ir_context) const {
  const auto& synonymous_instruction = message_.synonymous_instruction();

  const auto* inst =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  assert(inst);

  auto iter = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(message_.result_id()), inst);
  assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(
             static_cast<SpvOp>(synonymous_instruction.opcode()), iter) &&
         "Can't insert synonymous instruction into the module");

  iter.InsertBefore(InstructionFromMessage(ir_context, synonymous_instruction));
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  synonymous_instruction.result_id());
}

protobufs::Transformation TransformationAddSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_synonym() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
