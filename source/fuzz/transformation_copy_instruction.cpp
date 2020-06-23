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

#include "source/fuzz/transformation_copy_instruction.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationCopyInstruction::TransformationCopyInstruction(
    const protobufs::TransformationCopyInstruction& message)
    : message_(message) {}

TransformationCopyInstruction::TransformationCopyInstruction(
    const protobufs::InstructionDescriptor& instruction_descriptor,
    const std::vector<uint32_t>& fresh_ids) {
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
  for (auto id : fresh_ids) {
    message_.add_fresh_id(id);
  }
}

bool TransformationCopyInstruction::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |instruction_descriptor| should be valid.
  auto* inst = FindInstruction(message_.instruction_descriptor(), ir_context);
  if (!inst) {
    return false;
  }

  // We should be able to copy |inst| into predecessors.
  if (!CanCopyInstruction(ir_context, inst)) {
    return false;
  }

  const auto* inst_block = ir_context->get_instr_block(inst);
  assert(inst_block &&
         "CanCopyInstruction should've checked for global instructions");

  auto num_predecessors = ir_context->cfg()->preds(inst_block->id()).size();
  assert(num_predecessors != 0 &&
         "Basic block must have at least one predecessor");

  // Check that fresh ids are valid.
  std::vector<uint32_t> fresh_ids(message_.fresh_id().begin(),
                                  message_.fresh_id().end());
  if (fresh_ids.size() != num_predecessors) {
    return false;
  }

  return !fuzzerutil::HasDuplicates(fresh_ids) &&
         std::all_of(fresh_ids.begin(), fresh_ids.end(),
                     [ir_context](uint32_t id) {
                       return fuzzerutil::IsFreshId(ir_context, id);
                     });
}

void TransformationCopyInstruction::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* inst = FindInstruction(message_.instruction_descriptor(), ir_context);
  assert(inst && "|instruction_descriptor| is invalid");

  auto* inst_block = ir_context->get_instr_block(inst);
  assert(inst_block && "Can't copy global instruction");

  std::vector<uint32_t> fresh_ids(message_.fresh_id().begin(),
                                  message_.fresh_id().end());
  assert(!fresh_ids.empty() &&
         "Number of fresh ids should be equal to the number of predecessors");

  opt::Instruction::OperandList new_phi_operands;
  for (auto predecessor_id : ir_context->cfg()->preds(inst_block->id())) {
    // Clone the instruction. This will later be inserted into a predecessor
    // block. Clone method returns a raw pointer and transfers memory ownership
    // to the caller.
    std::unique_ptr<opt::Instruction> inst_clone(inst->Clone(ir_context));

    // The copied instruction must have a fresh id which will later be used in
    // the OpPhi instruction.
    inst_clone->SetResultId(fresh_ids.back());
    fresh_ids.pop_back();

    new_phi_operands.push_back(
        {SPV_OPERAND_TYPE_ID, {inst_clone->result_id()}});
    new_phi_operands.push_back({SPV_OPERAND_TYPE_ID, {predecessor_id}});

    // Change operands of the |inst_clone|.
    for (uint32_t i = 0; i < inst_clone->NumInOperands(); ++i) {
      auto& operand = inst_clone->GetInOperand(i);
      if (operand.type != SPV_OPERAND_TYPE_ID) {
        continue;
      }

      auto* dependency =
          ir_context->get_def_use_mgr()->GetDef(operand.words[0]);
      assert(dependency && "Instruction uses an invalid id");

      // We don't need to change anything if |dependency| is from a
      // different block since it should dominate the last instruction in the
      // predecessor.
      if (ir_context->get_instr_block(dependency) != inst_block) {
        continue;
      }

      // If the dependency is from this block, it must be an OpPhi. Otherwise,
      // this transformation will break the domination rules.
      assert(dependency->opcode() == SpvOpPhi &&
             "Only OpPhi dependencies of |inst| are allowed to be from the same"
             "basic block");

      // Iterate through all operands of the OpPhi dependency.
      for (uint32_t j = 1; j < dependency->NumInOperands(); j += 2) {
        if (dependency->GetSingleWordInOperand(j) == predecessor_id) {
          // |inst_clone| now directly depends on the OpPhi's operand.
          operand.words[0] = dependency->GetSingleWordInOperand(j - 1);
        }
      }
    }

    // Insert |inst_clone| as the last instruction in the predecessor block.
    fuzzerutil::UpdateModuleIdBound(ir_context, inst_clone->result_id());
    GetLastInsertBeforeIterator(ir_context->cfg()->block(predecessor_id),
                                inst_clone->opcode())
        .InsertBefore(std::move(inst_clone));
  }

  // Create a new OpPhi instruction at the beginning of the block.
  // |*inst_block->begin()| is technically the second instruction in the block
  // (OpLabel is stored separately). This transformation is not applicable to
  // the entry block of the function since it has no predecessors. This implies
  // that we shouldn't have any trouble with OpVariable instructions.
  assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpPhi,
                                                      inst_block->begin()) &&
         "Can't insert OpPhi at the beginning of the block");

  inst_block->begin()->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, SpvOpPhi, inst->type_id(), inst->result_id(),
      std::move(new_phi_operands)));

  // RemoveFromList method transfers memory ownership back tho the caller.
  inst->RemoveFromList();
  delete inst;

  // Make sure our changes are analyzed
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationCopyInstruction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_copy_instruction() = message_;
  return result;
}

bool TransformationCopyInstruction::CanCopyInstruction(
    opt::IRContext* ir_context, opt::Instruction* inst) {
  // An instruction must have a result id and a type id.
  if (!inst->result_id() || !inst->type_id()) {
    return false;
  }

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3477):
  //  uncomment when the PR is merged
  /*if (!fuzzerutil::CanMoveOpcode(inst->opcode())) {
    return false;
  }*/

  // Global instructions are not supported.
  auto* inst_block = ir_context->get_instr_block(inst);
  if (!inst_block) {
    return false;
  }

  // Check that instruction's operands allow us to move the instruction.
  for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
    const auto& operand = inst->GetInOperand(i);
    if (operand.type != SPV_OPERAND_TYPE_ID) {
      continue;
    }

    auto* dependency = ir_context->get_def_use_mgr()->GetDef(operand.words[0]);
    const auto* dependency_block = ir_context->get_instr_block(dependency);

    // We can't move the instruction if it depends on some instruction from the
    // same basic block that is not an OpPhi.
    if (dependency_block == inst_block && dependency->opcode() != SpvOpPhi) {
      return false;
    }
  }

  // The block must have at least one predecessor to copy the instruction into.
  auto predecessors = ir_context->cfg()->preds(inst_block->id());
  if (predecessors.empty()) {
    return false;
  }

  for (auto predecessor_id : predecessors) {
    auto* predecessor = ir_context->cfg()->block(predecessor_id);
    assert(predecessor && "|predecessor_id| is invalid");

    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3477):
    //  uncomment when the PR is merged.
    /*if (!fuzzerutil::PathsHaveNoMemoryBarriers(
            ir_context, GetLastInsertBeforeIterator(block, inst->opcode()),
            fuzzerutil::GetIteratorForInstruction(inst_block, inst))) {
      return false;
    }*/
  }

  return true;
}

opt::BasicBlock::iterator
TransformationCopyInstruction::GetLastInsertBeforeIterator(
    opt::BasicBlock* block, SpvOp opcode) {
  auto it = block->rbegin();
  while (it != block->rend() &&
         !fuzzerutil::CanInsertOpcodeBeforeInstruction(opcode, &*it)) {
    --it;
  }

  assert(it != block->rend() &&
         "Can't insert |inst| into a |predecessor| block");

  return (--it).base();
}

}  // namespace fuzz
}  // namespace spvtools
