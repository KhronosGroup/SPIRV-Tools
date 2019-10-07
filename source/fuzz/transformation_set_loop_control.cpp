// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_set_loop_control.h"

namespace spvtools {
namespace fuzz {

TransformationSetLoopControl::TransformationSetLoopControl(
        const spvtools::fuzz::protobufs::TransformationSetLoopControl& message)
        : message_(message) {}

TransformationSetLoopControl::TransformationSetLoopControl(
        uint32_t block_id, uint32_t loop_control, uint32_t peel_count, uint32_t partial_count) {
  message_.set_block_id(block_id);
  message_.set_loop_control(loop_control);
  message_.set_peel_count(peel_count);
  message_.set_partial_count(partial_count);
}

bool TransformationSetLoopControl::IsApplicable(
        opt::IRContext* context, const FactManager& /*unused*/) const {
  auto block = context->get_instr_block(message_.block_id());
  if (!block) {
    return false;
  }
  auto merge_inst = block->GetMergeInst();
  if (!merge_inst || merge_inst->opcode() != SpvOpLoopMerge) {
    return false;
  }

  uint32_t all_loop_control_mask_bits_set = SpvLoopControlUnrollMask | SpvLoopControlDontUnrollMask | SpvLoopControlDependencyInfiniteMask | SpvLoopControlDependencyLengthMask |
  SpvLoopControlMinIterationsMask | SpvLoopControlMaxIterationsMask |
  SpvLoopControlIterationMultipleMask |
  SpvLoopControlPeelCountMask |
  SpvLoopControlPartialCountMask;

  // The variable is only used in an assertion; the following keeps release-mode compilers happy.
  (void)(all_loop_control_mask_bits_set);

  // No additional bits should be set.
  assert (!(message_.loop_control() & ~all_loop_control_mask_bits_set));

  auto existing_loop_control_mask = merge_inst->GetSingleWordInOperand(2);

  // Check that there is no attempt to set one of the loop controls that requires guarantees to hold.
  for (SpvLoopControlMask mask : {SpvLoopControlDependencyInfiniteMask, SpvLoopControlDependencyLengthMask, SpvLoopControlMinIterationsMask, SpvLoopControlMaxIterationsMask,
                                  SpvLoopControlIterationMultipleMask}) {
    if (LoopControlBitIsAddedByTransformation(mask, existing_loop_control_mask)) {
      return false;
    }
  }

  if (!(message_.loop_control() & SpvLoopControlPeelCountMask) && message_.peel_count() > 0) {
    // Peel count provided, but peel count mask bit not set.
    return false;
  }

  if (!(message_.loop_control() & SpvLoopControlPartialCountMask) && message_.partial_count() > 0) {
    // Partial count provided, but partial count mask bit not set.
    return false;
  }

  // We must not be setting both 'don't unroll' and one of 'peel count' or 'partial count'.
  return !((message_.loop_control() & SpvLoopControlDontUnrollMask) && (message_.loop_control() & (SpvLoopControlPeelCountMask | SpvLoopControlPartialCountMask)));

}

void TransformationSetLoopControl::Apply(opt::IRContext* context,
                                              FactManager* /*unused*/) const {
  auto merge_inst = context->get_instr_block(message_.block_id())->GetMergeInst();
  auto existing_loop_control_mask = merge_inst->GetSingleWordInOperand(2);

  opt::Instruction::OperandList new_operands;
  new_operands.push_back(merge_inst->GetInOperand(0));
  new_operands.push_back(merge_inst->GetInOperand(1));
  new_operands.push_back({SPV_OPERAND_TYPE_LOOP_CONTROL, {message_.loop_control()}});

  uint32_t literal_index = 0;
  for (SpvLoopControlMask mask : {SpvLoopControlDependencyLengthMask, SpvLoopControlMinIterationsMask, SpvLoopControlMaxIterationsMask,
          SpvLoopControlIterationMultipleMask}) {
    if (existing_loop_control_mask & mask) {
      if (message_.loop_control() & mask) {
        new_operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {merge_inst->GetSingleWordInOperand(3 + literal_index)}});
      }
      literal_index++;
    }
  }

  if (message_.loop_control() & SpvLoopControlPeelCountMask) {
    new_operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {message_.peel_count()}});
  }

  if (message_.loop_control() & SpvLoopControlPartialCountMask) {
    new_operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {message_.partial_count()}});
  }

  merge_inst->SetInOperands(std::move(new_operands));
}

protobufs::Transformation TransformationSetLoopControl::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_set_loop_control() = message_;
  return result;
}

bool TransformationSetLoopControl::LoopControlBitIsAddedByTransformation(SpvLoopControlMask loop_control_single_bit_mask, uint32_t existing_loop_control_mask) const {
  return
  !(loop_control_single_bit_mask & existing_loop_control_mask) && (loop_control_single_bit_mask & message_.loop_control());
}

}  // namespace fuzz
}  // namespace spvtools
