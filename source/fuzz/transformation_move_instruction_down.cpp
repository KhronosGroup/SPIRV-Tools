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

#include "source/fuzz/transformation_move_instruction_down.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationMoveInstructionDown::TransformationMoveInstructionDown(
    const protobufs::TransformationMoveInstructionDown& message)
    : message_(message) {}

TransformationMoveInstructionDown::TransformationMoveInstructionDown(
    const protobufs::InstructionDescriptor& instruction) {
  *message_.mutable_instruction() = instruction;
}

bool TransformationMoveInstructionDown::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |instruction| must be valid.
  auto* inst = FindInstruction(message_.instruction(), ir_context);
  if (!inst) {
    return false;
  }

  // Instruction's opcode must be supported by this transformation.
  if (!IsOpcodeSupported(inst->opcode())) {
    return false;
  }

  auto* inst_block = ir_context->get_instr_block(inst);
  assert(inst_block &&
         "Global instructions and function parameters are not supported");

  auto inst_it = fuzzerutil::GetIteratorForInstruction(inst_block, inst);

  // |instruction| can't be the last instruction in the block.
  ++inst_it;
  if (inst_it == inst_block->end()) {
    return false;
  }

  // Check that |instruction| doesn't dominate the next instruction in the
  // block.
  if (ir_context->GetDominatorAnalysis(inst_block->GetParent())
          ->Dominates(inst, &*inst_it)) {
    return false;
  }

  // Check that we can insert |instruction| after |inst_it| (note that, at this
  // point, |inst_it| points to the instruction after |instruction|).
  ++inst_it;
  return inst_it != inst_block->end() &&
         fuzzerutil::CanInsertOpcodeBeforeInstruction(inst->opcode(), inst_it);
}

void TransformationMoveInstructionDown::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* inst = FindInstruction(message_.instruction(), ir_context);
  assert(inst &&
         "The instruction should've been validated in the IsApplicable");

  auto inst_it = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(inst), inst);

  // Remove an instruction |inst| from the list and insert it after its previous
  // successor.
  inst->InsertAfter(&*++inst_it);
}

protobufs::Transformation TransformationMoveInstructionDown::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_move_instruction_down() = message_;
  return result;
}

bool TransformationMoveInstructionDown::IsOpcodeSupported(SpvOp opcode) {
  // TODO(): we only support "simple" instructions that work with memory.
  //  We should extend this so that we support the ones that modify the memory
  //  too.
  switch (opcode) {
    case SpvOpNop:
    case SpvOpUndef:
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
    case SpvOpArrayLength:
    case SpvOpVectorExtractDynamic:
    case SpvOpVectorInsertDynamic:
    case SpvOpVectorShuffle:
    case SpvOpCompositeConstruct:
    case SpvOpCompositeExtract:
    case SpvOpCompositeInsert:
    case SpvOpCopyObject:
    case SpvOpTranspose:
    case SpvOpConvertFToU:
    case SpvOpConvertFToS:
    case SpvOpConvertSToF:
    case SpvOpConvertUToF:
    case SpvOpUConvert:
    case SpvOpSConvert:
    case SpvOpFConvert:
    case SpvOpQuantizeToF16:
    case SpvOpSatConvertSToU:
    case SpvOpSatConvertUToS:
    case SpvOpBitcast:
    case SpvOpSNegate:
    case SpvOpFNegate:
    case SpvOpIAdd:
    case SpvOpFAdd:
    case SpvOpISub:
    case SpvOpFSub:
    case SpvOpIMul:
    case SpvOpFMul:
    case SpvOpUDiv:
    case SpvOpSDiv:
    case SpvOpFDiv:
    case SpvOpUMod:
    case SpvOpSRem:
    case SpvOpSMod:
    case SpvOpFRem:
    case SpvOpFMod:
    case SpvOpVectorTimesScalar:
    case SpvOpMatrixTimesScalar:
    case SpvOpVectorTimesMatrix:
    case SpvOpMatrixTimesVector:
    case SpvOpMatrixTimesMatrix:
    case SpvOpOuterProduct:
    case SpvOpDot:
    case SpvOpIAddCarry:
    case SpvOpISubBorrow:
    case SpvOpUMulExtended:
    case SpvOpSMulExtended:
    case SpvOpAny:
    case SpvOpAll:
    case SpvOpIsNan:
    case SpvOpIsInf:
    case SpvOpIsFinite:
    case SpvOpIsNormal:
    case SpvOpSignBitSet:
    case SpvOpLessOrGreater:
    case SpvOpOrdered:
    case SpvOpUnordered:
    case SpvOpLogicalEqual:
    case SpvOpLogicalNotEqual:
    case SpvOpLogicalOr:
    case SpvOpLogicalAnd:
    case SpvOpLogicalNot:
    case SpvOpSelect:
    case SpvOpIEqual:
    case SpvOpINotEqual:
    case SpvOpUGreaterThan:
    case SpvOpSGreaterThan:
    case SpvOpUGreaterThanEqual:
    case SpvOpSGreaterThanEqual:
    case SpvOpULessThan:
    case SpvOpSLessThan:
    case SpvOpULessThanEqual:
    case SpvOpSLessThanEqual:
    case SpvOpFOrdEqual:
    case SpvOpFUnordEqual:
    case SpvOpFOrdNotEqual:
    case SpvOpFUnordNotEqual:
    case SpvOpFOrdLessThan:
    case SpvOpFUnordLessThan:
    case SpvOpFOrdGreaterThan:
    case SpvOpFUnordGreaterThan:
    case SpvOpFOrdLessThanEqual:
    case SpvOpFUnordLessThanEqual:
    case SpvOpFOrdGreaterThanEqual:
    case SpvOpFUnordGreaterThanEqual:
    case SpvOpShiftRightLogical:
    case SpvOpShiftRightArithmetic:
    case SpvOpShiftLeftLogical:
    case SpvOpBitwiseOr:
    case SpvOpBitwiseXor:
    case SpvOpBitwiseAnd:
    case SpvOpNot:
    case SpvOpBitFieldInsert:
    case SpvOpBitFieldSExtract:
    case SpvOpBitFieldUExtract:
    case SpvOpBitReverse:
    case SpvOpBitCount:
    case SpvOpCopyLogical:
    case SpvOpPtrEqual:
    case SpvOpPtrNotEqual:
      return true;
    default:
      return false;
  }
}

}  // namespace fuzz
}  // namespace spvtools
