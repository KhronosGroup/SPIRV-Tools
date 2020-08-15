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
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
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
  assert(inst_it != inst_block->end() &&
         "Can't get an iterator for the instruction");

  // |instruction| can't be the last instruction in the block.
  auto successor_it = ++inst_it;
  if (successor_it == inst_block->end()) {
    return false;
  }

  // We don't risk swapping a memory instruction with an unsupported one.
  if (!IsSimpleOpcode(inst->opcode()) &&
      !IsOpcodeSupported(successor_it->opcode())) {
    return false;
  }

  // We should be able to swap memory instructions without changing semantics of
  // the module.
  if (IsOpcodeSupported(successor_it->opcode()) &&
      !CanSwapMaybeSimpleInstructions(*inst, *successor_it,
                                      transformation_context)) {
    return false;
  }

  // Check that we can insert |instruction| after |inst_it|.
  auto successors_successor_it = ++inst_it;
  if (successors_successor_it == inst_block->end() ||
      !fuzzerutil::CanInsertOpcodeBeforeInstruction(inst->opcode(),
                                                    successors_successor_it)) {
    return false;
  }

  // Check that |instruction|'s successor doesn't depend on the |instruction|.
  if (inst->result_id()) {
    for (uint32_t i = 0; i < successor_it->NumInOperands(); ++i) {
      const auto& operand = successor_it->GetInOperand(i);
      if (operand.type == SPV_OPERAND_TYPE_ID &&
          operand.words[0] == inst->result_id()) {
        return false;
      }
    }
  }

  return true;
}

void TransformationMoveInstructionDown::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* inst = FindInstruction(message_.instruction(), ir_context);
  assert(inst &&
         "The instruction should've been validated in the IsApplicable");

  auto inst_it = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(inst), inst);

  // Move the instruction down in the block.
  inst->InsertAfter(&*++inst_it);

  ir_context->InvalidateAnalyses(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationMoveInstructionDown::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_move_instruction_down() = message_;
  return result;
}

bool TransformationMoveInstructionDown::IsOpcodeSupported(SpvOp opcode) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3605):
  //  We only support "simple" instructions that work don't with memory.
  //  We should extend this so that we support the ones that modify the memory
  //  too.
  return IsSimpleOpcode(opcode) || IsMemoryOpcode(opcode) ||
         IsBarrierOpcode(opcode);
}

bool TransformationMoveInstructionDown::IsSimpleOpcode(SpvOp opcode) {
  switch (opcode) {
    case SpvOpNop:
    case SpvOpUndef:
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
      // OpAccessChain and OpInBoundsAccessChain are considered simple
      // instructions since they result in a pointer to the object in memory,
      // not the object itself.
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
      return true;
    default:
      return false;
  }
}

bool TransformationMoveInstructionDown::IsMemoryReadOpcode(SpvOp opcode) {
  return opcode == SpvOpLoad || opcode == SpvOpCopyMemory;
}

uint32_t TransformationMoveInstructionDown::GetMemoryReadTarget(
    const opt::Instruction& inst) {
  switch (inst.opcode()) {
    case SpvOpLoad:
      return inst.GetSingleWordInOperand(0);
    case SpvOpStore:
    case SpvOpCopyMemory:
      return inst.GetSingleWordInOperand(1);
    default:
      assert(!IsMemoryOpcode(inst.opcode()) &&
             "Not all memory opcodes are handled");
      return 0;
  }
}

bool TransformationMoveInstructionDown::IsMemoryWriteOpcode(SpvOp opcode) {
  return opcode == SpvOpStore || opcode == SpvOpCopyMemory;
}

uint32_t TransformationMoveInstructionDown::GetMemoryWriteTarget(
    const opt::Instruction& inst) {
  switch (inst.opcode()) {
    case SpvOpLoad:
      return inst.result_id();
    case SpvOpStore:
    case SpvOpCopyMemory:
      return inst.GetSingleWordInOperand(0);
    default:
      assert(!IsMemoryOpcode(inst.opcode()) &&
             "Not all memory opcodes are handled");
      return 0;
  }
}

bool TransformationMoveInstructionDown::IsMemoryOpcode(SpvOp opcode) {
  return IsMemoryReadOpcode(opcode) || IsMemoryWriteOpcode(opcode);
}

bool TransformationMoveInstructionDown::IsBarrierOpcode(SpvOp opcode) {
  return opcode == SpvOpMemoryBarrier || opcode == SpvOpControlBarrier ||
         opcode == SpvOpMemoryNamedBarrier;
}

bool TransformationMoveInstructionDown::CanSwapMaybeSimpleInstructions(
    const opt::Instruction& a, const opt::Instruction& b,
    const TransformationContext& transformation_context) {
  assert(IsOpcodeSupported(a.opcode()) && IsOpcodeSupported(b.opcode()) &&
         "Both opcodes must be supported");

  // One of opcodes is simple - we can swap them without any side-effects.
  if (IsSimpleOpcode(a.opcode()) || IsSimpleOpcode(b.opcode())) {
    return true;
  }

  // Both parameters are either memory instruction or barriers.

  // One of the opcodes is a barrier - can't swap them.
  if (IsBarrierOpcode(a.opcode()) || IsBarrierOpcode(b.opcode())) {
    return false;
  }

  // Both parameters are memory instructions.

  // Both parameters only read from memory - it's OK to swap them.
  if (!IsMemoryWriteOpcode(a.opcode()) && !IsMemoryWriteOpcode(b.opcode())) {
    return true;
  }

  const auto* fact_manager = transformation_context.GetFactManager();

  // At least one of parameters is a memory read instruction.

  // In theory, we can swap two memory instructions, one of which reads
  // from the memory, if read target (the pointer the memory is read from) and
  // the write target (the memory is written into):
  // - point to different memory regions
  // - point to the same region with irrelevant value
  // - point to the same region and the region is not used in the block anymore.
  //
  // However, we can't currently determine if two pointers point to two
  // different memory regions. That being said, if two pointers are not
  // synonymous, they still might point to the same memory region. For example:
  //   %1 = OpVariable ...
  //   %2 = OpAccessChain %1 0
  //   %3 = OpAccessChain %1 0
  // In this pseudo-code, %2 and %3 are not synonymous but point to the same
  // memory location. This implies that we can't determine if some memory
  // location is not used in the block.
  //
  // With this in mind, consider two cases (we will build a table for each one):
  // - one instruction only reads from memory, the other one only writes to it.
  //   A - both point to the same memory region.
  //   B - both point to different memory regions.
  //   0, 1, 2 - neither, one of or both of the memory regions are irrelevant.
  //   |-| - can't swap; |+| - can swap.
  //     | 0 | 1 | 2 |
  //   A : -   +   +
  //   B : +   +   +
  // - both instructions write to memory. Notation is the same.
  //     | 0 | 1 | 2 |
  //   A : *   +   +
  //   B : +   +   +
  //   * - we can swap two instructions that write into the same non-irrelevant
  //   memory region if the written value is the same.
  //
  // Note that we can't always distinguish between A and B. Also note that
  // in case of A, if one of the instructions is marked with
  // PointeeValueIsIrrelevant, then the pointee of the other one is irrelevant
  // as well even if the instruction is not marked with that fact.
  //
  // TODO(): This procedure can be improved when we can determine if two
  //  pointers point to different memory regions.

  // From now on we will denote an instruction that:
  // - only reads from memory - R
  // - only writes into memory - W
  // - reads and writes - RW
  //
  // Both |a| and |b| can be either W or RW at this point. Additionally, at most
  // one of them can be R. The procedure below checks all possible combinations
  // of R, W and RW according to the tables above.

  if (IsMemoryReadOpcode(a.opcode()) && IsMemoryWriteOpcode(b.opcode()) &&
      !fact_manager->PointeeValueIsIrrelevant(GetMemoryReadTarget(a)) &&
      !fact_manager->PointeeValueIsIrrelevant(GetMemoryWriteTarget(b))) {
    return false;
  }

  if (IsMemoryWriteOpcode(a.opcode()) && IsMemoryReadOpcode(b.opcode()) &&
      !fact_manager->PointeeValueIsIrrelevant(GetMemoryWriteTarget(a)) &&
      !fact_manager->PointeeValueIsIrrelevant(GetMemoryReadTarget(b))) {
    return false;
  }

  if (IsMemoryWriteOpcode(a.opcode()) && IsMemoryWriteOpcode(b.opcode()) &&
      !fact_manager->PointeeValueIsIrrelevant(GetMemoryWriteTarget(a)) &&
      !fact_manager->PointeeValueIsIrrelevant(GetMemoryWriteTarget(b))) {
    auto read_target_a = GetMemoryReadTarget(a);
    auto read_target_b = GetMemoryReadTarget(b);

    // |read_target_a| and |read_target_b| might have different types.
    return read_target_a == read_target_b ||
           fact_manager->IsSynonymous(MakeDataDescriptor(read_target_a, {}),
                                      MakeDataDescriptor(read_target_b, {}));
  }

  return true;
}

}  // namespace fuzz
}  // namespace spvtools
