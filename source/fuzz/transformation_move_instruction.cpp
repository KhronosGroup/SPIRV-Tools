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

#include "source/fuzz/transformation_move_instruction.h"

#include <queue>
#include <unordered_set>
#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationMoveInstruction::TransformationMoveInstruction(
    const protobufs::TransformationMoveInstruction& message)
    : message_(message) {}

TransformationMoveInstruction::TransformationMoveInstruction(
    const protobufs::InstructionDescriptor& insert_before,
    const protobufs::InstructionDescriptor& target) {
  *message_.mutable_insert_before() = insert_before;
  *message_.mutable_target() = target;
}

bool TransformationMoveInstruction::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |insert_before| must be valid.
  auto* insert_before_inst =
      FindInstruction(message_.insert_before(), ir_context);
  if (!insert_before_inst) {
    return false;
  }

  auto* target_inst = FindInstruction(message_.target(), ir_context);
  if (!target_inst) {
    return false;
  }

  return CanMoveInstruction(
      ir_context,
      fuzzerutil::GetIteratorForInstruction(
          ir_context->get_instr_block(insert_before_inst), insert_before_inst),
      fuzzerutil::GetIteratorForInstruction(
          ir_context->get_instr_block(target_inst), target_inst));
}

void TransformationMoveInstruction::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* insert_before_inst =
      FindInstruction(message_.insert_before(), ir_context);
  assert(insert_before_inst && "|insert_before| descriptor is invalid");

  auto* target_inst = FindInstruction(message_.target(), ir_context);
  assert(target_inst && "|target| descriptor is invalid");

  assert(insert_before_inst != target_inst &&
         "Can't insert target instruction before itself");

  // Target instruction is not disposed after removal.
  target_inst->RemoveFromList();

  insert_before_inst->InsertBefore(
      std::unique_ptr<opt::Instruction>(target_inst));

  ir_context->set_instr_block(target_inst,
                              ir_context->get_instr_block(insert_before_inst));

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationMoveInstruction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_move_instruction() = message_;
  return result;
}

bool TransformationMoveInstruction::IsMemoryBarrier(SpvOp opcode) {
  switch (opcode) {
    case SpvOpNop:
    case SpvOpUndef:
    case SpvOpFunctionCall:
    case SpvOpLoad:
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
    case SpvOpDPdx:
    case SpvOpDPdy:
    case SpvOpFwidth:
    case SpvOpDPdxFine:
    case SpvOpDPdyFine:
    case SpvOpFwidthFine:
    case SpvOpDPdxCoarse:
    case SpvOpDPdyCoarse:
    case SpvOpFwidthCoarse:
    case SpvOpPhi:
    case SpvOpLoopMerge:
    case SpvOpSelectionMerge:
    case SpvOpLabel:
    case SpvOpBranch:
    case SpvOpBranchConditional:
    case SpvOpSwitch:
    case SpvOpKill:
    case SpvOpReturn:
    case SpvOpReturnValue:
    case SpvOpUnreachable:
    case SpvOpNoLine:
    case SpvOpSizeOf:
    case SpvOpCopyLogical:
    case SpvOpPtrEqual:
    case SpvOpPtrNotEqual:
    case SpvOpPtrDiff:
      return false;
    default:
      return true;
  }
}

bool TransformationMoveInstruction::CanMoveOpcode(SpvOp opcode) {
  switch (opcode) {
    case SpvOpSwitch:
    case SpvOpLabel:
    case SpvOpBranch:
    case SpvOpBranchConditional:
    case SpvOpSelectionMerge:
    case SpvOpLoopMerge:
    case SpvOpReturn:
    case SpvOpReturnValue:
    case SpvOpFunctionEnd:
    case SpvOpKill:
    case SpvOpUnreachable:
    case SpvOpPhi:
    case SpvOpVariable:
      return false;
    default:
      return !IsMemoryBarrier(opcode);
  }
}

bool TransformationMoveInstruction::CanMoveInstruction(
    opt::IRContext* ir_context, opt::BasicBlock::iterator insert_before_it,
    opt::BasicBlock::iterator target_it) {
  if (insert_before_it == target_it) {
    // We can't insert an instruction before itself.
    return false;
  }

  // Control-flow instructions and memory barriers can't be moved.
  if (!CanMoveOpcode(target_it->opcode())) {
    return false;
  }

  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(target_it->opcode(),
                                                    insert_before_it)) {
    return false;
  }

  const auto* insert_before_block =
      ir_context->get_instr_block(&*insert_before_it);
  const auto* target_block = ir_context->get_instr_block(&*target_it);

  // Global instructions and function parameters are not supported by this
  // transformation.
  if (!insert_before_block || !target_block) {
    return false;
  }

  // Both instructions must be in the same function.
  if (insert_before_block->GetParent() != target_block->GetParent()) {
    return false;
  }

  const auto* dominator_analysis =
      ir_context->GetDominatorAnalysis(insert_before_block->GetParent());

  if (dominator_analysis->Dominates(&*insert_before_it, &*target_it)) {
    // We are moving |target| instruction up in the control flow graph. Since
    // |insert_before| dominates |target| and the domination rules are
    // transitive, we can infer that |insert_before| dominates all usages of
    // |target| instruction. We only need to check that
    // |insert_before| is dominated by all dependencies of |target|.

    for (uint32_t i = 0, n = target_it->NumInOperands(); i < n; ++i) {
      const auto& operand = target_it->GetInOperand(i);
      if (operand.type != SPV_OPERAND_TYPE_ID) {
        continue;
      }

      auto id = operand.words[0];
      if (insert_before_it->result_id() == id) {
        // A small corner case: if |target| depends on |insert_before|, this
        // transformation will break domination rules if applied.
        return false;
      }

      auto* dependency = ir_context->get_def_use_mgr()->GetDef(id);
      assert(dependency && "|target| depends on invalid id");

      // If |dependency| is a global instruction, it will always dominate
      // |insert_before_it| that points to a local instruction.
      if (ir_context->get_instr_block(dependency) != nullptr &&
          !dominator_analysis->Dominates(dependency, &*insert_before_it)) {
        return false;
      }
    }

    // Check that none of the paths from |insert_before_block| to |target_block|
    // contain memory barriers.
    return PathsHaveNoMemoryBarriers(ir_context, insert_before_it, target_it);
  } else if (dominator_analysis->Dominates(&*target_it, &*insert_before_it)) {
    // We are moving |target| instruction down in control flow graph. Since
    // |target| dominates |insert_before| and domination rules are transitive,
    // we know that |insert_before| is dominated by all dependencies of |target|
    // as well. We have to make sure that |insert_before| dominates all usages
    // of |target|.

    return ir_context->get_def_use_mgr()->WhileEachUser(
               target_it->result_id(),
               [ir_context, dominator_analysis,
                insert_before_it](opt::Instruction* user) {
                 if (!ir_context->get_instr_block(user)) {
                   // A global instruction uses a local instruction: it didn't
                   // break domination rules before the transformation - it
                   // won't after. This situation can happen when |user| is an
                   // OpDecorate instruction.
                   return true;
                 }

                 // There are no tricky corner cases if |insert_before| uses
                 // |target|.
                 return dominator_analysis->Dominates(&*insert_before_it, user);
               }) &&
           PathsHaveNoMemoryBarriers(ir_context, target_it, insert_before_it);
  } else {
    // Domination rules will be broken if neither of blocks dominates the
    // other one.
    //
    // TODO(review): I've been thinking whether its possible to have no
    //  domination relationship between |insert_before| and |target| and still
    //  be able to move the instruction without invalidating the module. I
    //  figured that it's not possible without breaking the rules of structured
    //  control flow. In any case, I think it's better to skip this
    //  situation here.
    return false;
  }
}

bool TransformationMoveInstruction::PathsHaveNoMemoryBarriers(
    opt::IRContext* ir_context, opt::BasicBlock::iterator source_it,
    opt::BasicBlock::iterator dest_it) {
  auto* source_block = ir_context->get_instr_block(&*source_it);
  auto* dest_block = ir_context->get_instr_block(&*dest_it);

  if (ir_context->GetStructuredCFGAnalysis()->ContainingLoop(
          source_block->id()) != 0 ||
      ir_context->GetStructuredCFGAnalysis()->ContainingLoop(
          dest_block->id()) != 0) {
    // TODO: We don't handle cases when an instruction is moved in or out of the
    //  loop.
    return false;
  }

  std::queue<opt::BasicBlock::iterator> q({source_it});
  std::unordered_set<uint32_t> visited_blocks;

  while (!q.empty()) {
    auto it = q.front();
    q.pop();

    auto* block = ir_context->get_instr_block(&*it);
    if (visited_blocks.find(block->id()) != visited_blocks.end()) {
      continue;
    }

    auto end = block == dest_block ? dest_it : block->end();
    assert(IteratorsAreOrderedCorrectly(block, it, end) &&
           "|it| and |end| must belong to the same block and |it| must precede "
           "|end|");

    visited_blocks.insert(block->id());

    for (; it != end; ++it) {
      if (IsMemoryBarrier(it->opcode())) {
        return false;
      }
    }

    if (block != dest_block) {
      block->ForEachSuccessorLabel([&q, ir_context](uint32_t block_id) {
        q.push(ir_context->cfg()->block(block_id)->begin());
      });
    }
  }

  return true;
}

bool TransformationMoveInstruction::IteratorsAreOrderedCorrectly(
    opt::BasicBlock* block, opt::BasicBlock::iterator first,
    opt::BasicBlock::iterator second) {
  if (first == second) {
    return true;
  }

  std::vector<opt::BasicBlock::iterator> order;
  order.reserve(2);

  for (auto it = block->begin(), end = block->end(); it != end; ++it) {
    if (it == first || it == second) {
      order.push_back(it);
    }
  }

  if (order.size() == 1) {
    return (first == block->end() && order[0] == second) ||
           (second == block->end() && order[0] == first);
  }

  return order.size() == 2 && order[0] == first && order[1] == second;
}

}  // namespace fuzz
}  // namespace spvtools
