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

#include <deque>
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

  // |insert_before_inst| must belong to some block (i.e. it is neither a global
  // instruction nor function parameter).
  auto* insert_before_block = ir_context->get_instr_block(insert_before_inst);
  if (!insert_before_block) {
    return false;
  }

  // |target| must be valid.
  auto* target_inst = FindInstruction(message_.target(), ir_context);
  if (!target_inst) {
    return false;
  }

  // |target_inst| must belong to some block (i.e. it is neither a global
  // instruction nor function parameter).
  auto* target_block = ir_context->get_instr_block(target_inst);
  if (!target_block) {
    return false;
  }

  // We should be able to move |target| before |insert_before|.
  return CanMoveInstruction(
      ir_context,
      fuzzerutil::GetIteratorForInstruction(insert_before_block,
                                            insert_before_inst),
      fuzzerutil::GetIteratorForInstruction(target_block, target_inst));
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

  // Target instruction is not disposed after removal, thus it is OK to use it
  // again below.
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

bool TransformationMoveInstruction::ModifiesOrOrdersMemory(SpvOp opcode) {
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
      return !ModifiesOrOrdersMemory(opcode);
  }
}

bool TransformationMoveInstruction::CanMoveInstruction(
    opt::IRContext* ir_context, opt::BasicBlock::iterator insert_before_it,
    opt::BasicBlock::iterator target_it) {
  if (insert_before_it == target_it) {
    // We can't insert an instruction before itself.
    return false;
  }

  // Control-flow instructions, memory barriers/modifiers can't be moved.
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

  // Domination rules are transitive. Thus, if |target_inst| dominates
  // |insert_before_inst|, then the latter is dominated by the former's
  // dependencies as well. Otherwise, we need to ensure that |insert_before_it|
  // is dominated by all dependencies of |target_inst|.
  if (!dominator_analysis->Dominates(&*target_it, &*insert_before_it)) {
    for (uint32_t i = 0; i < target_it->NumInOperands(); ++i) {
      const auto& operand = target_it->GetInOperand(i);
      // TODO(review): Maybe use spvIsInIdType(SpvOp) here?
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

      // If |dependency| is a global instruction or a function parameter, it
      // will always dominate |insert_before_it| that points to a local
      // instruction.
      if (ir_context->get_instr_block(dependency) != nullptr &&
          !dominator_analysis->Dominates(dependency, &*insert_before_it)) {
        return false;
      }
    }
  }

  // As discussed in the previous comment, domination rules are transitive.
  // Thus, if |insert_before_inst| dominates |target_inst|, then the former
  // dominates all users of the latter as well. Otherwise, we need to ensure
  // that |insert_before_it| dominates all users of |target_inst|.
  if (!dominator_analysis->Dominates(&*insert_before_it, &*target_it)) {
    auto does_dominate_users = ir_context->get_def_use_mgr()->WhileEachUser(
        target_it->result_id(), [ir_context, dominator_analysis,
                                 insert_before_it](opt::Instruction* user) {
          if (!ir_context->get_instr_block(user)) {
            // A global instruction uses a local instruction: it didn't
            // break domination rules before the transformation - it
            // won't after. This situation can happen when |user| is an
            // OpDecorate or a similar instruction. Note that
            // OpFunctionParameter instruction can't use a local id.
            return true;
          }

          // There are no tricky corner cases if |insert_before| uses
          // |target|.
          return dominator_analysis->Dominates(&*insert_before_it, user);
        });

    if (!does_dominate_users) {
      return false;
    }
  }

  auto checker = [](const opt::Instruction& inst) {
    // TODO: we can allow some memory modifiers/barriers in nested function
    //  calls (e.g. OpStore that uses a local variable).
    return ModifiesOrOrdersMemory(inst.opcode());
  };

  // There can be no path from |insert_before_inst| to |target_inst| and vice
  // versa that contains a memory modifier/barrier.
  return !AnyPathContains(ir_context, insert_before_it, target_it, checker) &&
         !AnyPathContains(ir_context, target_it, insert_before_it, checker);
}

std::vector<uint32_t> TransformationMoveInstruction::GetBlockIdsFromAllPaths(
    opt::IRContext* ir_context, uint32_t source_block_id,
    uint32_t dest_block_id) {
  std::unordered_set<uint32_t> visited_block_ids;
  std::vector<std::vector<uint32_t>> successor_stack = {{source_block_id}};
  std::vector<size_t> path = {0};
  std::vector<uint32_t> result;

  while (!path.empty()) {
    assert(successor_stack.size() == path.size() &&
           "Invariant: |successor_stack| and |path| should have the same size");

    auto index = path.back();
    const auto& current_successors = successor_stack.back();

    assert(index != current_successors.size() &&
           "Invariant: |path| always contains valid indices");
    assert(!visited_block_ids.count(current_successors[index]) &&
           "Invariant: top of the stack always contains an unvisited block");

    if (current_successors[index] == dest_block_id) {
      assert(!path.empty() &&
             "|path.size() - 1| causes underflow if |path| is empty");
      for (size_t i = 0; i < path.size() - 1; ++i) {
        result.push_back(successor_stack[i][path[i]]);
      }

      ++index;
    } else {
      visited_block_ids.insert(current_successors[index]);

      while (index < current_successors.size()) {
        if (!visited_block_ids.count(current_successors[index++])) {
          break;
        }
      }
    }

    if (index == current_successors.size()) {
      path.pop_back();
      successor_stack.pop_back();
      continue;
    }

    path.back() = index;

    std::vector<uint32_t> successors;
    ir_context->cfg()
        ->block(current_successors[index - 1])
        ->ForEachSuccessorLabel(
            [&successors](uint32_t id) { successors.push_back(id); });

    successor_stack.push_back(std::move(successors));
    path.push_back(0);
  }

  return result;
}

bool TransformationMoveInstruction::AnyPathContains(
    opt::IRContext* ir_context, opt::BasicBlock::iterator source_it,
    opt::BasicBlock::iterator dest_it,
    const std::function<bool(const opt::Instruction&)>& check) {
  auto* source_block = ir_context->get_instr_block(&*source_it);
  auto* dest_block = ir_context->get_instr_block(&*dest_it);

  assert(source_block && dest_block &&
         "|source_it| and |dest_it| can't point to global instructions "
         "or function parameters");

  auto check_with_functions = [ir_context,
                               &check](const opt::Instruction& inst) {
    if (check(inst)) {
      return true;
    }

    if (inst.opcode() == SpvOpFunctionCall) {
      const auto* function =
          fuzzerutil::FindFunction(ir_context, inst.GetSingleWordInOperand(0));
      assert(function && "|function_id| is invalid");

      return function->WhileEachInst(
          [&check](const opt::Instruction* inst) { return check(*inst); });
    }

    return false;
  };

  if (source_block == dest_block) {
    assert(IteratorsAreOrderedCorrectly(source_block, source_it, dest_it) &&
           "|source_it| must precede |dest_it| in the block");
    return std::any_of(source_it, dest_it, check_with_functions);
  }

  assert(IteratorsAreOrderedCorrectly(source_block, source_it,
                                      source_block->end()) &&
         "|source_it| is invalid");
  assert(IteratorsAreOrderedCorrectly(dest_block, dest_block->begin(),
                                      dest_it) &&
         "|dest_it| is invalid");

  if (std::any_of(source_it, source_block->end(), check_with_functions) ||
      std::any_of(dest_block->begin(), dest_it, check_with_functions)) {
    return true;
  }

  // We are using a DFS with explicit stack.
  auto block_ids_on_paths =
      GetBlockIdsFromAllPaths(ir_context, source_block->id(), dest_block->id());

  if (block_ids_on_paths.empty() && !source_block->IsSuccessor(dest_block)) {
    return false;
  }

  for (auto id : block_ids_on_paths) {
    const auto* block = ir_context->cfg()->block(id);
    if (std::any_of(block->begin(), block->end(), check_with_functions)) {
      return true;
    }
  }

  return false;
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
