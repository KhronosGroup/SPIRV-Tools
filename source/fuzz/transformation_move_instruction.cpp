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

#include <unordered_map>
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
        target_it->result_id(),
        [ir_context, dominator_analysis, insert_before_it, target_it,
         insert_before_block](opt::Instruction* user) {
          if (user->opcode() == SpvOpPhi) {
            // We must make sure that |insert_before_block| dominates
            // the relevant block in the |user|'s operands.

            for (uint32_t i = 0; i < user->NumInOperands(); i += 2) {
              if (user->GetSingleWordInOperand(i) == target_it->result_id()) {
                return dominator_analysis->Dominates(
                    insert_before_block->id(),
                    user->GetSingleWordInOperand(i + 1));
              }
            }

            assert(false && "We should've returned from the loop above");
            return false;
          }

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
  //
  // It is guaranteed that at least one path exists either from
  // |insert_before_it| to |target_it| or vice versa. To prove that, suppose
  // that there is no path from either |insert_before_it| to |target_it| or vice
  // versa and it is still possible to move |target_inst| before
  // |insert_before_inst|. In that case, both the former and the latter must
  // dominate the users of the former. That means that all paths from the
  // entry-point block to some user's block must contain both
  // |insert_before_inst| and |target_inst|. Which, in turn, implies that there
  // is a path from either |insert_before_inst| to |target_inst| or vice versa -
  // contradiction.
  return !AnyExecutionPathContains(ir_context, insert_before_it, target_it,
                                   checker) &&
         !AnyExecutionPathContains(ir_context, target_it, insert_before_it,
                                   checker);
}

std::vector<uint32_t> TransformationMoveInstruction::GetBlockIdsFromAllPaths(
    opt::IRContext* ir_context, uint32_t source_block_id,
    uint32_t dest_block_id) {
  // This function returns all ids of blocks from all paths from
  // |source_block_id| to |dest_block_id|. If the former and the latter are the
  // same block, there is no blocks between them.
  if (source_block_id == dest_block_id) {
    return {};
  }

  // The following implements an in-place version of DFS. The IterationState
  // struct is used to store... the state of iteration over all successors' ids
  // of some block.
  struct IterationState {
    std::vector<uint32_t> data;
    size_t index;
  };

  // A map from the block id to its successors.
  std::unordered_map<uint32_t, IterationState> successors;

  std::vector<uint32_t> stack = {source_block_id};

  // Contains all the elements from the |stack|. Used for constant-time queries.
  // It is a subset of |visited_block_ids|.
  std::unordered_set<uint32_t> on_stack = {source_block_id};

  std::unordered_set<uint32_t> visited_block_ids = {source_block_id};
  std::vector<uint32_t> result;

  while (!stack.empty()) {
    assert(stack.size() == on_stack.size() && on_stack.count(stack.back()) &&
           "Invariant: |stack| and |on_stack| have the same elements");
    assert(visited_block_ids.count(stack.back()) &&
           "Invariant: |stack| contains only visited blocks");

    auto current_block_id = stack.back();

    if (current_block_id == dest_block_id) {
      // We've reached the destination block - copy all block on the paths
      // from the source block to the destination block into a |result|.
      assert(stack.front() == source_block_id &&
             stack.back() == dest_block_id &&
             "Invariant: at least |source_block_id| and |dest_block_id|"
             "must be present on the stack");
      result.insert(result.end(), stack.begin() + 1, stack.end() - 1);

      // We still want to be able to iterate over all successors of the
      // destination block to find possible loops.
    }

    if (!successors.count(current_block_id)) {
      // We are visiting current block for the first time - insert all
      // its successors into a map.
      std::vector<uint32_t> current_successors;
      ir_context->cfg()
          ->block(current_block_id)
          ->ForEachSuccessorLabel([&current_successors](uint32_t id) {
            current_successors.push_back(id);
          });

      successors[current_block_id] = {std::move(current_successors), 0};
    }

    auto& current_successors = successors.at(current_block_id);
    const auto& data = current_successors.data;
    auto& index = current_successors.index;

    for (; index < data.size(); ++index) {
      if (on_stack.count(data[index])) {
        // The successor is an element on the stack - we've found a loop.
        // Insert all the elements from the path into a |result| vector.
        assert(stack.front() == source_block_id &&
               "Invariant: at least |source_block_id| must be present on the "
               "stack");
        result.insert(result.end(), stack.begin() + 1, stack.end());
        assert(visited_block_ids.count(data[index]) &&
               "Invariant: |on_stack| is a subset of |visited_block_ids|");
        continue;
      }

      if (!visited_block_ids.count(data[index])) {
        // We haven't visited this block yet - push it on the stack.
        visited_block_ids.insert(data[index]);
        stack.push_back(data[index]);
        on_stack.insert(data[index]);
        break;
      }
    }

    if (index == data.size()) {
      // There are no more successors of the current block - pop it off the
      // stack. We also know that we won't be visiting this block again so we
      // can remove its successors from the map.
      on_stack.erase(stack.back());
      successors.erase(stack.back());
      stack.pop_back();
    }
  }

  assert(!fuzzerutil::HasDuplicates(result) &&
         "Invariant: |result| may not have any duplicates");
  return result;
}

bool TransformationMoveInstruction::AnyExecutionPathContains(
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

  // Get all blocks on all paths from |source_block| to |dest_block|.
  // The former and the latter are not included in the result.
  auto block_ids_on_paths =
      GetBlockIdsFromAllPaths(ir_context, source_block->id(), dest_block->id());

  // If |block_ids_on_paths| is empty then either |dest_block| is an immediate
  // successor of |source_block|, they are the same or there is no path between
  // them. We've already handled the case of equal blocks above. Thus,
  // if |dest_block| is not an immediate successor, we were unable to find any
  // required instruction.
  if (block_ids_on_paths.empty() && !source_block->IsSuccessor(dest_block)) {
    return false;
  }

  // At this point we are certain that there is a path from |source_block| to
  // |dest_block|.
  assert(IteratorsAreOrderedCorrectly(source_block, source_it,
                                      source_block->end()) &&
         "|source_it| is invalid");
  assert(
      IteratorsAreOrderedCorrectly(dest_block, dest_block->begin(), dest_it) &&
      "|dest_it| is invalid");

  if (std::any_of(source_it, source_block->end(), check_with_functions) ||
      std::any_of(dest_block->begin(), dest_it, check_with_functions)) {
    return true;
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

  return (order.size() == 2 && order[0] == first && order[1] == second) ||
         (order.size() == 1 && order[0] == first && second == block->end());
}

}  // namespace fuzz
}  // namespace spvtools
