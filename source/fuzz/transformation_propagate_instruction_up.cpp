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

#include "source/fuzz/transformation_propagate_instruction_up.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {
namespace {

std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>
ComputeMappingFromOpPhiToResultId(opt::IRContext* ir_context,
                                  opt::Instruction* inst) {
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> result;

  const auto* inst_block = ir_context->get_instr_block(inst);

  for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
    const auto& operand = inst->GetInOperand(i);
    if (operand.type != SPV_OPERAND_TYPE_ID) {
      continue;
    }

    auto* dependency = ir_context->get_def_use_mgr()->GetDef(operand.words[0]);
    assert(dependency && "|inst| depends on invalid id");

    if (ir_context->get_instr_block(dependency) == inst_block &&
        dependency->opcode() == SpvOpPhi &&
        !result.count(dependency->result_id())) {
      std::unordered_map<uint32_t, uint32_t> label_id_to_result_id;

      for (uint32_t j = 1; j < dependency->NumInOperands(); j += 2) {
        label_id_to_result_id[dependency->GetSingleWordInOperand(j)] =
            dependency->GetSingleWordInOperand(j - 1);
      }

      result[dependency->result_id()] = std::move(label_id_to_result_id);
    }
  }

  return result;
}

}  // namespace

TransformationPropagateInstructionUp::TransformationPropagateInstructionUp(
    const protobufs::TransformationPropagateInstructionUp& message)
    : message_(message) {}

TransformationPropagateInstructionUp::TransformationPropagateInstructionUp(
    uint32_t block_id,
    const std::map<uint32_t, uint32_t>& predecessor_id_to_fresh_id) {
  message_.set_block_id(block_id);
  *message_.mutable_predecessor_id_to_fresh_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(predecessor_id_to_fresh_id);
}

bool TransformationPropagateInstructionUp::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that we can apply this transformation to the |block_id|.
  if (!IsApplicableToTheBlock(ir_context, message_.block_id())) {
    return false;
  }

  const auto predecessor_id_to_fresh_id = fuzzerutil::RepeatedUInt32PairToMap(
      message_.predecessor_id_to_fresh_id());
  std::vector<uint32_t> maybe_fresh_ids;
  for (auto id : ir_context->cfg()->preds(message_.block_id())) {
    // Each predecessor must have a fresh id in the |predecessor_id_to_fresh_id|
    // map.
    if (!predecessor_id_to_fresh_id.count(id)) {
      return false;
    }

    maybe_fresh_ids.push_back(predecessor_id_to_fresh_id.at(id));
  }

  // All ids must be unique and fresh.
  return !fuzzerutil::HasDuplicates(maybe_fresh_ids) &&
         std::all_of(maybe_fresh_ids.begin(), maybe_fresh_ids.end(),
                     [ir_context](uint32_t id) {
                       return fuzzerutil::IsFreshId(ir_context, id);
                     });
}

void TransformationPropagateInstructionUp::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* inst = GetInstructionToPropagate(ir_context, message_.block_id());
  assert(inst &&
         "The block must have at least one supported instruction to propagate");

  // |inst| might depend on OpPhi instructions from the same basic block.
  // |op_phi_to_result_id| contains a mapping from the result id of such an
  // OpPhi instruction to the map of its operands
  // (i.e. |op_phi_to_result_id[op_phi_id][label_id] == result_id|).
  const auto op_phi_to_result_id =
      ComputeMappingFromOpPhiToResultId(ir_context, inst);

  opt::Instruction::OperandList op_phi_operands;
  const auto predecessor_id_to_fresh_id = fuzzerutil::RepeatedUInt32PairToMap(
      message_.predecessor_id_to_fresh_id());
  for (auto predecessor_id : ir_context->cfg()->preds(message_.block_id())) {
    auto new_result_id = predecessor_id_to_fresh_id.at(predecessor_id);

    // Compute InOperands for the OpPhi instruction to be inserted later.
    op_phi_operands.push_back({SPV_OPERAND_TYPE_ID, {new_result_id}});
    op_phi_operands.push_back({SPV_OPERAND_TYPE_ID, {predecessor_id}});

    // Create a clone of the |inst| to be inserted into the |predecessor_id|.
    std::unique_ptr<opt::Instruction> clone(inst->Clone(ir_context));
    clone->SetResultId(new_result_id);

    fuzzerutil::UpdateModuleIdBound(ir_context, new_result_id);

    // Adjust |clone|'s operands to account for possible dependencies on OpPhi
    // instructions from the same basic block.
    for (uint32_t i = 0; i < clone->NumInOperands(); ++i) {
      auto& operand = clone->GetInOperand(i);

      if (operand.type == SPV_OPERAND_TYPE_ID &&
          op_phi_to_result_id.count(operand.words[0])) {
        operand.words[0] =
            op_phi_to_result_id.at(operand.words[0]).at(predecessor_id);
      }
    }

    // Insert cloned instruction into the predecessor.
    auto* predecessor = ir_context->cfg()->block(predecessor_id);
    assert(predecessor && "|predecessor_id| is invalid");

    GetLastInsertBeforeInstruction(predecessor, clone->opcode())
        ->InsertBefore(std::move(clone));
  }

  // Insert an OpPhi instruction into the basic block of |inst|.
  ir_context->get_instr_block(inst)->begin()->InsertBefore(
      MakeUnique<opt::Instruction>(ir_context, SpvOpPhi, inst->type_id(),
                                   inst->result_id(),
                                   std::move(op_phi_operands)));

  // Remove |inst| from the basic block.
  inst->RemoveFromList();
  delete inst;  // RemoveFromList doesn't clear the memory.

  // Make sure our changes are analyzed
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);

  // Remove unused OpPhi instructions.
  for (const auto& entry : op_phi_to_result_id) {
    auto op_phi_result_id = entry.first;

    if (ir_context->get_def_use_mgr()->NumUsers(op_phi_result_id) == 0) {
      auto* unused_op_phi_inst =
          ir_context->get_def_use_mgr()->GetDef(op_phi_result_id);
      assert(unused_op_phi_inst && "|op_phi_result_id| must be valid");

      unused_op_phi_inst->RemoveFromList();
      delete unused_op_phi_inst;
    }
  }

  // Make sure our changes are analyzed after we've removed unused OpPhi
  // instructions.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationPropagateInstructionUp::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_propagate_instruction_up() = message_;
  return result;
}

bool TransformationPropagateInstructionUp::IsOpcodeSupported(SpvOp opcode) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3605):
  //  We only support "simple" instructions that don't work with memory.
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

opt::Instruction*
TransformationPropagateInstructionUp::GetInstructionToPropagate(
    opt::IRContext* ir_context, uint32_t block_id) {
  auto* block = ir_context->cfg()->block(block_id);
  assert(block && "|block_id| is invalid");

  for (auto& inst : *block) {
    if (inst.opcode() == SpvOpPhi || !IsOpcodeSupported(inst.opcode())) {
      continue;
    }

    auto valid = true;
    for (uint32_t i = 0; i < inst.NumInOperands(); ++i) {
      const auto& operand = inst.GetInOperand(i);
      if (operand.type != SPV_OPERAND_TYPE_ID) {
        continue;
      }

      auto* dependency =
          ir_context->get_def_use_mgr()->GetDef(operand.words[0]);
      assert(dependency && "Operand has invalid id");

      if (ir_context->get_instr_block(dependency) == block &&
          dependency->opcode() != SpvOpPhi) {
        valid = false;
        break;
      }
    }

    if (valid) {
      return &inst;
    }
  }

  return nullptr;
}

bool TransformationPropagateInstructionUp::IsApplicableToTheBlock(
    opt::IRContext* ir_context, uint32_t block_id) {
  // Check that |block_id| is valid.
  const auto* block = ir_context->cfg()->block(block_id);
  if (!block) {
    return false;
  }

  // Check that |block| has predecessors.
  if (ir_context->cfg()->preds(block_id).empty()) {
    return false;
  }

  // The block must contain an instruction to propagate.
  const auto* inst_to_propagate =
      GetInstructionToPropagate(ir_context, block_id);
  if (!inst_to_propagate) {
    return false;
  }

  // We should be able to insert |inst_to_propagate| into every predecessor of
  // |block|.
  for (auto id : ir_context->cfg()->preds(block_id)) {
    auto* predecessor = ir_context->cfg()->block(id);
    assert(predecessor && "Predecessor id is invalid");

    if (!GetLastInsertBeforeInstruction(predecessor,
                                        inst_to_propagate->opcode())) {
      return false;
    }
  }

  return true;
}

opt::Instruction*
TransformationPropagateInstructionUp::GetLastInsertBeforeInstruction(
    opt::BasicBlock* block, SpvOp opcode) {
  auto it = block->rbegin();

  while (it != block->rend() &&
         !fuzzerutil::CanInsertOpcodeBeforeInstruction(opcode, &*it)) {
    --it;
  }

  return it == block->rend() ? nullptr : &*it;
}

}  // namespace fuzz
}  // namespace spvtools
