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
#include "source/fuzz/transformation_invert_comparison_operators.h"

namespace spvtools {
namespace fuzz {

TransformationInvertComparisonOperators ::
    TransformationInvertComparisonOperators(
        protobufs::TransformationInvertComparisonOperators message)
    : message_(std::move(message)) {}

TransformationInvertComparisonOperators ::
    TransformationInvertComparisonOperators(uint32_t operator_id,
                                            uint32_t fresh_id) {
  message_.set_operator_id(operator_id);
  message_.set_fresh_id(fresh_id);
}

bool TransformationInvertComparisonOperators::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.operator_id| must be valid and inversion must be supported for
  // it.
  auto* inst = ir_context->get_def_use_mgr()->GetDef(message_.operator_id());
  if (!inst || !IsInversionSupported(inst->opcode())) {
    return false;
  }

  // Check that we can insert negation instruction.
  auto* block = ir_context->get_instr_block(inst);
  assert(block && "Instruction must have a basic block");

  auto iter = fuzzerutil::GetIteratorForInstruction(block, inst);
  ++iter;
  assert(iter != block->end() && "Instruction can't be the last in the block");
  assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLogicalNot, iter) &&
         "Can't insert negation after comparison operator");

  // |message_.fresh_id| must be fresh.
  return fuzzerutil::IsFreshId(ir_context, message_.fresh_id());
}

void TransformationInvertComparisonOperators::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* inst = ir_context->get_def_use_mgr()->GetDef(message_.operator_id());
  assert(inst);

  // Insert negation after |inst|.
  auto iter = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(inst), inst);
  ++iter;

  iter.InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, SpvOpLogicalNot, inst->type_id(), inst->result_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.fresh_id()}}}));

  inst->SetResultId(message_.fresh_id());
  inst->SetOpcode(GetInvertedInstruction(inst->opcode()));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

bool TransformationInvertComparisonOperators::IsInversionSupported(
    SpvOp opcode) {
  switch (opcode) {
    case SpvOpSGreaterThan:
    case SpvOpSGreaterThanEqual:
    case SpvOpSLessThan:
    case SpvOpSLessThanEqual:
    case SpvOpUGreaterThan:
    case SpvOpUGreaterThanEqual:
    case SpvOpULessThan:
    case SpvOpULessThanEqual:
    case SpvOpIEqual:
    case SpvOpINotEqual:
      return true;
    default:
      return false;
  }
}

SpvOp TransformationInvertComparisonOperators::GetInvertedInstruction(
    SpvOp opcode) {
  assert(IsInversionSupported(opcode) && "Inversion must be supported");

  switch (opcode) {
    case SpvOpSGreaterThan:
      return SpvOpULessThanEqual;
    case SpvOpSGreaterThanEqual:
      return SpvOpULessThan;
    case SpvOpSLessThan:
      return SpvOpUGreaterThanEqual;
    case SpvOpSLessThanEqual:
      return SpvOpSGreaterThan;
    case SpvOpUGreaterThan:
      return SpvOpULessThanEqual;
    case SpvOpUGreaterThanEqual:
      return SpvOpULessThan;
    case SpvOpULessThan:
      return SpvOpUGreaterThanEqual;
    case SpvOpULessThanEqual:
      return SpvOpUGreaterThan;
    case SpvOpIEqual:
      return SpvOpINotEqual;
    case SpvOpINotEqual:
      return SpvOpIEqual;
    default:
      // The program will fail in the debug mode because of the assertion
      // at the beginning of the function.
      return SpvOpNop;
  }
}

protobufs::Transformation TransformationInvertComparisonOperators::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_invert_comparison_operators() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
