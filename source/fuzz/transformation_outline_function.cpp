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

#include "source/fuzz/transformation_outline_function.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationOutlineFunction::TransformationOutlineFunction(
    const spvtools::fuzz::protobufs::TransformationOutlineFunction& message)
    : message_(message) {}

TransformationOutlineFunction::TransformationOutlineFunction(
    uint32_t entry_block, uint32_t exit_block, uint32_t new_function_type_id,
    uint32_t new_function_id, uint32_t new_function_entry_block,
    uint32_t new_function_exit_block, uint32_t function_call_result_id) {
  message_.set_entry_block(entry_block);
  message_.set_exit_block(exit_block);
  message_.set_new_function_type_id(new_function_type_id);
  message_.set_new_function_id(new_function_id);
  message_.set_new_function_entry_block(new_function_entry_block);
  message_.set_new_function_exit_block(new_function_exit_block);
  message_.set_function_call_result_id(function_call_result_id);
}

bool TransformationOutlineFunction::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  std::set<uint32_t> ids_used_by_this_transformation;

  // The various new ids used by the transformation must be fresh and distinct.

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_function_type_id(), context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_function_id(), context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_function_entry_block(), context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  // It is OK for the new function entry and exit blocks to be the same, so long
  // as the existing function entry and exit blocks are the same.
  if (message_.new_function_exit_block() ==
      message_.new_function_entry_block()) {
    if (message_.entry_block() != message_.exit_block()) {
      return false;
    }
  } else if (!CheckIdIsFreshAndNotUsedByThisTransformation(
                 message_.new_function_exit_block(), context,
                 &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.function_call_result_id(), context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  // TODO - check that the necessary ingredients to make the function's type
  //  are in place.

  // The entry and exit block ids must indeed refer to blocks.
  for (auto block_id : {message_.entry_block(), message_.exit_block()}) {
    auto block_label = context->get_def_use_mgr()->GetDef(block_id);
    if (!block_label || block_label->opcode() != SpvOpLabel) {
      return false;
    }
  }

  auto entry_block = context->cfg()->block(message_.entry_block());
  auto exit_block = context->cfg()->block(message_.exit_block());

  // The block must be in the same function.
  if (entry_block->GetParent() != exit_block->GetParent()) {
    return false;
  }

  // The entry block must dominate the exit block.
  auto dominator_analysis =
      context->GetDominatorAnalysis(entry_block->GetParent());
  if (!dominator_analysis->Dominates(entry_block, exit_block)) {
    return false;
  }

  // The exit block must post-dominate the entry block.
  auto postdominator_analysis =
      context->GetPostDominatorAnalysis(entry_block->GetParent());
  if (!postdominator_analysis->Dominates(exit_block, entry_block)) {
    return false;
  }

  return true;
}

void TransformationOutlineFunction::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* /*unused*/) const {
  opt::analysis::Void void_type;
  auto return_type_id = context->get_type_mgr()->GetId(&void_type);
  opt::analysis::Type* registered_void_type =
      context->get_type_mgr()->GetType(return_type_id);

  opt::analysis::Function function_type(registered_void_type, {});
  auto function_type_id = context->get_type_mgr()->GetId(&function_type);
  if (!function_type_id) {
    function_type_id = message_.new_function_type_id();
    context->module()->AddType(MakeUnique<opt::Instruction>(
        context, SpvOpTypeFunction, 0, function_type_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {return_type_id}}})));
  }

  std::unique_ptr<opt::Instruction> function_instruction =
      MakeUnique<opt::Instruction>(
          context, SpvOpFunction, return_type_id, message_.new_function_id(),
          opt::Instruction::OperandList(
              {{spv_operand_type_t ::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                {SpvFunctionControlMaskNone}},
               {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {function_type_id}}}));

  std::unique_ptr<opt::Function> outlined_function =
      MakeUnique<opt::Function>(std::move(function_instruction));

  // TODO: add parameters

  std::unique_ptr<opt::BasicBlock> new_entry_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          context, SpvOpLabel, 0, message_.new_function_entry_block(),
          opt::Instruction::OperandList()));
  new_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
      context, SpvOpReturn, 0, 0, opt::Instruction::OperandList()));
  outlined_function->AddBasicBlock(std::move(new_entry_block));
  outlined_function->SetFunctionEnd(MakeUnique<opt::Instruction>(
      context, SpvOpFunctionEnd, 0, 0, opt::Instruction::OperandList()));

  context->module()->AddFunction(std::move(outlined_function));

  auto entry_block = context->cfg()->block(message_.entry_block());

  for (auto& inst : *entry_block) {
    if (inst.opcode() != SpvOpPhi) {
      inst.InsertBefore(MakeUnique<opt::Instruction>(
          context, SpvOpFunctionCall, return_type_id,
          message_.function_call_result_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.new_function_id()}}})));
      break;
    }
  }
}

protobufs::Transformation TransformationOutlineFunction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_outline_function() = message_;
  return result;
}

bool TransformationOutlineFunction::
    CheckIdIsFreshAndNotUsedByThisTransformation(
        uint32_t id, opt::IRContext* context,
        std::set<uint32_t>* ids_used_by_this_transformation) const {
  if (!fuzzerutil::IsFreshId(context, id)) {
    return false;
  }
  if (ids_used_by_this_transformation->count(id) != 0) {
    return false;
  }
  ids_used_by_this_transformation->insert(id);
  return true;
}

}  // namespace fuzz
}  // namespace spvtools
