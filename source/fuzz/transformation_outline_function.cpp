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

#include <set>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationOutlineFunction::TransformationOutlineFunction(
    const spvtools::fuzz::protobufs::TransformationOutlineFunction& message)
    : message_(message) {}

TransformationOutlineFunction::TransformationOutlineFunction(
    uint32_t entry_block, uint32_t exit_block,
    uint32_t new_function_struct_return_type_id, uint32_t new_function_type_id,
    uint32_t new_function_id, uint32_t new_function_entry_block,
    uint32_t new_caller_result_id, uint32_t new_callee_result_id,
    std::map<uint32_t, uint32_t>&& output_id_to_fresh_id) {
  message_.set_entry_block(entry_block);
  message_.set_exit_block(exit_block);
  message_.set_new_function_struct_return_type_id(
      new_function_struct_return_type_id);
  message_.set_new_function_type_id(new_function_type_id);
  message_.set_new_function_id(new_function_id);
  message_.set_new_function_entry_block(new_function_entry_block);
  message_.set_new_caller_result_id(new_caller_result_id);
  message_.set_new_callee_result_id(new_callee_result_id);
  for (auto& entry : output_id_to_fresh_id) {
    protobufs::UInt32Pair pair;
    pair.set_first(entry.first);
    pair.set_second(entry.second);
    *message_.add_output_id_to_fresh_id() = pair;
  }
}

bool TransformationOutlineFunction::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  std::set<uint32_t> ids_used_by_this_transformation;

  // The various new ids used by the transformation must be fresh and distinct.

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_function_struct_return_type_id(), context,
          &ids_used_by_this_transformation)) {
    return false;
  }

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

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_caller_result_id(), context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_callee_result_id(), context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  for (auto pair : message_.output_id_to_fresh_id()) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            pair.second(), context, &ids_used_by_this_transformation)) {
      return false;
    }
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

  std::map<uint32_t, uint32_t> output_id_to_fresh_id_map =
      GetOutputIdToFreshIdMap();
  std::vector<uint32_t> ids_defined_in_region_and_used_outside_region =
      GetIdsDefinedInRegionAndUsedOutsideRegion(context);
  for (auto id : ids_defined_in_region_and_used_outside_region) {
    if (output_id_to_fresh_id_map.count(id) == 0) {
      return false;
    }
  }

  return true;
}

void TransformationOutlineFunction::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* /*unused*/) const {
  auto def_use_manager_before_changes = context->get_def_use_mgr();

  fuzzerutil::UpdateModuleIdBound(
      context, message_.new_function_struct_return_type_id());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_function_type_id());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_function_id());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_function_entry_block());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_caller_result_id());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_callee_result_id());

  std::map<uint32_t, uint32_t> output_id_to_fresh_id_map =
      GetOutputIdToFreshIdMap();
  std::vector<uint32_t> ids_defined_in_region_and_used_outside_region =
      GetIdsDefinedInRegionAndUsedOutsideRegion(context);

  for (uint32_t id : ids_defined_in_region_and_used_outside_region) {
    def_use_manager_before_changes->GetDef(id)->SetResultId(
        output_id_to_fresh_id_map[id]);
  }

  for (auto& entry : output_id_to_fresh_id_map) {
    fuzzerutil::UpdateModuleIdBound(context, entry.second);
  }

  std::map<uint32_t, uint32_t> output_id_to_type_id;

  uint32_t return_type_id;
  uint32_t function_type;
  if (message_.output_id_to_fresh_id().empty()) {
    opt::analysis::Void void_type;
    return_type_id = context->get_type_mgr()->GetId(&void_type);
    opt::analysis::Function void_function_type(&void_type, {});
    function_type = context->get_type_mgr()->GetId(&void_function_type);
  } else {
    opt::Instruction::OperandList struct_member_types;
    for (uint32_t output_id : ids_defined_in_region_and_used_outside_region) {
      auto type_id =
          def_use_manager_before_changes->GetDef(output_id)->type_id();
      struct_member_types.push_back({SPV_OPERAND_TYPE_ID, {type_id}});
      output_id_to_type_id[output_id] = type_id;
    }
    context->module()->AddType(MakeUnique<opt::Instruction>(
        context, SpvOpTypeStruct, 0,
        message_.new_function_struct_return_type_id(),
        std::move(struct_member_types)));
    return_type_id = message_.new_function_struct_return_type_id();
    context->module()->AddType(MakeUnique<opt::Instruction>(
        context, SpvOpTypeFunction, 0, message_.new_function_type_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {return_type_id}}})));
    function_type = message_.new_function_type_id();
  }

  // Create a new function with |message_.new_function_id| as the function id,
  // and the return type and function type prepared above.
  std::unique_ptr<opt::Function> outlined_function =
      MakeUnique<opt::Function>(MakeUnique<opt::Instruction>(
          context, SpvOpFunction, return_type_id, message_.new_function_id(),
          opt::Instruction::OperandList(
              {{spv_operand_type_t ::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                {SpvFunctionControlMaskNone}},
               {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {function_type}}})));

  // TODO: add parameters

  // The entry block of the new function is identical to the entry block of the
  // region being outlined, except that OpPhi instructions of the original block
  // do not get outlined.
  std::unique_ptr<opt::BasicBlock> new_entry_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          context, SpvOpLabel, 0, message_.new_function_entry_block(),
          opt::Instruction::OperandList()));
  auto entry_block = context->cfg()->block(message_.entry_block());
  for (auto& inst : *entry_block) {
    if (inst.opcode() == SpvOpPhi) {
      continue;
    }
    new_entry_block->AddInstruction(
        std::unique_ptr<opt::Instruction>(inst.Clone(context)));
  }
  outlined_function->AddBasicBlock(std::move(new_entry_block));

  // We now go through the single-entry single-exit region defined by the entry
  // and exit blocks, and clones of all such blocks to the new function.
  auto enclosing_function = entry_block->GetParent();
  auto dominator_analysis = context->GetDominatorAnalysis(enclosing_function);
  auto exit_block = context->cfg()->block(message_.exit_block());
  auto postdominator_analysis =
      context->GetPostDominatorAnalysis(enclosing_function);

  // Consider every block in the enclosing function.
  for (auto block_it = enclosing_function->begin();
       block_it != enclosing_function->end();) {
    // Skip the region's entry block - we already dealt with it above.
    if (&*block_it == entry_block) {
      ++block_it;
      continue;
    }
    // Skip any blocks that are not in the region.
    if (!(dominator_analysis->Dominates(entry_block, &*block_it) &&
          postdominator_analysis->Dominates(exit_block, &*block_it))) {
      ++block_it;
      continue;
    }
    // Add the block to the new function.
    outlined_function->AddBasicBlock(
        std::unique_ptr<opt::BasicBlock>(block_it->Clone(context)));
    block_it = block_it.Erase();
  }
  auto final_block = --outlined_function->end();
  std::unique_ptr<opt::Instruction> cloned_merge = nullptr;
  if (final_block->GetMergeInst()) {
    cloned_merge = std::unique_ptr<opt::Instruction>(
        final_block->GetMergeInst()->Clone(context));
    final_block->GetMergeInst()->RemoveFromList();
  }
  std::unique_ptr<opt::Instruction> cloned_terminator =
      std::unique_ptr<opt::Instruction>(
          final_block->terminator()->Clone(context));
  final_block->terminator()->RemoveFromList();

  if (ids_defined_in_region_and_used_outside_region.empty()) {
    final_block->AddInstruction(MakeUnique<opt::Instruction>(
        context, SpvOpReturn, 0, 0, opt::Instruction::OperandList()));
  } else {
    opt::Instruction::OperandList struct_member_operands;
    for (uint32_t id : ids_defined_in_region_and_used_outside_region) {
      struct_member_operands.push_back(
          {SPV_OPERAND_TYPE_ID, {output_id_to_fresh_id_map[id]}});
    }
    final_block->AddInstruction(MakeUnique<opt::Instruction>(
        context, SpvOpCompositeConstruct,
        message_.new_function_struct_return_type_id(),
        message_.new_callee_result_id(), struct_member_operands));
    final_block->AddInstruction(MakeUnique<opt::Instruction>(
        context, SpvOpReturnValue, 0, 0,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.new_callee_result_id()}}})));
  }

  outlined_function->SetFunctionEnd(MakeUnique<opt::Instruction>(
      context, SpvOpFunctionEnd, 0, 0, opt::Instruction::OperandList()));

  context->module()->AddFunction(std::move(outlined_function));

  for (auto inst_it = entry_block->begin(); inst_it != entry_block->end();) {
    if (inst_it->opcode() == SpvOpPhi) {
      ++inst_it;
      continue;
    }
    inst_it = inst_it.Erase();
  }
  entry_block->AddInstruction(MakeUnique<opt::Instruction>(
      context, SpvOpFunctionCall, return_type_id,
      message_.new_caller_result_id(),
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {message_.new_function_id()}}})));

  for (uint32_t index = 0;
       index < ids_defined_in_region_and_used_outside_region.size(); ++index) {
    uint32_t id = ids_defined_in_region_and_used_outside_region[index];
    entry_block->AddInstruction(MakeUnique<opt::Instruction>(
        context, SpvOpCompositeExtract, output_id_to_type_id[id], id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.new_caller_result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {index}}})));
  }

  if (cloned_merge != nullptr) {
    entry_block->AddInstruction(std::move(cloned_merge));
  }
  entry_block->AddInstruction(std::move(cloned_terminator));

  context->InvalidateAnalysesExceptFor(opt::IRContext::Analysis::kAnalysisNone);
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

std::vector<uint32_t>
TransformationOutlineFunction::GetIdsDefinedInRegionAndUsedOutsideRegion(
    opt::IRContext* context) const {
  std::set<opt::BasicBlock*> blocks_in_region;
  std::vector<opt::BasicBlock*> list_of_blocks_in_region;

  auto entry_block = context->cfg()->block(message_.entry_block());
  auto exit_block = context->cfg()->block(message_.exit_block());

  auto enclosing_function = entry_block->GetParent();
  auto dominator_analysis = context->GetDominatorAnalysis(enclosing_function);
  auto postdominator_analysis =
      context->GetPostDominatorAnalysis(enclosing_function);

  for (auto& block : *enclosing_function) {
    if (dominator_analysis->Dominates(entry_block, &block) &&
        postdominator_analysis->Dominates(exit_block, &block)) {
      blocks_in_region.insert(&block);
      list_of_blocks_in_region.push_back(&block);
    }
  }

  std::vector<uint32_t> result;

  for (auto block : list_of_blocks_in_region) {
    for (auto& inst : *block) {
      context->get_def_use_mgr()->WhileEachUse(
          &inst,
          [&blocks_in_region, context, &inst, &result](
              opt::Instruction* use, uint32_t /*unused*/) -> bool {
            auto use_block = context->get_instr_block(use);
            if (use_block && blocks_in_region.count(use_block) == 0) {
              result.push_back(inst.result_id());
              return false;
            }
            return true;
          });
    }
  }

  return result;
}

std::map<uint32_t, uint32_t>
TransformationOutlineFunction::GetOutputIdToFreshIdMap() const {
  std::map<uint32_t, uint32_t> result;
  for (auto& pair : message_.output_id_to_fresh_id()) {
    result[pair.first()] = pair.second();
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
