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

namespace {

std::map<uint32_t, uint32_t> PairSequenceToMap(
    const google::protobuf::RepeatedPtrField<protobufs::UInt32Pair>&
        pair_sequence) {
  std::map<uint32_t, uint32_t> result;
  for (auto& pair : pair_sequence) {
    result[pair.first()] = pair.second();
  }
  return result;
}

}  // namespace

TransformationOutlineFunction::TransformationOutlineFunction(
    const spvtools::fuzz::protobufs::TransformationOutlineFunction& message)
    : message_(message) {}

TransformationOutlineFunction::TransformationOutlineFunction(
    uint32_t entry_block, uint32_t exit_block,
    uint32_t new_function_struct_return_type_id, uint32_t new_function_type_id,
    uint32_t new_function_id, uint32_t new_function_first_block,
    uint32_t new_function_region_entry_block, uint32_t new_caller_result_id,
    uint32_t new_callee_result_id,
    std::map<uint32_t, uint32_t>&& input_id_to_fresh_id,
    std::map<uint32_t, uint32_t>&& output_id_to_fresh_id) {
  message_.set_entry_block(entry_block);
  message_.set_exit_block(exit_block);
  message_.set_new_function_struct_return_type_id(
      new_function_struct_return_type_id);
  message_.set_new_function_type_id(new_function_type_id);
  message_.set_new_function_id(new_function_id);
  message_.set_new_function_first_block(new_function_first_block);
  message_.set_new_function_region_entry_block(new_function_region_entry_block);
  message_.set_new_caller_result_id(new_caller_result_id);
  message_.set_new_callee_result_id(new_callee_result_id);
  for (auto& entry : input_id_to_fresh_id) {
    protobufs::UInt32Pair pair;
    pair.set_first(entry.first);
    pair.set_second(entry.second);
    *message_.add_input_id_to_fresh_id() = pair;
  }
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
          message_.new_function_first_block(), context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_function_region_entry_block(), context,
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

  for (auto& pair : message_.input_id_to_fresh_id()) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            pair.second(), context, &ids_used_by_this_transformation)) {
      return false;
    }
  }

  for (auto& pair : message_.output_id_to_fresh_id()) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            pair.second(), context, &ids_used_by_this_transformation)) {
      return false;
    }
  }

  // The entry and exit block ids must indeed refer to blocks.
  for (auto block_id : {message_.entry_block(), message_.exit_block()}) {
    auto block_label = context->get_def_use_mgr()->GetDef(block_id);
    if (!block_label || block_label->opcode() != SpvOpLabel) {
      return false;
    }
  }

  auto entry_block = context->cfg()->block(message_.entry_block());
  auto exit_block = context->cfg()->block(message_.exit_block());

  // The entry block cannot start with OpVariable - this would mean that
  // outlining would remove a variable from the function containing the region
  // being outlined.
  if (entry_block->begin()->opcode() == SpvOpVariable) {
    return false;
  }

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

  // Find all the blocks dominated by |message_.entry_block| and post-dominated
  // by |message_.exit_block|.
  auto region_set = GetRegionBlocks(
      context, entry_block = context->cfg()->block(message_.entry_block()),
      exit_block = context->cfg()->block(message_.exit_block()));

  // Check whether |region_set| really is a single-entry single-exit region, and
  // also check whether structured control flow constructs and their merge
  // and continue constructs are either wholly in or wholly out of the region -
  // e.g. avoid the situation where the region contains the head of a loop but
  // not the loop's continue construct.
  //
  // This is achieved by going through every block in the function that contains
  // the region.
  for (auto& block : *entry_block->GetParent()) {
    if (&block == exit_block) {
      // It is OK (and typically expected) for the exit block of the region to
      // have successors outside the region.  It is also OK for the exit block
      // to head a structured control flow construct - the block containing the
      // call to the outlined function will end up heading this construct if
      // outlining takes place.
      continue;
    }

    if (region_set.count(&block) != 0) {
      // The block is in the region and is not the region's exit block.  Let's
      // see whether all of the block's successors are in the region.  If they
      // are not, the region is not single-entry single-exit.
      bool all_successors_in_region = true;
      block.WhileEachSuccessorLabel([&all_successors_in_region, context,
                                     &region_set](uint32_t successor) -> bool {
        if (region_set.count(context->cfg()->block(successor)) == 0) {
          all_successors_in_region = false;
          return false;
        }
        return true;
      });
      if (!all_successors_in_region) {
        return false;
      }
    }

    if (auto merge = block.GetMergeInst()) {
      // The block is a loop or selection header -- the header and its
      // associated merge block had better both be in the region or both be
      // outside the region.
      //
      // There is one exception: if the merge is the exit block for the region
      // then it doesn't matter whether the header is in the region or not;
      // if the header turns out not to be in the region, the header's merge
      // will be changed to the block containing the outlined function call.
      auto merge_block = context->cfg()->block(merge->GetSingleWordOperand(0));
      if (merge_block != exit_block &&
          region_set.count(&block) != region_set.count(merge_block)) {
        return false;
      }
    }

    if (auto loop_merge = block.GetLoopMergeInst()) {
      // Similar to the above, but for the continue target of a loop.
      auto continue_target =
          context->cfg()->block(loop_merge->GetSingleWordOperand(1));
      if (continue_target != exit_block &&
          region_set.count(&block) != region_set.count(continue_target)) {
        return false;
      }
    }
  }

  // For technical reasons it is simplest to leave OpPhi instructions that
  // appear at the start of the region's entry block where they are, rather
  // than trying to move them into the outlined function.
  //
  // Leaving them where they are would create a problem in the case where the
  // entry block is a loop header, and the OpPhi consumes an id generated by the
  // body of the loop.  We disallow this case.
  if (entry_block->GetLoopMergeInst() &&
      entry_block->begin()->opcode() == SpvOpPhi) {
    return false;
  }

  // For each region input id -- i.e. every id defined outside the region but
  // used inside the region -- there needs to be a corresponding fresh id to be
  // used as a function parameter.
  std::map<uint32_t, uint32_t> input_id_to_fresh_id_map =
      PairSequenceToMap(message_.input_id_to_fresh_id());
  for (auto id :
       GetRegionInputIds(context, region_set, entry_block, exit_block)) {
    if (input_id_to_fresh_id_map.count(id) == 0) {
      return false;
    }
  }

  // For each region output id -- i.e. every id defined inside the region but
  // used outside the region -- there needs to be a corresponding fresh id that
  // can hold the value for this id computed in the outlined function.
  std::map<uint32_t, uint32_t> output_id_to_fresh_id_map =
      PairSequenceToMap(message_.output_id_to_fresh_id());
  for (auto id :
       GetRegionOutputIds(context, region_set, entry_block, exit_block)) {
    if (output_id_to_fresh_id_map.count(id) == 0) {
      return false;
    }
  }

  return true;
}

void TransformationOutlineFunction::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* /*unused*/) const {
  // The entry block for the region before outlining.
  auto original_region_entry_block =
      context->cfg()->block(message_.entry_block());

  // The exit block for the region before outlining.
  auto original_region_exit_block =
      context->cfg()->block(message_.exit_block());

  // The single-entry single-exit region defined by |message_.entry_block| and
  // |message_.exit_block|.
  std::set<opt::BasicBlock*> region_blocks = GetRegionBlocks(
      context, original_region_entry_block, original_region_exit_block);

  // Input and output ids for the region being outlined.
  std::vector<uint32_t> region_input_ids =
      GetRegionInputIds(context, region_blocks, original_region_entry_block,
                        original_region_exit_block);
  std::vector<uint32_t> region_output_ids =
      GetRegionOutputIds(context, region_blocks, original_region_entry_block,
                         original_region_exit_block);

  // Maps from input and output ids to fresh ids.
  std::map<uint32_t, uint32_t> input_id_to_fresh_id_map =
      PairSequenceToMap(message_.input_id_to_fresh_id());
  std::map<uint32_t, uint32_t> output_id_to_fresh_id_map =
      PairSequenceToMap(message_.output_id_to_fresh_id());

  UpdateModuleIdBoundForFreshIds(context, input_id_to_fresh_id_map,
                                 output_id_to_fresh_id_map);

  std::map<uint32_t, uint32_t> output_id_to_type_id;
  for (uint32_t output_id : region_output_ids) {
    output_id_to_type_id[output_id] =
        context->get_def_use_mgr()->GetDef(output_id)->type_id();
  }

  std::unique_ptr<opt::Instruction> cloned_exit_block_merge =
      original_region_exit_block->GetMergeInst()
          ? std::unique_ptr<opt::Instruction>(
                original_region_exit_block->GetMergeInst()->Clone(context))
          : nullptr;
  std::unique_ptr<opt::Instruction> cloned_exit_block_terminator =
      std::unique_ptr<opt::Instruction>(
          original_region_exit_block->terminator()->Clone(context));
  assert(cloned_exit_block_terminator != nullptr &&
         "Every block must have a terminator.");

  std::unique_ptr<opt::Function> outlined_function =
      PrepareFunctionPrototype(context, region_input_ids, region_output_ids,
                               input_id_to_fresh_id_map, output_id_to_type_id);

  RemapInputAndOutputIdsInRegion(
      context, *original_region_entry_block, *original_region_exit_block,
      region_blocks, region_input_ids, region_output_ids,
      input_id_to_fresh_id_map, output_id_to_fresh_id_map);

  PopulateOutlinedFunction(context, *original_region_entry_block,
                           *original_region_exit_block, region_blocks,
                           region_output_ids, output_id_to_fresh_id_map,
                           outlined_function.get());

  ContractOriginalRegion(
      context, region_blocks, region_input_ids, region_output_ids,
      output_id_to_type_id, outlined_function->type_id(),
      std::move(cloned_exit_block_merge),
      std::move(cloned_exit_block_terminator), original_region_entry_block);

  context->module()->AddFunction(std::move(outlined_function));

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

std::vector<uint32_t> TransformationOutlineFunction::GetRegionInputIds(
    opt::IRContext* context, const std::set<opt::BasicBlock*>& region_set,
    opt::BasicBlock* region_entry_block, opt::BasicBlock* region_exit_block) {
  std::vector<uint32_t> result;

  region_entry_block->ForEachPhiInst(
      [context, &region_set, &result](opt::Instruction* phi_inst) {
        context->get_def_use_mgr()->WhileEachUse(
            phi_inst, [context, phi_inst, &region_set, &result](
                          opt::Instruction* use, uint32_t /*unused*/) {
              auto use_block = context->get_instr_block(use);
              if (use_block && region_set.count(use_block) != 0) {
                result.push_back(phi_inst->result_id());
                return false;
              }
              return true;
            });
      });

  region_entry_block->GetParent()->ForEachParam(
      [context, &region_set, &result](opt::Instruction* function_parameter) {
        context->get_def_use_mgr()->WhileEachUse(
            function_parameter,
            [context, function_parameter, &region_set, &result](
                opt::Instruction* use, uint32_t /*unused*/) {
              auto use_block = context->get_instr_block(use);
              if (use_block && region_set.count(use_block) != 0) {
                result.push_back(function_parameter->result_id());
                return false;
              }
              return true;
            });
      });

  for (auto& block : *region_entry_block->GetParent()) {
    if (region_set.count(&block) != 0) {
      continue;
    }
    for (auto& inst : block) {
      context->get_def_use_mgr()->WhileEachUse(
          &inst,
          [context, &inst, region_exit_block, &region_set, &result](
              opt::Instruction* use, uint32_t /*unused*/) -> bool {
            auto use_block = context->get_instr_block(use);
            // TODO comment on why we don't want to regard a use in the exit
            // block terminator as being an input id.
            if (use_block && region_set.count(use_block) != 0 &&
                !(use_block == region_exit_block && use->IsBlockTerminator())) {
              result.push_back(inst.result_id());
              return false;
            }
            return true;
          });
    }
  }
  return result;
}

std::vector<uint32_t> TransformationOutlineFunction::GetRegionOutputIds(
    opt::IRContext* context, const std::set<opt::BasicBlock*>& region_set,
    opt::BasicBlock* region_entry_block, opt::BasicBlock* region_exit_block) {
  std::vector<uint32_t> result;
  for (auto& block : *region_entry_block->GetParent()) {
    if (region_set.count(&block) != 0) {
      for (auto& inst : block) {
        context->get_def_use_mgr()->WhileEachUse(
            &inst,
            [&region_set, context, &inst, region_exit_block, &result](
                opt::Instruction* use, uint32_t /*unused*/) -> bool {
              auto use_block = context->get_instr_block(use);
              // TODO comment on why we care about region exit block - it is
              // due to returning a definition from the region requiring it to
              // be passed out.
              if (use_block && (region_set.count(use_block) == 0 ||
                                (use_block == region_exit_block &&
                                 use->IsBlockTerminator()))) {
                result.push_back(inst.result_id());
                return false;
              }
              return true;
            });
      }
    }
  }
  return result;
}

std::set<opt::BasicBlock*> TransformationOutlineFunction::GetRegionBlocks(
    opt::IRContext* context, opt::BasicBlock* entry_block,
    opt::BasicBlock* exit_block) {
  auto enclosing_function = entry_block->GetParent();
  auto dominator_analysis = context->GetDominatorAnalysis(enclosing_function);
  auto postdominator_analysis =
      context->GetPostDominatorAnalysis(enclosing_function);

  std::set<opt::BasicBlock*> result;
  for (auto& block : *enclosing_function) {
    if (dominator_analysis->Dominates(entry_block, &block) &&
        postdominator_analysis->Dominates(exit_block, &block)) {
      result.insert(&block);
    }
  }
  return result;
}

std::unique_ptr<opt::Function>
TransformationOutlineFunction::PrepareFunctionPrototype(
    opt::IRContext* context, const std::vector<uint32_t>& region_input_ids,
    const std::vector<uint32_t>& region_output_ids,
    const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
    const std::map<uint32_t, uint32_t>& output_id_to_type_id) const {
  uint32_t return_type_id = 0;
  uint32_t function_type_id = 0;

  // First, try to find an existing function type that is suitable.  This is
  // only possible if the region generates no output ids; if it generates output
  // ids we are going to make a new struct for those, and since that struct does
  // not exist there cannot already be a function type with this struct as its
  // return type.
  if (region_output_ids.empty()) {
    opt::analysis::Void void_type;
    return_type_id = context->get_type_mgr()->GetId(&void_type);
    std::vector<const opt::analysis::Type*> argument_types;
    for (auto id : region_input_ids) {
      argument_types.push_back(context->get_type_mgr()->GetType(
          context->get_def_use_mgr()->GetDef(id)->type_id()));
    }
    opt::analysis::Function function_type(&void_type, argument_types);
    function_type_id = context->get_type_mgr()->GetId(&function_type);
  }

  if (function_type_id == 0) {
    opt::Instruction::OperandList struct_member_types;
    assert(
        ((return_type_id == 0) == !region_output_ids.empty()) &&
        "We should only have set the return type if there are no output ids.");
    if (!region_output_ids.empty()) {
      for (uint32_t output_id : region_output_ids) {
        struct_member_types.push_back(
            {SPV_OPERAND_TYPE_ID, {output_id_to_type_id.at(output_id)}});
      }
      context->module()->AddType(MakeUnique<opt::Instruction>(
          context, SpvOpTypeStruct, 0,
          message_.new_function_struct_return_type_id(),
          std::move(struct_member_types)));
      return_type_id = message_.new_function_struct_return_type_id();
    }
    assert(
        return_type_id != 0 &&
        "We should either have a void return type, or have created a struct.");

    opt::Instruction::OperandList function_type_operands;
    function_type_operands.push_back({SPV_OPERAND_TYPE_ID, {return_type_id}});
    for (auto id : region_input_ids) {
      function_type_operands.push_back(
          {SPV_OPERAND_TYPE_ID,
           {context->get_def_use_mgr()->GetDef(id)->type_id()}});
    }
    context->module()->AddType(MakeUnique<opt::Instruction>(
        context, SpvOpTypeFunction, 0, message_.new_function_type_id(),
        function_type_operands));
    function_type_id = message_.new_function_type_id();
  }

  // Create a new function with |message_.new_function_id| as the function id,
  // and the return type and function type prepared above.
  std::unique_ptr<opt::Function> outlined_function =
      MakeUnique<opt::Function>(MakeUnique<opt::Instruction>(
          context, SpvOpFunction, return_type_id, message_.new_function_id(),
          opt::Instruction::OperandList(
              {{spv_operand_type_t ::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                {SpvFunctionControlMaskNone}},
               {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                {function_type_id}}})));

  for (auto id : region_input_ids) {
    outlined_function->AddParameter(MakeUnique<opt::Instruction>(
        context, SpvOpFunctionParameter,
        context->get_def_use_mgr()->GetDef(id)->type_id(),
        input_id_to_fresh_id_map.at(id), opt::Instruction::OperandList()));
  }

  return outlined_function;
}

void TransformationOutlineFunction::UpdateModuleIdBoundForFreshIds(
    opt::IRContext* context,
    const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
    const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map) const {
  // Enlarge the module's id bound as needed to accommodate the various fresh
  // ids associated with the transformation.
  fuzzerutil::UpdateModuleIdBound(
      context, message_.new_function_struct_return_type_id());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_function_type_id());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_function_id());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_function_first_block());
  fuzzerutil::UpdateModuleIdBound(context,
                                  message_.new_function_region_entry_block());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_caller_result_id());
  fuzzerutil::UpdateModuleIdBound(context, message_.new_callee_result_id());

  for (auto& entry : input_id_to_fresh_id_map) {
    fuzzerutil::UpdateModuleIdBound(context, entry.second);
  }

  for (auto& entry : output_id_to_fresh_id_map) {
    fuzzerutil::UpdateModuleIdBound(context, entry.second);
  }
}

void TransformationOutlineFunction::RemapInputAndOutputIdsInRegion(
    opt::IRContext* context, const opt::BasicBlock& original_region_entry_block,
    const opt::BasicBlock& original_region_exit_block,
    const std::set<opt::BasicBlock*>& region_blocks,
    const std::vector<uint32_t>& region_input_ids,
    const std::vector<uint32_t>& region_output_ids,
    const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
    const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map) const {
  // Change all uses of input ids inside the region to the corresponding fresh
  // ids that will ultimately be parameters of the outlined function.
  // This is done by considering each region input id in turn.
  for (uint32_t id : region_input_ids) {
    // We then consider each use of the input id.
    context->get_def_use_mgr()->ForEachUse(
        id,
        [context, &original_region_entry_block, id, &input_id_to_fresh_id_map,
         region_blocks](opt::Instruction* use, uint32_t operand_index) {
          // Find the block in which this use of the input id occurs.
          opt::BasicBlock* use_block = context->get_instr_block(use);
          // We want to rewrite the use id if its block occurs in the outlined
          // region, with one exception: if the use appears in an OpPhi in the
          // region's entry block then we leave it alone, as we will not pull
          // such OpPhi instructions into the outlined function.
          if (
              // The block is in the region ...
              region_blocks.count(use_block) != 0 &&
              // ... and the use is not in an OpPhi in the region's entry block.
              !(use->opcode() == SpvOpPhi &&
                use_block == &original_region_entry_block)) {
            // Rewrite this use of the input id.
            use->SetOperand(operand_index, {input_id_to_fresh_id_map.at(id)});
          }
        });
  }

  // Change each definition of a region output id to define the corresponding
  // fresh ids that will store intermediate value for the output ids.  Also
  // change all uses of the output id located in the outlined region.
  // This is done by considering each region output id in turn.
  for (uint32_t id : region_output_ids) {
    // First consider each use of the output id and update the relevant uses.
    context->get_def_use_mgr()->ForEachUse(
        id,
        [context, &original_region_exit_block, id, &output_id_to_fresh_id_map,
         region_blocks](opt::Instruction* use, uint32_t operand_index) {
          // Find the block in which this use of the output id occurs.
          auto use_block = context->get_instr_block(use);
          // We want to rewrite the use id if its block occurs in the outlined
          // region, with one exception: the terminator of the exit block of
          // the region is going to remain in the original function, so if the
          // use appears in such a terminator instruction we leave it alone.
          if (
              // The block is in the region ...
              region_blocks.count(use_block) != 0 &&
              // ... and the use is not in the terminator instruction of the
              // region's exit block.
              !(use_block == &original_region_exit_block &&
                use->IsBlockTerminator())) {
            // Rewrite this use of the output id.
            use->SetOperand(operand_index, {output_id_to_fresh_id_map.at(id)});
          }
        });

    // Now change the instruction that defines the output id so that it instead
    // defines the corresponding fresh id.  We do this after changing all the
    // uses so that the definition of the original id is still registered when
    // we analyse its uses.
    context->get_def_use_mgr()->GetDef(id)->SetResultId(
        output_id_to_fresh_id_map.at(id));
  }
}

void TransformationOutlineFunction::PopulateOutlinedFunction(
    opt::IRContext* context, const opt::BasicBlock& original_region_entry_block,
    const opt::BasicBlock& original_region_exit_block,
    const std::set<opt::BasicBlock*>& region_blocks,
    const std::vector<uint32_t>& region_output_ids,
    const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map,
    opt::Function* outlined_function) const {
  // TODO comment why we do this
  std::unique_ptr<opt::BasicBlock> new_function_first_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          context, SpvOpLabel, 0, message_.new_function_first_block(),
          opt::Instruction::OperandList()));
  new_function_first_block->SetParent(outlined_function);
  new_function_first_block->AddInstruction(MakeUnique<opt::Instruction>(
      context, SpvOpBranch, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID,
            {message_.new_function_region_entry_block()}}})));
  outlined_function->AddBasicBlock(std::move(new_function_first_block));

  // When we create the exit block for the outlined region, we use this pointer
  // to track of it so that we can manipulate it later.
  opt::BasicBlock* outlined_region_exit_block = nullptr;

  // The region entry block in the new function is identical to the entry block
  // of the region being outlined, except that OpPhi instructions of the
  // original block do not get outlined.
  std::unique_ptr<opt::BasicBlock> outlined_region_entry_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          context, SpvOpLabel, 0, message_.new_function_region_entry_block(),
          opt::Instruction::OperandList()));
  outlined_region_entry_block->SetParent(outlined_function);
  if (&original_region_entry_block == &original_region_exit_block) {
    outlined_region_exit_block = outlined_region_entry_block.get();
  }

  for (auto& inst : original_region_entry_block) {
    if (inst.opcode() == SpvOpPhi) {
      continue;
    }
    outlined_region_entry_block->AddInstruction(
        std::unique_ptr<opt::Instruction>(inst.Clone(context)));
  }
  outlined_function->AddBasicBlock(std::move(outlined_region_entry_block));

  // We now go through the single-entry single-exit region defined by the entry
  // and exit blocks, and clones of all such blocks to the new function.

  // Consider every block in the enclosing function.
  auto enclosing_function = original_region_entry_block.GetParent();
  for (auto block_it = enclosing_function->begin();
       block_it != enclosing_function->end();) {
    // Skip the region's entry block - we already dealt with it above.
    if (region_blocks.count(&*block_it) == 0 ||
        &*block_it == &original_region_entry_block) {
      ++block_it;
      continue;
    }
    // Clone the block so that it can be added to the new function.
    auto cloned_block =
        std::unique_ptr<opt::BasicBlock>(block_it->Clone(context));

    // If this is the region's exit block, then the cloned block is the outlined
    // region's exit block.
    if (&*block_it == &original_region_exit_block) {
      assert(outlined_region_exit_block == nullptr &&
             "We should not yet have encountered the exit block.");
      outlined_region_exit_block = cloned_block.get();
    }

    cloned_block->SetParent(outlined_function);
    // Redirect any OpPhi operands whose values are the original region entry
    // block to become the new function entry block.
    cloned_block->ForEachPhiInst([this](opt::Instruction* phi_inst) {
      for (uint32_t predecessor_index = 1;
           predecessor_index < phi_inst->NumInOperands();
           predecessor_index += 2) {
        if (phi_inst->GetSingleWordInOperand(predecessor_index) ==
            message_.entry_block()) {
          phi_inst->SetInOperand(predecessor_index,
                                 {message_.new_function_region_entry_block()});
        }
      }
    });
    switch (cloned_block->terminator()->opcode()) {
      case SpvOpBranch:
        if (cloned_block->terminator()->GetSingleWordInOperand(0) ==
            message_.entry_block()) {
          cloned_block->terminator()->SetInOperand(
              0, {message_.new_function_region_entry_block()});
        }
        break;
      case SpvOpBranchConditional:
        for (uint32_t index : {0u, 1u}) {
          if (cloned_block->terminator()->GetSingleWordInOperand(index) ==
              message_.entry_block()) {
            cloned_block->terminator()->SetInOperand(
                index, {message_.new_function_region_entry_block()});
          }
        }
        break;
      default:
        break;
    }
    outlined_function->AddBasicBlock(std::move(cloned_block));
    block_it = block_it.Erase();
  }
  assert(outlined_region_exit_block != nullptr &&
         "We should have encountered the region's exit block when iterating "
         "through the function");
  for (auto inst_it = outlined_region_exit_block->begin();
       inst_it != outlined_region_exit_block->end();) {
    if (inst_it->opcode() == SpvOpLoopMerge ||
        inst_it->opcode() == SpvOpSelectionMerge) {
      inst_it = inst_it.Erase();
    } else if (inst_it->IsBlockTerminator()) {
      inst_it = inst_it.Erase();
    } else {
      ++inst_it;
    }
  }

  if (region_output_ids.empty()) {
    outlined_region_exit_block->AddInstruction(MakeUnique<opt::Instruction>(
        context, SpvOpReturn, 0, 0, opt::Instruction::OperandList()));
  } else {
    opt::Instruction::OperandList struct_member_operands;
    for (uint32_t id : region_output_ids) {
      struct_member_operands.push_back(
          {SPV_OPERAND_TYPE_ID, {output_id_to_fresh_id_map.at(id)}});
    }
    outlined_region_exit_block->AddInstruction(MakeUnique<opt::Instruction>(
        context, SpvOpCompositeConstruct,
        message_.new_function_struct_return_type_id(),
        message_.new_callee_result_id(), struct_member_operands));
    outlined_region_exit_block->AddInstruction(MakeUnique<opt::Instruction>(
        context, SpvOpReturnValue, 0, 0,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.new_callee_result_id()}}})));
  }

  outlined_function->SetFunctionEnd(MakeUnique<opt::Instruction>(
      context, SpvOpFunctionEnd, 0, 0, opt::Instruction::OperandList()));
}

void TransformationOutlineFunction::ContractOriginalRegion(
    opt::IRContext* context, std::set<opt::BasicBlock*>& region_blocks,
    const std::vector<uint32_t>& region_input_ids,
    const std::vector<uint32_t>& region_output_ids,
    const std::map<uint32_t, uint32_t>& output_id_to_type_id,
    uint32_t return_type_id,
    std::unique_ptr<opt::Instruction> cloned_exit_block_merge,
    std::unique_ptr<opt::Instruction> cloned_exit_block_terminator,
    opt::BasicBlock* original_region_entry_block) const {
  // Consider every block in the enclosing function.
  auto enclosing_function = original_region_entry_block->GetParent();
  for (auto block_it = enclosing_function->begin();
       block_it != enclosing_function->end();) {
    if (&*block_it == original_region_entry_block) {
      ++block_it;
    } else if (region_blocks.count(&*block_it) == 0) {
      // The block is not in the region.  Check whether it uses the last block
      // of the region as a merge block continue target, or has the last block
      // of the region as an OpPhi predecessor, and change such occurrences
      // to be the first block of the region (i.e. the block containing the call
      // to what was outlined).
      if (block_it->MergeBlockIdIfAny() == message_.exit_block()) {
        block_it->GetMergeInst()->SetInOperand(0, {message_.entry_block()});
      }
      if (block_it->ContinueBlockIdIfAny() == message_.exit_block()) {
        block_it->GetMergeInst()->SetInOperand(1, {message_.entry_block()});
      }
      block_it->ForEachPhiInst([this](opt::Instruction* phi_inst) {
        for (uint32_t predecessor_index = 1;
             predecessor_index < phi_inst->NumInOperands();
             predecessor_index += 2) {
          if (phi_inst->GetSingleWordInOperand(predecessor_index) ==
              message_.exit_block()) {
            phi_inst->SetInOperand(predecessor_index, {message_.entry_block()});
          }
        }
      });
      ++block_it;
    } else {
      block_it = block_it.Erase();
    }
  }

  for (auto inst_it = original_region_entry_block->begin();
       inst_it != original_region_entry_block->end();) {
    if (inst_it->opcode() == SpvOpPhi) {
      ++inst_it;
      continue;
    }
    inst_it = inst_it.Erase();
  }
  opt::Instruction::OperandList function_call_operands;
  function_call_operands.push_back(
      {SPV_OPERAND_TYPE_ID, {message_.new_function_id()}});
  for (auto id : region_input_ids) {
    function_call_operands.push_back({SPV_OPERAND_TYPE_ID, {id}});
  }

  original_region_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
      context, SpvOpFunctionCall, return_type_id,
      message_.new_caller_result_id(), function_call_operands));

  for (uint32_t index = 0; index < region_output_ids.size(); ++index) {
    uint32_t id = region_output_ids[index];
    original_region_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
        context, SpvOpCompositeExtract, output_id_to_type_id.at(id), id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.new_caller_result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {index}}})));
  }

  if (cloned_exit_block_merge != nullptr) {
    original_region_entry_block->AddInstruction(
        std::move(cloned_exit_block_merge));
  }
  original_region_entry_block->AddInstruction(
      std::move(cloned_exit_block_terminator));
}

}  // namespace fuzz
}  // namespace spvtools
