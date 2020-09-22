// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_merge_function_returns.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationMergeFunctionReturns::TransformationMergeFunctionReturns(
    const protobufs::TransformationMergeFunctionReturns& message)
    : message_(message) {}

TransformationMergeFunctionReturns::TransformationMergeFunctionReturns(
    uint32_t function_id, uint32_t outer_header_id, uint32_t outer_return_id,
    uint32_t return_val_id, uint32_t any_returnable_val_id,
    const std::vector<protobufs::ReturnMergingInfo>& returns_merging_info) {
  message_.set_function_id(function_id);
  message_.set_outer_header_id(outer_header_id);
  message_.set_outer_return_id(outer_return_id);
  message_.set_return_val_id(return_val_id);
  message_.set_any_returnable_val_id(any_returnable_val_id);
  for (const auto& return_merging_info : returns_merging_info) {
    *message_.add_return_merging_info() = return_merging_info;
  }
}

bool TransformationMergeFunctionReturns::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  auto function = ir_context->GetFunction(message_.function_id());
  // The function must exist.
  if (!function) {
    return false;
  }

  // The entry block must end in an unconditional branch.
  if (function->entry()->terminator()->opcode() != SpvOpBranch) {
    return false;
  }

  // If the function has a non-void return type,
  // |message_.any_returnable_val_id| must exist, have the same type as the
  // return type of the function and be available at the end of the entry block.
  auto function_type = ir_context->get_type_mgr()->GetType(function->type_id());
  assert(function_type && "The function type should always exist.");

  // Get a map from the types for which ids are available at the end of the
  // entry block to one of the ids with that type. We compute this here to avoid
  // potentially doing it multiple times later on.
  auto types_to_available_ids =
      GetTypesToIdAvailableAfterEntryBlock(ir_context);

  if (!function_type->AsVoid()) {
    auto returnable_val_def =
        ir_context->get_def_use_mgr()->GetDef(message_.any_returnable_val_id());
    if (!returnable_val_def) {
      // Check if a suitable id can be found in the module.
      if (types_to_available_ids.count(function->type_id()) == 0) {
        return false;
      }
    } else if (returnable_val_def->type_id() != function->type_id()) {
      return false;
    } else if (!fuzzerutil::IdIsAvailableBeforeInstruction(
                   ir_context, function->entry()->terminator(),
                   message_.any_returnable_val_id())) {
      // The id must be available at the end of the entry block.
      return false;
    }
  }

  // Get the reachable return blocks.
  auto return_blocks =
      fuzzerutil::GetReachableReturnBlocks(ir_context, message_.function_id());

  // Get all the merge blocks of loops containing reachable return blocks.
  std::set<uint32_t> merge_blocks;
  for (uint32_t block : return_blocks) {
    uint32_t merge_block =
        ir_context->GetStructuredCFGAnalysis()->LoopMergeBlock(block);
    while (merge_block != 0 && !merge_blocks.count(merge_block)) {
      merge_blocks.emplace(merge_block);
      merge_block =
          ir_context->GetStructuredCFGAnalysis()->LoopMergeBlock(block);
    }
  }

  // All of the relevant merge blocks must not contain instructions whose opcode
  // is not one of OpLabel, OpPhi or OpBranch.
  for (uint32_t merge_block : merge_blocks) {
    bool all_instructions_allowed =
        ir_context->get_instr_block(merge_block)
            ->WhileEachInst([](opt::Instruction* inst) {
              return inst->opcode() == SpvOpLabel ||
                     inst->opcode() == SpvOpPhi ||
                     inst->opcode() == SpvOpBranch;
            });
    if (!all_instructions_allowed) {
      return false;
    }
  }

  // The module must contain an OpConstantTrue instruction.
  if (!fuzzerutil::MaybeGetBoolConstant(ir_context, transformation_context,
                                        true, false)) {
    return false;
  }

  // The module must contain an OpConstantFalse instruction.
  if (!fuzzerutil::MaybeGetBoolConstant(ir_context, transformation_context,
                                        false, false)) {
    return false;
  }

  // Check that the fresh ids provided are fresh and distinct.
  std::set<uint32_t> used_fresh_ids;
  for (uint32_t id : {message_.outer_header_id(), message_.outer_return_id()}) {
    if (!id || !CheckIdIsFreshAndNotUsedByThisTransformation(id, ir_context,
                                                             &used_fresh_ids)) {
      return false;
    }
  }

  // Check the additional fresh id required if the function is not void.
  if (!function_type->AsVoid() &&
      (!message_.return_val_id() ||
       !CheckIdIsFreshAndNotUsedByThisTransformation(
           message_.return_val_id(), ir_context, &used_fresh_ids))) {
    return false;
  }

  auto merge_blocks_to_info = GetMappingOfMergeBlocksToInfo();

  // For each relevant merge block, check that the correct ids are available.
  for (uint32_t merge_block : merge_blocks) {
    // A map from OpPhi ids to ids of the same type available at the beginning
    // of the merge block.
    std::map<uint32_t, uint32_t> phi_to_id;

    if (merge_blocks_to_info.count(merge_block) > 0) {
      // If the map contains an entry for the merge block, check that the fresh
      // ids are fresh and distinct.
      auto info = merge_blocks_to_info[merge_block];
      if (!info.is_returning_id() ||
          !CheckIdIsFreshAndNotUsedByThisTransformation(
              info.is_returning_id(), ir_context, &used_fresh_ids)) {
        return false;
      }

      if (!function_type->AsVoid() &&
          (!info.maybe_return_val_id() ||
           !CheckIdIsFreshAndNotUsedByThisTransformation(
               info.maybe_return_val_id(), ir_context, &used_fresh_ids))) {
        return false;
      }

      // Get the mapping from OpPhis to suitable ids.
      phi_to_id = fuzzerutil::RepeatedUInt32PairToMap(
          *info.mutable_opphi_to_suitable_id());
    } else {
      // If the map does not contain an entry for the merge block, check that
      // overflow ids are available.
      if (!transformation_context.GetOverflowIdSource()->HasOverflowIds()) {
        return false;
      }
    }

    // For each OpPhi instruction, check that a suitable placeholder id is
    // available.
    bool suitable_info_for_phi =
        ir_context->get_instr_block(merge_block)
            ->WhileEachPhiInst([ir_context, &phi_to_id,
                                &types_to_available_ids](
                                   opt::Instruction* inst) {
              if (phi_to_id.count(inst->result_id()) > 0) {
                // If there exists a mapping for this instruction and the
                // placeholder id exists in the module, check that it has the
                // correct type and it is available before the instruction.
                auto placeholder_def = ir_context->get_def_use_mgr()->GetDef(
                    phi_to_id[inst->result_id()]);
                if (placeholder_def) {
                  if (inst->type_id() != placeholder_def->type_id()) {
                    return false;
                  }
                  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
                          ir_context, inst, placeholder_def->result_id())) {
                    return false;
                  }

                  return true;
                }
              }

              // If there is no mapping, check if there is a suitable id
              // available at the end of the entry block.
              return types_to_available_ids.count(inst->type_id()) > 0;
            });

    if (!suitable_info_for_phi) {
      return false;
    }
  }

  return true;
}

void TransformationMergeFunctionReturns::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationMergeFunctionReturns::ToMessage()
    const {
  return protobufs::Transformation();
}

std::map<uint32_t, protobufs::ReturnMergingInfo>
TransformationMergeFunctionReturns::GetMappingOfMergeBlocksToInfo() const {
  std::map<uint32_t, protobufs::ReturnMergingInfo> result;
  for (const auto& info : message_.return_merging_info()) {
    result.emplace(info.merge_block_id(), info);
  }
  return result;
}

std::map<uint32_t, uint32_t>
TransformationMergeFunctionReturns::GetTypesToIdAvailableAfterEntryBlock(
    opt::IRContext* ir_context) const {
  std::map<uint32_t, uint32_t> result;
  // Consider all global declarations
  for (auto& global : ir_context->module()->types_values()) {
    if (global.HasResultId() && global.type_id()) {
      result.emplace(global.type_id(), global.result_id());
    }
  }

  auto function = ir_context->GetFunction(message_.function_id());
  assert(function && "The function must exist.");

  // Consider all function parameters
  function->ForEachParam([&result](opt::Instruction* param) {
    if (param->HasResultId() && param->type_id()) {
      result.emplace(param->type_id(), param->result_id());
    }
  });

  // Consider all the instructions in the entry block.
  for (auto& inst : *function->entry()) {
    if (inst.HasResultId() && inst.type_id()) {
      result.emplace(inst.type_id(), inst.result_id());
    }
  }

  return result;
}

}  // namespace fuzz
}  // namespace spvtools
