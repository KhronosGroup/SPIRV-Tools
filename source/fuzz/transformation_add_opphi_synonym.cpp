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

#include "source/fuzz/transformation_add_opphi_synonym.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {
TransformationAddOpPhiSynonym::TransformationAddOpPhiSynonym(
    const protobufs::TransformationAddOpPhiSynonym& message)
    : message_(message) {}

TransformationAddOpPhiSynonym::TransformationAddOpPhiSynonym(
    uint32_t block_id, const std::map<uint32_t, uint32_t>& preds_to_ids,
    uint32_t fresh_id) {
  message_.set_block_id(block_id);
  *message_.mutable_pred_to_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(preds_to_ids);
  message_.set_fresh_id(fresh_id);
}

bool TransformationAddOpPhiSynonym::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Check that |message_.block_id| is a block label id.
  auto block = fuzzerutil::MaybeFindBlock(ir_context, message_.block_id());
  if (!block) {
    return false;
  }

  // Check that |message_.fresh_id| is actually fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // Check that |message_.pred_to_id| contains a mapping for all of the block's
  // predecessors.
  std::vector<uint32_t> predecessors = ir_context->cfg()->preds(block->id());

  // There must be at least one predecessor.
  if (predecessors.empty()) {
    return false;
  }

  std::map<uint32_t, uint32_t> preds_to_ids =
      fuzzerutil::RepeatedUInt32PairToMap(message_.pred_to_id());

  // There must not be repeated key values in |message_.pred_to_id|.
  if (preds_to_ids.size() != static_cast<size_t>(message_.pred_to_id_size())) {
    return false;
  }

  // Check that each predecessor has a corresponding mapping.
  for (uint32_t pred : predecessors) {
    if (preds_to_ids.count(pred) == 0) {
      return false;
    }
  }

  // Check that all the ids exist in the module.
  for (auto& pair : message_.pred_to_id()) {
    if (!ir_context->get_def_use_mgr()->GetDef(pair.second())) {
      return false;
    }
  }

  // Check that all the ids are synonymous, they all have the same type and they
  // are available to use at the end of the corresponding predecessor block.
  uint32_t first_id = preds_to_ids.begin()->second;
  uint32_t type_id = ir_context->get_def_use_mgr()->GetDef(first_id)->type_id();
  for (auto& pair : preds_to_ids) {
    // Check that the id is synonymous with the others by checking that it is
    // synonymous with the first one.
    if (!transformation_context.GetFactManager()->IsSynonymous(
            MakeDataDescriptor(pair.second, {}),
            MakeDataDescriptor(first_id, {}))) {
      return false;
    }

    // Check that the id has the same type as the other ones.
    if (ir_context->get_def_use_mgr()->GetDef(pair.second)->type_id() !=
        type_id) {
      return false;
    }

    // Check that the id is available at the end of the corresponding
    // predecessor block.

    auto pred_block = ir_context->get_instr_block(pair.first);
    // We should always be able to find the predecessor block, since it is in
    // the predecessors list of |block|.
    assert(pred_block && "Could not find one of the predecessor blocks.");

    if (!fuzzerutil::IdIsAvailableBeforeInstruction(
            ir_context, pred_block->terminator(), pair.second)) {
      return false;
    }
  }

  return true;
}

void TransformationAddOpPhiSynonym::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Get the type id from one of the ids.
  uint32_t first_id = message_.pred_to_id(0).second();
  uint32_t type_id = ir_context->get_def_use_mgr()->GetDef(first_id)->type_id();

  // Define the operand list.
  opt::Instruction::OperandList operand_list;

  // For each predecessor, add the corresponding operands.
  for (auto& pair : message_.pred_to_id()) {
    operand_list.emplace_back(
        opt::Operand{SPV_OPERAND_TYPE_ID, {pair.second()}});
    operand_list.emplace_back(
        opt::Operand{SPV_OPERAND_TYPE_ID, {pair.first()}});
  }

  // Add a new OpPhi instructions at the beginning of the block.
  ir_context->get_instr_block(message_.block_id())
      ->begin()
      .InsertBefore(MakeUnique<opt::Instruction>(ir_context, SpvOpPhi, type_id,
                                                 message_.fresh_id(),
                                                 std::move(operand_list)));

  // Update the module id bound.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Invalidate all analyses, since we added an instruction to the module.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);

  // Record the fact that the new id is synonym with the other ones by declaring
  // that it is a synonym of the first one.
  transformation_context->GetFactManager()->AddFactDataSynonym(
      MakeDataDescriptor(message_.fresh_id(), {}),
      MakeDataDescriptor(first_id, {}), ir_context);
}

protobufs::Transformation TransformationAddOpPhiSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_opphi_synonym() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
