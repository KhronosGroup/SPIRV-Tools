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

#include "source/fuzz/fuzzer_pass_replace_irrelevant_ids.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/id_use_descriptor.h"
#include "source/fuzz/transformation_replace_irrelevant_id.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass that, for every use of an id that has been recorded as
// irrelevant, randomly decides whether to replace it with another id of the
// same type.
FuzzerPassReplaceIrrelevantIds::FuzzerPassReplaceIrrelevantIds(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassReplaceIrrelevantIds::~FuzzerPassReplaceIrrelevantIds() = default;

void FuzzerPassReplaceIrrelevantIds::Apply() {
  // Keep track of the irrelevant ids.
  std::vector<uint32_t> irrelevant_ids;

  // Keep a map from the type ids of irrelevant ids to all the ids with that
  // type.
  std::unordered_map<uint32_t, std::vector<uint32_t>> types_to_ids;

  for (auto id :
       GetTransformationContext()->GetFactManager()->GetIrrelevantIds()) {
    // Check that the id still exists in the module.
    auto declaration = GetIRContext()->get_def_use_mgr()->GetDef(id);
    if (!declaration) {
      continue;
    }

    irrelevant_ids.push_back(id);

    // If the type of this id has not been seen before, add a mapping from this
    // type id to an empty list in |types_to_ids|. The list will be filled later
    // on.
    if (types_to_ids.count(declaration->type_id()) == 0) {
      types_to_ids[declaration->type_id()] = {};
    }
  }

  // If no irrelevant ids were found, return.
  if (irrelevant_ids.empty()) {
    return;
  }

  // Loop through all the ids in the module and fill all the empty lists in
  // |types_to_ids|.
  for (const auto& pair : GetIRContext()->get_def_use_mgr()->id_to_defs()) {
    uint32_t type_id = pair.second->type_id();
    if (types_to_ids.count(type_id)) {
      types_to_ids[type_id].push_back(pair.first);
    }
  }

  // Keep a list of all the transformations to perform. We avoid applying the
  // transformations while traversing the uses since applying the transformation
  // invalidates all analyses, and we want to avoid invalidating and recomputing
  // them every time.
  std::vector<TransformationReplaceIrrelevantId> transformations_to_apply;

  // Loop through all the uses of irrelevant ids, check that the id can be
  // replaced and randomly decide whether to apply the transformation.
  for (auto irrelevant_id : irrelevant_ids) {
    uint32_t type_id =
        GetIRContext()->get_def_use_mgr()->GetDef(irrelevant_id)->type_id();

    GetIRContext()->get_def_use_mgr()->ForEachUse(
        irrelevant_id, [this, &irrelevant_id, &type_id, &types_to_ids,
                        &transformations_to_apply](opt::Instruction* use_inst,
                                                   uint32_t use_index) {
          // The id must be used as an input operand.
          if (use_index < use_inst->NumOperands() - use_inst->NumInOperands()) {
            // The id is used as an output operand, so we cannot replace this
            // usage.
            return;
          }

          // Randomly decide whether to consider this use.
          if (!GetFuzzerContext()->ChoosePercentage(
                  GetFuzzerContext()->GetChanceOfReplacingIrrelevantId())) {
            return;
          }

          // Get the input operand index for this use, from the absolute operand
          // index.
          uint32_t in_index =
              fuzzerutil::InOperandIndexFromOperandIndex(*use_inst, use_index);

          // Only go ahead if this id use can be replaced in principle.
          if (!fuzzerutil::IdUseCanBeReplaced(GetIRContext(), use_inst,
                                              in_index)) {
            return;
          }

          // Find out which ids could be used to replace this use.
          std::vector<uint32_t> available_replacement_ids;

          for (auto replacement_id : types_to_ids[type_id]) {
            // We cannot replace an id with itself.
            if (replacement_id == irrelevant_id) {
              continue;
            }

            // Only consider this replacement if it is available at the id use
            // point.
            if (fuzzerutil::IdIsAvailableAtUse(GetIRContext(), use_inst,
                                               in_index, replacement_id)) {
              available_replacement_ids.push_back(replacement_id);
            }
          }

          // Only go ahead if there is at least one id with which this use can
          // be replaced.
          if (available_replacement_ids.empty()) {
            return;
          }

          // Choose the replacement id randomly.
          uint32_t replacement_id =
              available_replacement_ids[GetFuzzerContext()->RandomIndex(
                  available_replacement_ids)];

          // Add this replacement to the list of transformations to apply.
          transformations_to_apply.emplace_back(
              TransformationReplaceIrrelevantId(
                  MakeIdUseDescriptorFromUse(GetIRContext(), use_inst,
                                             in_index),
                  replacement_id));
        });
  }

  // Apply all the transformations.
  for (const auto& transformation : transformations_to_apply) {
    ApplyTransformation(transformation);
  }
}

}  // namespace fuzz
}  // namespace spvtools
