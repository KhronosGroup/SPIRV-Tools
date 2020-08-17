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

#include "source/fuzz/fuzzer_pass_add_opphi_synonyms.h"

#include "source/fuzz/transformation_add_opphi_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddOpPhiSynonyms::FuzzerPassAddOpPhiSynonyms(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddOpPhiSynonyms::~FuzzerPassAddOpPhiSynonyms() = default;

void FuzzerPassAddOpPhiSynonyms::Apply() {}

std::vector<std::set<uint32_t>>
FuzzerPassAddOpPhiSynonyms::GetIdEquivalenceClasses() {
  std::vector<std::set<uint32_t>> id_equivalence_classes;

  // Keep track of all the ids that have already be assigned to a class.
  std::set<uint32_t> already_in_a_class;

  for (const auto& pair : GetIRContext()->get_def_use_mgr()->id_to_defs()) {
    // Exclude ids that have already been assigned to a class.
    if (already_in_a_class.count(pair.first)) {
      continue;
    }

    // Exclude irrelevant ids.
    if (GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
            pair.first)) {
      continue;
    }

    // Exclude ids having a type that is not allowed by the transformation.
    if (!TransformationAddOpPhiSynonym::CheckTypeIsAllowed(
            GetIRContext(), pair.second->type_id())) {
      continue;
    }

    // We need a new equivalence class for this id.
    std::set<uint32_t> new_equivalence_class;

    // Add this id to the class.
    new_equivalence_class.emplace(pair.first);
    already_in_a_class.emplace(pair.first);

    // Add all the synonyms with the same type to this class.
    for (auto synonym :
         GetTransformationContext()->GetFactManager()->GetSynonymsForId(
             pair.first)) {
      // The synonym must not be an indexed access into a composite.
      if (synonym->index_size() > 0) {
        continue;
      }

      // The synonym must not be irrelevant.
      if (GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
              synonym->object())) {
        continue;
      }

      auto synonym_def =
          GetIRContext()->get_def_use_mgr()->GetDef(synonym->object());
      // The synonym must exist and have the same type as the id we are
      // considering.
      if (!synonym_def || synonym_def->type_id() != pair.second->type_id()) {
        continue;
      }

      // We can add this synonym to the new equivalence class.
      new_equivalence_class.emplace(synonym->object());
      already_in_a_class.emplace(synonym->object());
    }

    // Add the new equivalence class to the list of equivalence classes.
    id_equivalence_classes.emplace_back(std::move(new_equivalence_class));
  }

  return id_equivalence_classes;
}

}  // namespace fuzz
}  // namespace spvtools
