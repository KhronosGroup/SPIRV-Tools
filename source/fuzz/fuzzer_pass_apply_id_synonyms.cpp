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

#include "source/fuzz/fuzzer_pass_apply_id_synonyms.h"

#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

FuzzerPassApplyIdSynonyms::FuzzerPassApplyIdSynonyms(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassApplyIdSynonyms::~FuzzerPassApplyIdSynonyms() = default;

void FuzzerPassApplyIdSynonyms::Apply() {
  for (auto id_with_known_synonyms :
       GetFactManager()->GetIdsForWhichSynonymsAreKnown()) {
    // A nullptr |dominator_analysis| is used to indicate that the id for which
    // synonyms are known is defined at global scope.  Otherwise
    // |dominator_analysis| provides access to dominance information for the
    // function in which this id is defined.
    opt::DominatorAnalysis* dominator_analysis = nullptr;
    auto block_containing_id =
        GetIRContext()->get_instr_block(id_with_known_synonyms);
    if (block_containing_id) {
      dominator_analysis = GetIRContext()->GetDominatorAnalysis(
          block_containing_id->GetParent());
    }

    GetIRContext()->get_def_use_mgr()->ForEachUse(
        id_with_known_synonyms,
        [this, dominator_analysis, id_with_known_synonyms](
            opt::Instruction* use_inst, uint32_t use_index) -> void {
          auto block_containing_use = GetIRContext()->get_instr_block(use_inst);
          // The use might not be in a block; e.g. it could be a decoration.
          if (!block_containing_use) {
            return;
          }
          if (!GetFuzzerContext()->ChoosePercentage(
                  GetFuzzerContext()->GetChanceOfReplacingIdWithSynonym())) {
            return;
          }
          std::vector<const protobufs::DataDescriptor*> synonyms_to_try;
          for (auto& data_descriptor :
               GetFactManager()->GetSynonymsForId(id_with_known_synonyms)) {
            synonyms_to_try.push_back(&data_descriptor);
          }
          while (!synonyms_to_try.empty()) {
            auto synonym_index =
                GetFuzzerContext()->RandomIndex(synonyms_to_try);
            auto synonym_to_try = synonyms_to_try[synonym_index];
            synonyms_to_try.erase(synonyms_to_try.begin() + synonym_index);
            assert(synonym_to_try->index().size() == 0 &&
                   "Right now we only support id == id synonyms; supporting "
                   "e.g. id == index-into-vector will come later");
            auto inst_defining_synonym =
                GetIRContext()->get_def_use_mgr()->GetDef(
                    synonym_to_try->object());
            if (!dominator_analysis || dominator_analysis->Dominates(
                                           inst_defining_synonym, use_inst)) {
              assert(0);
            }
            (void)(use_index);
          }
        });
    (void)(id_with_known_synonyms);
  }
}

}  // namespace fuzz
}  // namespace spvtools
