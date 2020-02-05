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

#include "source/fuzz/fuzzer_pass_add_global_variables.h"

#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_type_pointer.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddGlobalVariables::FuzzerPassAddGlobalVariables(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassAddGlobalVariables::~FuzzerPassAddGlobalVariables() = default;

void FuzzerPassAddGlobalVariables::Apply() {
  auto base_type_ids_and_pointers =
      GetAvailableBaseTypesAndPointers(SpvStorageClassPrivate);
  auto& base_type_ids = base_type_ids_and_pointers.first;
  auto& base_type_to_pointer = base_type_ids_and_pointers.second;

  while (GetFuzzerContext()->ChoosePercentage(
      GetFuzzerContext()->GetChanceOfAddingGlobalVariable())) {
    uint32_t base_type_id =
        base_type_ids[GetFuzzerContext()->RandomIndex(base_type_ids)];
    uint32_t pointer_type_id;
    std::vector<uint32_t>& available_pointers =
        base_type_to_pointer.at(base_type_id);
    if (available_pointers.empty()) {
      pointer_type_id = GetFuzzerContext()->GetFreshId();
      available_pointers.push_back(pointer_type_id);
      ApplyTransformation(TransformationAddTypePointer(
          pointer_type_id, SpvStorageClassPrivate, base_type_id));
    } else {
      pointer_type_id = available_pointers[GetFuzzerContext()->RandomIndex(
          available_pointers)];
    }
    ApplyTransformation(TransformationAddGlobalVariable(
        GetFuzzerContext()->GetFreshId(), pointer_type_id,
        FindOrCreateZeroConstant(base_type_id), true));
  }
}

}  // namespace fuzz
}  // namespace spvtools
