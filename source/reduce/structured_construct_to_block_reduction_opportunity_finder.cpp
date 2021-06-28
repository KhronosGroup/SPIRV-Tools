// Copyright (c) 2021 Google LLC
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

#include "source/reduce/structured_construct_to_block_reduction_opportunity_finder.h"

#include "source/reduce/structured_construct_to_block_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

std::vector<std::unique_ptr<ReductionOpportunity>>
StructuredConstructToBlockReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context, uint32_t target_function) const {
  (void)(context);
  (void)(target_function);
  std::vector<std::unique_ptr<ReductionOpportunity>> result;
  assert(false);
  return {};
}

std::string StructuredConstructToBlockReductionOpportunityFinder::GetName()
    const {
  return "StructuredConstructToBlockReductionOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
