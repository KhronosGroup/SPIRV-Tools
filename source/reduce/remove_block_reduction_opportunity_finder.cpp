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

#include "source/reduce/remove_block_reduction_opportunity_finder.h"
#include "source/reduce/remove_block_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

using namespace opt;

std::string RemoveBlockReductionOpportunityFinder::GetName() const {
  return "RemoveBlockReductionOpportunityFinder";
}

std::vector<std::unique_ptr<ReductionOpportunity>>
RemoveBlockReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  // Consider every block in every function.
  for (auto& function : *context->module()) {

    // Skip first block; we don't want to end up with no blocks.
    auto bi = function.begin();
    if (bi != function.end()) {
      ++bi;
      for (; bi != function.end(); ++bi) {
        if (context->get_def_use_mgr()->NumUsers(bi->id()) == 0) {
          result.push_back(
              spvtools::MakeUnique<RemoveBlockReductionOpportunity>(&function,
                                                                    &*bi));
        }
      }
    }
  }
  return result;
}

}  // namespace reduce
}  // namespace spvtools
