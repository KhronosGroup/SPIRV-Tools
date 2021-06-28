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

#ifndef SOURCE_REDUCE_STRUCTURED_CONSTRUCT_TO_BLOCK_REDUCTION_OPPORTUNITY_FINDER_H
#define SOURCE_REDUCE_STRUCTURED_CONSTRUCT_TO_BLOCK_REDUCTION_OPPORTUNITY_FINDER_H

#include "source/reduce/reduction_opportunity_finder.h"

namespace spvtools {
namespace reduce {

// A finder for opportunities to replace a skeletal structured control flow
// construct - that is, a construct that does not define anything that's used
// outside the construct - into its header block.
class StructuredConstructToBlockReductionOpportunityFinder
    : public ReductionOpportunityFinder {
 public:
  StructuredConstructToBlockReductionOpportunityFinder() = default;

  ~StructuredConstructToBlockReductionOpportunityFinder() override = default;

  std::string GetName() const final;

  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      opt::IRContext* context, uint32_t target_function) const final;

 private:
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_STRUCTURED_CONSTRUCT_TO_BLOCK_REDUCTION_OPPORTUNITY_FINDER_H
