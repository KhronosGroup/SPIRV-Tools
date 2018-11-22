// Copyright (c) 2018 Google LLC
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

#include "cut_loop_reduction_pass.h"

namespace spvtools {
namespace reduce {

using namespace opt;

std::vector<std::unique_ptr<ReductionOpportunity>>
CutLoopReductionPass::GetAvailableOpportunities(
    opt::IRContext* /*context*/) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;
  return result;
}

std::string CutLoopReductionPass::GetName() const {
  return "CutLoopReductionPass";
}

}  // namespace reduce
}  // namespace spvtools
