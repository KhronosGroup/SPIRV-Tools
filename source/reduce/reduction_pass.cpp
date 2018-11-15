// Copyright (c) 2018 Google Inc.
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

#include <algorithm>

#include "reduction_pass.h"

#include "source/opt/build_module.h"

namespace spvtools {
namespace reduce {

std::vector<uint32_t> ReductionPass::ApplyReduction(
    const std::vector<uint32_t>& binary) {
  std::unique_ptr<opt::IRContext> context =
      BuildModule(target_env_, consumer_, binary.data(), binary.size());
  assert(context);

  std::vector<std::unique_ptr<ReductionOpportunity>> opportunities =
      GetAvailableOpportunities(context.get());

  if (!is_initialized_) {
    assert(granularity_ == 0);
    assert(index_ == 0);
    is_initialized_ = true;
    index_ = 0;
    granularity_ = std::max((uint32_t)1, (uint32_t)opportunities.size());
  }

  assert(granularity_ > 0);

  if (opportunities.empty()) {
    granularity_ = 1;
    return std::vector<uint32_t>();
  }

  if (index_ >= opportunities.size()) {
    index_ = 0;
    granularity_ = std::max((uint32_t)1, granularity_ / 2);
    return std::vector<uint32_t>();
  }

  for (uint32_t i = index_;
       i < std::min(index_ + granularity_, (uint32_t)opportunities.size());
       ++i) {
    opportunities[i]->TryToApply();
  }

  index_ += granularity_;

  std::vector<uint32_t> result;
  context->module()->ToBinary(&result, false);
  return result;
}

void ReductionPass::SetMessageConsumer(MessageConsumer consumer) {
  consumer_ = std::move(consumer);
}

bool ReductionPass::ReachedMinimumGranularity() const {
  assert(granularity_ != 0);
  return granularity_ == 1;
}

}  // namespace reduce
}  // namespace spvtools
