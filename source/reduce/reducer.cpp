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

#include <cassert>
#include <sstream>

#include "source/spirv_reducer_options.h"

#include "reducer.h"
#include "reduction_pass.h"

namespace spvtools {
namespace reduce {

struct Reducer::Impl {
  explicit Impl(spv_target_env env)
      : target_env(env), made_progress_this_round(false) {}

  const spv_target_env target_env;  // Target environment.
  MessageConsumer consumer;         // Message consumer.
  InterestingFunction is_interesting;
  std::vector<std::unique_ptr<ReductionPass>> passes;
  std::vector<std::unique_ptr<ReductionPass>>::iterator current_pass;
  bool made_progress_this_round;

  std::vector<uint32_t> ApplyReduction(const std::vector<uint32_t>& binary);
  bool Finished();
};

Reducer::Reducer(spv_target_env env) : impl_(MakeUnique<Impl>(env)) {}

Reducer::~Reducer() = default;

void Reducer::SetMessageConsumer(MessageConsumer c) {
  for (auto& pass : impl_->passes) {
    pass->SetMessageConsumer(c);
  }
  impl_->consumer = std::move(c);
}

void Reducer::SetInterestingFunction(
    Reducer::InterestingFunction interestingFunction) {
  impl_->is_interesting = std::move(interestingFunction);
}

bool Reducer::Run(std::vector<uint32_t>&& binary_in,
                  std::vector<uint32_t>& binary_out,
                  spv_const_reducer_options options) const {
  impl_->current_pass = impl_->passes.begin();

  uint32_t current_step = 0;

  std::vector<uint32_t> current = std::move(binary_in);

  // Initial state should be interesting.
  assert(impl_->is_interesting(current));

  for (;;) {
    assert(current_step < options->step_limit);

    {
      std::string msg = "Reduction step ";
      msg += std::to_string(current_step);
      impl_->consumer(SPV_MSG_INFO, nullptr, {}, msg.c_str());
    }

    std::vector<uint32_t> reduction_step_result =
        impl_->ApplyReduction(current);

    if (reduction_step_result.empty()) {
      impl_->consumer(SPV_MSG_INFO, nullptr, {},
                      "No more to reduce; stopping.");
      break;
    }

    if (impl_->is_interesting(reduction_step_result)) {
      // Interesting:
      impl_->made_progress_this_round = true;
      impl_->consumer(SPV_MSG_INFO, nullptr, {}, "Reduction step succeeded.");
      current = std::move(reduction_step_result);
    } else {
      impl_->consumer(SPV_MSG_INFO, nullptr, {}, "Reduction step failed.");
    }

    ++current_step;
    if (current_step == options->step_limit) {
      break;
    }
  }

  binary_out = std::move(current);

  return true;
}

void Reducer::AddReductionPass(
    std::unique_ptr<ReductionPass>&& reduction_pass) {
  impl_->passes.push_back(std::move(reduction_pass));
}

std::vector<uint32_t> Reducer::Impl::ApplyReduction(
    const std::vector<uint32_t>& binary) {
  consumer(SPV_MSG_INFO, nullptr, {}, "Applying a reduction step.");
  for (;;) {
    assert(current_pass != passes.end());
    while (current_pass != passes.end()) {
      consumer(SPV_MSG_INFO, nullptr, {},
               ("Trying pass " + (*current_pass)->GetName() + ".").c_str());
      auto maybe_result = (*current_pass)->ApplyReduction(binary);
      if (!maybe_result.empty()) {
        consumer(
            SPV_MSG_INFO, nullptr, {},
            ("Pass " + (*current_pass)->GetName() + " made a reduction step.")
                .c_str());
        return maybe_result;
      }
      consumer(
          SPV_MSG_INFO, nullptr, {},
          ("Pass " + (*current_pass)->GetName() + " made no reduction step.")
              .c_str());
      ++current_pass;
    }
    consumer(SPV_MSG_INFO, nullptr, {}, "Completed a round of passes.");
    if (Finished()) {
      consumer(SPV_MSG_INFO, nullptr, {},
               "No reduction step could be applied.");
      return std::vector<uint32_t>();
    }
    made_progress_this_round = false;
    current_pass = passes.begin();
  }
}

bool Reducer::Impl::Finished() {
  if (made_progress_this_round) {
    consumer(SPV_MSG_INFO, nullptr, {},
             "Not finished because some pass made progress last round.");
    return false;
  }
  for (auto& pass : passes) {
    if (!pass->ReachedMinimumGranularity()) {
      consumer(SPV_MSG_INFO, nullptr, {},
               ("Not finished because pass " + pass->GetName() +
                " can be applied with finer granularity.")
                   .c_str());
      return false;
    }
  }
  return true;
}

}  // namespace reduce
}  // namespace spvtools