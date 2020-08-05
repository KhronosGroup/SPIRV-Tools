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

#include "source/fuzz/decision_maker.h"

#include <functional>

namespace spvtools {
namespace fuzz {

DecisionMaker::DecisionMaker() = default;

DecisionMaker DecisionMaker::CreateInstance(uint32_t seed) {
  // We use a short variable name |r| for the result, as we use it a lot below.
  DecisionMaker r;
  r.engine_.seed(seed);

  r.fuzzer_pass_add_access_chains_chance_of_adding_access_chain_ =
      r.ChooseBetweenMinAndMax({5, 50});
  r.fuzzer_pass_add_access_chains_chance_of_going_deeper_ =
      r.ChooseBetweenMinAndMax({50, 95});

  return r;
}

DecisionMaker::~DecisionMaker() = default;

bool DecisionMaker::FuzzerPassAddAccessChainsShouldAddLoad() {
  return ChoosePercentage(
      fuzzer_pass_add_access_chains_chance_of_adding_access_chain_);
}

opt::Instruction*
DecisionMaker::FuzzerPassAddAccessChainsChoosePointerInstruction(
    const std::vector<opt::Instruction*>& pointer_instructions) {
  return pointer_instructions[RandomIndex(pointer_instructions.size())];
}

bool DecisionMaker::FuzzerPassAddAccessChainsShouldGoDeeper() {
  return ChoosePercentage(
      fuzzer_pass_add_access_chains_chance_of_going_deeper_);
}

bool DecisionMaker::FuzzerPassAddAccessChainsChooseIndexForAccessChain(
    uint32_t composite_size_bound) {
  return RandomIndex(composite_size_bound);
}

bool DecisionMaker::FuzzerPassAddAccessChainsChooseIsSigned() {
  return ChooseIsSigned();
}

// Protected member functions:

bool DecisionMaker::ChooseIsSigned() { return ChooseEven(); }

uint32_t DecisionMaker::ChooseBetweenMinAndMax(
    const std::pair<uint32_t, uint32_t>& min_max) {
  assert(min_max.first <= min_max.second);
  return min_max.first + RandomUint32(min_max.second - min_max.first + 1);
}

bool DecisionMaker::ChoosePercentage(uint32_t percentage_chance) {
  assert(percentage_chance <= 100);
  // We use 101 because we want a result in the closed interval [0, 100], and
  // RandomUint32 is not inclusive of its bound.
  return RandomUint32(101) < percentage_chance;
}

uint32_t DecisionMaker::RandomIndex(size_t size) {
  assert(size);
  return RandomUint32(static_cast<uint32_t>(size));
}

bool DecisionMaker::ChooseEven() {
  // We use 2 to get the closed interval [0, 1].
  return RandomUint32(2) == 0;
}

uint32_t DecisionMaker::RandomUint32(uint32_t bound) {
  assert(bound > 0 && "Bound must be positive");
  return static_cast<uint32_t>(
      std::uniform_int_distribution<>(0, bound - 1)(engine_));
}

}  // namespace fuzz
}  // namespace spvtools
