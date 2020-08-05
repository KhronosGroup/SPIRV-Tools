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

#ifndef SOURCE_FUZZ_DECISION_MAKER_H_
#define SOURCE_FUZZ_DECISION_MAKER_H_

#include <random>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/function.h"

namespace spvtools {
namespace fuzz {

// Resolves all random decisions that need to be made by fuzzer passes.
// The current interface of the class (when including private members) does not
// hide/abstract the random number generator |engine_| nor the probability
// fields such as
// |fuzzer_pass_add_access_chains_chance_of_adding_access_chain_|, despite the
// fact that these details might not always be needed (e.g. when testing or
// when a different "decision strategy" is desired). Thus, this interface may
// change in the future. However, the public member functions are assumed to be
// fairly stable, so should be used by fuzzer passes and can be overridden for
// testing purposes.
//
// Use |DecisionMaker::CreateInstance(seed)| to get an instance.
//
class DecisionMaker {
 protected:
  DecisionMaker();

 public:
  virtual ~DecisionMaker();

  static DecisionMaker CreateInstance(uint32_t seed);

 public:
  //
  // FuzzerPassAddAccessChains.
  //

  virtual bool FuzzerPassAddAccessChainsShouldAddLoad();

  virtual opt::Instruction* FuzzerPassAddAccessChainsChoosePointerInstruction(
      const std::vector<opt::Instruction*>& pointer_instructions);

  virtual bool FuzzerPassAddAccessChainsShouldGoDeeper();

  virtual bool FuzzerPassAddAccessChainsChooseIndexForAccessChain(
      uint32_t composite_size_bound);

  virtual bool FuzzerPassAddAccessChainsChooseIsSigned();

  //
  // End of fuzz pass member functions.
  //

 protected:
  // Decides whether to use a signed number. We may want to bias this across all
  // passes, so all fuzzer-pass specific functions above should call this by
  // default.
  virtual bool ChooseIsSigned();

  virtual uint32_t ChooseBetweenMinAndMax(
      const std::pair<uint32_t, uint32_t>& min_max);

  virtual bool ChoosePercentage(uint32_t percentage_chance);

  virtual uint32_t RandomIndex(size_t size);

  virtual bool ChooseEven();

  virtual uint32_t RandomUint32(uint32_t bound);

  uint32_t fuzzer_pass_add_access_chains_chance_of_adding_access_chain_;
  uint32_t fuzzer_pass_add_access_chains_chance_of_going_deeper_;

 private:
  std::mt19937 engine_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_DECISION_MAKER_H_
