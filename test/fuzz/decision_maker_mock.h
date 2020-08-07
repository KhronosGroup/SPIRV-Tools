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

#ifndef TEST_FUZZ_DECISION_MAKER_MOCK_H_
#define TEST_FUZZ_DECISION_MAKER_MOCK_H_

#include "source/fuzz/decision_maker.h"

namespace spvtools {
namespace fuzz {

// A version of DecisionMaker that should be used for testing. |RandomUint32|
// has been overridden to assert false. Thus, all member functions that might
// lead to the use of the random number generator must be overridden as part of
// the test.
//
// For example, when testing FuzzerPassAddAccessChains, we must create a class
// deriving from DecisionMakerMock that overrides:
//  - |FuzzerPassAddAccessChainsShouldAddLoad|
//  - |FuzzerPassAddAccessChainsChoosePointerInstruction|
//  - etc.
//
// This ensures that all random choices in FuzzerPassAddAccessChains are
// controlled.
class DecisionMakerMock : public DecisionMaker {
 protected:
  virtual uint32_t RandomUint32(uint32_t bound);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // TEST_FUZZ_DECISION_MAKER_MOCK_H_
