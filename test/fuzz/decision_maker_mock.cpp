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

#include "test/fuzz/decision_maker_mock.h"

namespace spvtools {
namespace fuzz {

uint32_t DecisionMakerMock::RandomUint32(uint32_t bound) {
  assert(false &&
         "Should not have reached a random decision in DecisionMakerMock");
  return 0;
}
}  // namespace fuzz
}  // namespace spvtools
