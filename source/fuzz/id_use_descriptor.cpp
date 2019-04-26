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

#include "source/fuzz/id_use_descriptor.h"

namespace spvtools {
namespace fuzz {
namespace module_navigation {

uint32_t IdUseDescriptor::GetIdOfInterest() const { return id_of_interest_; }

opt::Instruction* IdUseDescriptor::FindInstruction(
    spvtools::opt::IRContext* context) const {
  (void)(context);
  assert(false && "TODO");
  return nullptr;
}

}  // namespace module_navigation
}  // namespace fuzz
}  // namespace spvtools
