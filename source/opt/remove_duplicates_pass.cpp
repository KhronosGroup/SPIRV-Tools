// Copyright (c) 2017 Pierre Moreau
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

#include "remove_duplicates_pass.h"

#include <unordered_set>

namespace spvtools {
namespace opt {

using ir::Instruction;
using ir::Operand;

Pass::Status RemoveDuplicatesPass::Process(ir::Module* module) {
  bool modified = false;

  // Remove duplicate capabilities
  std::unordered_set<uint32_t> capabilities;
  for (auto i = module->capability_begin(); i != module->capability_end();) {
    auto res = capabilities.insert(i->GetSingleWordOperand(0u));
    i = (res.second) ? ++i : i.Erase();
    modified |= res.second;
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
