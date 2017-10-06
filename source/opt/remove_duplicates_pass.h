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

#ifndef LIBSPIRV_OPT_REMOVE_DUPLICATES_PASS_H_
#define LIBSPIRV_OPT_REMOVE_DUPLICATES_PASS_H_

#include <unordered_map>

#include "decoration_manager.h"
#include "def_use_manager.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

using IdDecorationsList =
    std::unordered_map<uint32_t, std::vector<ir::Instruction*>>;

// See optimizer.hpp for documentation.
class RemoveDuplicatesPass : public Pass {
 public:
  const char* name() const override { return "remove-duplicates"; }
  Status Process(ir::Module*) override;
  // Returns whether two types are equal, and have the same decorations.
  static bool AreTypesEqual(const ir::Instruction& inst1,
                            const ir::Instruction& inst2,
                            const analysis::DefUseManager& defUseManager,
                            const analysis::DecorationManager& decoManager);

 private:
  bool RemoveDuplicateCapabilities(ir::Module* module) const;
  bool RemoveDuplicatesExtInstImports(
      ir::Module* module, analysis::DefUseManager& defUseManager) const;
  bool RemoveDuplicateTypes(ir::Module* module,
                            analysis::DefUseManager& defUseManager,
                            analysis::DecorationManager& decManager) const;
  bool RemoveDuplicateDecorations(ir::Module* module) const;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_REMOVE_DUPLICATES_PASS_H_
