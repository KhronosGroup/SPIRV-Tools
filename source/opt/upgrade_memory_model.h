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

#ifndef LIBSPIRV_OPT_UPGRADE_MEMORY_MODEL_H_
#define LIBSPIRV_OPT_UPGRADE_MEMORY_MODEL_H_

#include "pass.h"

namespace spvtools {
namespace opt {

class UpgradeMemoryModel : public Pass {
 public:
  const char* name() const override { return "upgrade-memory-model"; }
  Status Process(ir::IRContext* context) override;

  struct Tracker {
    ir::Instruction* inst;
    bool is_volatile;
    bool is_coherent;
    uint32_t in_operand;
    uint32_t nesting;
    int member_index;
    SpvScope scope;

    Tracker() = default;
    Tracker(const Tracker&) = default;

    Tracker(ir::Instruction* i)
        : inst(i),
          is_volatile(false),
          is_coherent(false),
          in_operand(0),
          nesting(0),
          member_index(-1),
          scope(SpvScopeDevice) {}
  };

 private:
  void UpgradeMemoryModelInstruction();
  void UpgradeInstructions();
  void UpgradeInstruction(const Tracker& tracker);
  void UpgradeFlags(const Tracker& tracker);
  uint32_t GetScopeConstant(SpvScope scope);
  void CleanupDecorations();
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_UPGRADE_MEMORY_MODEL_H_
