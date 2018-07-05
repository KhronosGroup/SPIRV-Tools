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

#include <tuple>

namespace spvtools {
namespace opt {

class UpgradeMemoryModel : public Pass {
 public:
  const char* name() const override { return "upgrade-memory-model"; }
  Status Process(ir::IRContext* context) override;

 private:
  void UpgradeMemoryModelInstruction();
  void UpgradeInstructions();
  std::tuple<bool, bool, SpvScope> GetInstructionAttributes(uint32_t id);
  void UpgradeFlags(ir::Instruction* inst, uint32_t in_operand,
                    bool is_coherent, bool is_volatile, bool visible,
                    bool is_memory);
  uint32_t GetScopeConstant(SpvScope scope);
  void CleanupDecorations();
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_UPGRADE_MEMORY_MODEL_H_
