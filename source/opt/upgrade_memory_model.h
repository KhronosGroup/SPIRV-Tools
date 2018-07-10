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

#include <functional>
#include <tuple>

namespace spvtools {
namespace opt {

struct CacheHash {
  size_t operator()(
      const std::pair<uint32_t, std::vector<uint32_t>>& item) const {
    std::u32string to_hash;
    to_hash.push_back(item.first);
    for (auto i : item.second) to_hash.push_back(i);
    return std::hash<std::u32string>()(to_hash);
  }
};

class UpgradeMemoryModel : public Pass {
 public:
  const char* name() const override { return "upgrade-memory-model"; }
  Status Process(ir::IRContext* context) override;

 private:
  void UpgradeMemoryModelInstruction();
  void UpgradeInstructions();
  std::tuple<bool, bool, SpvScope> GetInstructionAttributes(uint32_t id);
  std::pair<bool, bool> TraceInstruction(ir::Instruction* inst,
                                         std::vector<uint32_t> indices,
                                         std::unordered_set<uint32_t>* visited);
  bool HasDecoration(const ir::Instruction* inst, uint32_t value,
                     SpvDecoration decoration);
  std::pair<bool, bool> CheckType(uint32_t type_id,
                                  const std::vector<uint32_t>& indices);
  std::pair<bool, bool> CheckAllTypes(const ir::Instruction* inst);
  void UpgradeFlags(ir::Instruction* inst, uint32_t in_operand,
                    bool is_coherent, bool is_volatile, bool visible,
                    bool is_memory);
  uint32_t GetScopeConstant(SpvScope scope);
  uint64_t GetIndexValue(ir::Instruction* index_inst);
  void CleanupDecorations();

  std::unordered_map<std::pair<uint32_t, std::vector<uint32_t>>,
                     std::pair<bool, bool>, CacheHash>
      cache_;
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_UPGRADE_MEMORY_MODEL_H_
