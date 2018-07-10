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
  // Modifies the OpMemoryModel to use VulkanKHR. Adds the Vulkan memory model
  // capability and extension.
  void UpgradeMemoryModelInstruction();

  // Upgrades memory, image and barrier instructions.
  // Memory and image instructions convert coherent and volatile decorations
  // into flags on the instruction. Barriers in tessellation shaders get the
  // output storage semantic if appropriate.
  void UpgradeInstructions();

  // Returns whether |id| is coherent and/or volatile.
  std::tuple<bool, bool, SpvScope> GetInstructionAttributes(uint32_t id);

  // Traces |inst| to determine if it is coherent and/or volatile.
  // |indices| tracks the access chain indices seen so far.
  std::pair<bool, bool> TraceInstruction(ir::Instruction* inst,
                                         std::vector<uint32_t> indices,
                                         std::unordered_set<uint32_t>* visited);

  // Return true if |inst| is decorated with |decoration|.
  // If |inst| is decorated by member decorations then either |value| must
  // match the index or |value| must be a maximum allowable value. The max
  // value allows any element to match.
  bool HasDecoration(const ir::Instruction* inst, uint32_t value,
                     SpvDecoration decoration);

  // Returns whether |type_id| indexed via |indices| is coherent and/or
  // volatile.
  std::pair<bool, bool> CheckType(uint32_t type_id,
                                  const std::vector<uint32_t>& indices);

  // Returns whether any type/element under |inst| is coherent and/or volatile.
  std::pair<bool, bool> CheckAllTypes(const ir::Instruction* inst);

  // Modifies the flags of |inst| to include the new flags for the Vulkan
  // memory model. |visible| indicates whether flags should use MakeVisible
  // variants. |is_memory| indicates whether the Pointer variants of flags
  // should be used.
  void UpgradeFlags(ir::Instruction* inst, uint32_t in_operand,
                    bool is_coherent, bool is_volatile, bool visible,
                    bool is_memory);

  // Returns the result id for a constant for |scope|.
  uint32_t GetScopeConstant(SpvScope scope);

  // Returns the value of |index_inst|. |index_inst| must be an OpConstant of
  // integer type.g
  uint64_t GetIndexValue(ir::Instruction* index_inst);

  // Removes coherent and volatile decorations.
  void CleanupDecorations();

  // Caches the result of TraceInstruction. For a given result id and set of
  // indices, stores whether that combination is coherent and/or volatile.
  std::unordered_map<std::pair<uint32_t, std::vector<uint32_t>>,
                     std::pair<bool, bool>, CacheHash>
      cache_;
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_UPGRADE_MEMORY_MODEL_H_
