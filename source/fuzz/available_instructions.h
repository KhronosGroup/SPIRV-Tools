// Copyright (c) 2021 Alastair F. Donaldson
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

#ifndef SOURCE_FUZZ_AVAILABLE_INSTRUCTIONS_H_
#define SOURCE_FUZZ_AVAILABLE_INSTRUCTIONS_H_

#include <unordered_map>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class AvailableInstructions {
 public:
  class AvailableBeforeInstruction {
   public:
    AvailableBeforeInstruction(
        const AvailableInstructions& available_instructions,
        opt::Instruction* inst);
    opt::Instruction* operator[](uint32_t index) const;
    uint32_t size() const;
    bool empty() const;

   private:
    const AvailableInstructions& available_instructions_;
    opt::Instruction* inst_;
    mutable std::unordered_map<uint32_t, opt::Instruction*> index_cache;
  };

  AvailableInstructions(
      opt::IRContext* ir_context,
      const std::function<bool(opt::IRContext*, opt::Instruction*)>& filter);

  AvailableBeforeInstruction GetAvailableBeforeInstruction(
      opt::Instruction* inst) const;

 private:
  opt::IRContext* ir_context_;
  std::vector<opt::Instruction*> available_globals_;
  std::unordered_map<opt::Function*, std::vector<opt::Instruction*>>
      available_params_;
  std::unordered_map<opt::BasicBlock*, uint32_t> num_available_at_block_entry_;
  std::unordered_map<opt::Instruction*, uint32_t>
      num_available_before_instruction_;
  std::unordered_map<opt::BasicBlock*, std::vector<opt::Instruction*>>
      generated_by_block_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_AVAILABLE_INSTRUCTIONS_H_
