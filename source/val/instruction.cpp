// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "source/val/instruction.h"
#include "spirv-tools/instructions.hpp"

#include <utility>

namespace spvtools {
namespace val {

Instruction::Instruction(const spv_parsed_instruction_t* inst)
    : inst_(spvtools::Instruction::Make(inst)) {
  assert(inst_);
}

uint32_t Instruction::word(size_t index) const { return inst_->words()[index]; }

const std::vector<uint32_t>& Instruction::words() const {
  return inst_->words();
}

const spv_parsed_operand_t& Instruction::operand(size_t idx) const {
  return inst_->operands()[idx];
}

const std::vector<spv_parsed_operand_t>& Instruction::operands() const {
  return inst_->operands();
}

const spv_parsed_instruction_t& Instruction::c_inst() const {
  return inst_->c_inst();
}

void Instruction::RegisterUse(const Instruction* inst, uint32_t index) {
  uses_.push_back(std::make_pair(inst, index));
}

bool operator<(const Instruction& lhs, const Instruction& rhs) {
  return lhs.id() < rhs.id();
}
bool operator<(const Instruction& lhs, uint32_t rhs) { return lhs.id() < rhs; }
bool operator==(const Instruction& lhs, const Instruction& rhs) {
  return lhs.id() == rhs.id();
}
bool operator==(const Instruction& lhs, uint32_t rhs) {
  return lhs.id() == rhs;
}

}  // namespace val
}  // namespace spvtools
