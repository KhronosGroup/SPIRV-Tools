// Copyright (c) 2019 Google Inc.
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

#ifndef INCLUDE_SPIRV_TOOLS_INSTRUCTION_HPP_
#define INCLUDE_SPIRV_TOOLS_INSTRUCTION_HPP_

#include "source/latest_version_spirv_header.h"
#include "spirv-tools/libspirv.hpp"

#include <cassert>
#include <memory>
#include <vector>

namespace spvtools {

class Instruction {
 public:
  virtual ~Instruction() {}

  // Construct derived class corresponding to 'inst'
  static std::shared_ptr<Instruction> Make(
      const spv_parsed_instruction_t* inst);

  // Safe cast to derived instruction.
  template <typename T>
  const T* Get() const {
    if (inst_.opcode == T::Opcode) return static_cast<const T*>(this);
    return nullptr;
  }

  // Alternative safe cast.
  template <typename T>
  bool Get(T* ptr) const {
    T* p = Get<T>();
    if (p) {
      *ptr = p;
      return true;
    }
    return false;
  }

  uint32_t Opcode() const { return inst_.opcode; }

  // TODO(fjhenigman): Eliminate uses of the following functions then delete
  // them.

  // Words from the start of operand 'pos' to end of instruction.
  const std::vector<uint32_t> GetWords(unsigned pos) const {
    return getvec<uint32_t>(pos);
  }

  /// The words used to define the Instruction
  const std::vector<uint32_t>& words() const { return words_; }

  /// The operands of the Instruction
  const std::vector<spv_parsed_operand_t>& operands() const {
    return operands_;
  }

  /// Provides direct access to the stored C instruction object.
  const spv_parsed_instruction_t& c_inst() const { return inst_; }

 protected:
  explicit Instruction(const spv_parsed_instruction_t* inst)
      : words_(inst->words, inst->words + inst->num_words),
        operands_(inst->operands, inst->operands + inst->num_operands),
        inst_({words_.data(), inst->num_words, inst->opcode,
               inst->ext_inst_type, inst->type_id, inst->result_id,
               operands_.data(), inst->num_operands}) {}

  // Operand 'pos' as 'T'
  template <typename T>
  T getval(unsigned pos) const {
    // TODO(fjhenigman): If op[x].offset + op[x].num_words always equals
    // op[x+1].offset (or end of instruction if 'x' is last op) then the next
    // three lines (which is how spvtools::val::Instruction did it) can be
    // removed as they are equivalent to the simpler size check below.
    const spv_parsed_operand_t& op = inst_.operands[pos];
    assert(op.num_words * sizeof(uint32_t) == sizeof(T));
    assert(op.offset + op.num_words <= inst_.num_words);

    assert(pos < inst_.num_operands);
    assert(getbegin<char>(pos + 1) - getbegin<char>(pos) == sizeof(T));
    return *getptr<T>(pos);
  }

  // Operand 'pos' as string
  const char* getstr(unsigned pos) const {
    assert(pos < inst_.num_operands);
    return getptr<char>(pos);
  }

  // Operand 'pos' as vector<T> using all remaining words in instruction
  template <typename T>
  const std::vector<T> getvec(unsigned pos) const {
    // Assert bytes available is multiple of T size
    assert((getend<char>() - getbegin<char>(pos)) % sizeof(T) == 0);
    return std::vector<T>(getbegin<T>(pos), getend<T>());
  }

 private:
  // Pointer to first word of operand 'pos' with no checking that operand exists
  template <typename T>
  const T* getptr(unsigned pos) const {
    return reinterpret_cast<const T*>(inst_.words + inst_.operands[pos].offset);
  }

  // Pointer to first word of operand 'pos.'
  // If operand not present returns getend<T>().
  template <typename T>
  const T* getbegin(unsigned pos) const {
    return pos < inst_.num_operands ? getptr<T>(pos) : getend<T>();
  }

  // End of instruction words i.e. one past last word.
  template <typename T>
  const T* getend() const {
    return reinterpret_cast<const T*>(inst_.words + inst_.num_words);
  }

  const std::vector<uint32_t> words_;
  const std::vector<spv_parsed_operand_t> operands_;
  const spv_parsed_instruction_t inst_;
};

#include <spirv-tools/instructions.hpp.inc>
}  // namespace spvtools

#endif  // INCLUDE_SPIRV_TOOLS_INSTRUCTION_HPP_
