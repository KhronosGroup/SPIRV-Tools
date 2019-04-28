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

#ifndef SOURCE_VAL_INSTRUCTION_H_
#define SOURCE_VAL_INSTRUCTION_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "source/table.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {

class Instruction;

namespace val {

class BasicBlock;
class Function;

/// Contains a spvtools::Instruction plus all validator-specific per-instruction
/// data and methods.
class Instruction {
 public:
  explicit Instruction(const spv_parsed_instruction_t* inst);

  /// Registers the use of the Instruction in instruction \p inst at \p index
  void RegisterUse(const Instruction* inst, uint32_t index);

  uint32_t id() const { return c_inst().result_id; }
  uint32_t type_id() const { return c_inst().type_id; }
  SpvOp opcode() const { return static_cast<SpvOp>(c_inst().opcode); }

  /// Returns the Function where the instruction was defined. nullptr if it was
  /// defined outside of a Function
  const Function* function() const { return function_; }
  void set_function(Function* func) { function_ = func; }

  /// Returns the BasicBlock where the instruction was defined. nullptr if it
  /// was defined outside of a BasicBlock
  const BasicBlock* block() const { return block_; }
  void set_block(BasicBlock* b) { block_ = b; }

  /// Returns a vector of pairs of all references to this instruction's result
  /// id. The first element is the instruction in which this result id was
  /// referenced and the second is the index of the word in that instruction
  /// where this result id appeared
  const std::vector<std::pair<const Instruction*, uint32_t>>& uses() const {
    return uses_;
  }

  size_t LineNum() const { return line_num_; }
  void SetLineNum(size_t pos) { line_num_ = pos; }

  const spvtools::Instruction* inst() const { return inst_.get(); };

  // TODO(fjhenigman): eliminate usage of the following methods and delete them.
  // The are not specific to the validator and we can use the
  // spvtools::Instruction interfaces instead.

  /// The word used to define the Instruction
  uint32_t word(size_t index) const;

  /// The words used to define the Instruction
  const std::vector<uint32_t>& words() const;

  /// Returns the operand at |idx|.
  const spv_parsed_operand_t& operand(size_t idx) const;

  /// The operands of the Instruction
  const std::vector<spv_parsed_operand_t>& operands() const;

  /// Provides direct access to the stored C instruction object.
  const spv_parsed_instruction_t& c_inst() const;

  /// Provides direct access to instructions spv_ext_inst_type_t object.
  const spv_ext_inst_type_t& ext_inst_type() const {
    return c_inst().ext_inst_type;
  }

  // Casts the words belonging to the operand under |index| to |T| and returns.
  template <typename T>
  T GetOperandAs(size_t index) const {
    const spv_parsed_operand_t& o = operands().at(index);
    assert(o.num_words * 4 >= sizeof(T));
    assert(o.offset + o.num_words <= c_inst().num_words);
    return *reinterpret_cast<const T*>(&words()[o.offset]);
  }

 private:
  std::shared_ptr<const spvtools::Instruction> inst_;

  size_t line_num_ = 0;

  /// The function in which this instruction was declared
  Function* function_ = nullptr;

  /// The basic block in which this instruction was declared
  BasicBlock* block_ = nullptr;

  /// This is a vector of pairs of all references to this instruction's result
  /// id. The first element is the instruction in which this result id was
  /// referenced and the second is the index of the word in the referencing
  /// instruction where this instruction appeared
  std::vector<std::pair<const Instruction*, uint32_t>> uses_;
};

bool operator<(const Instruction& lhs, const Instruction& rhs);
bool operator<(const Instruction& lhs, uint32_t rhs);
bool operator==(const Instruction& lhs, const Instruction& rhs);
bool operator==(const Instruction& lhs, uint32_t rhs);

}  // namespace val
}  // namespace spvtools

// custom specialization of std::hash for Instruction
namespace std {
template <>
struct hash<spvtools::val::Instruction> {
  typedef spvtools::val::Instruction argument_type;
  typedef std::size_t result_type;
  result_type operator()(const argument_type& inst) const {
    return hash<uint32_t>()(inst.id());
  }
};

}  // namespace std

#endif  // SOURCE_VAL_INSTRUCTION_H_
