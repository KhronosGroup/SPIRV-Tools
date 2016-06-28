// Copyright (c) 2016 Google Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#ifndef LIBSPIRV_OPT_INSTRUCTION_H_
#define LIBSPIRV_OPT_INSTRUCTION_H_

#include <cassert>
#include <functional>
#include <utility>
#include <vector>

#include "spirv-tools/libspirv.h"
#include "spirv/1.1/spirv.h"

namespace spvtools {
namespace ir {

class Function;
class Module;

// About operand:
//
// In the SPIR-V specification, the term "operand" is used to mean any single
// SPIR-V word following the leading wordcount-opcode word. Here, the term
// "operand" is used to mean a *logical* operand. A logical operand may consist
// of mulitple SPIR-V words, which together make up the same component. For
// example, a logical operand of a 64-bit integer needs two words to express.
//
// Further, we categorize logical operands into *in* and *out* operands.
// In operands are operands actually serve as input to operations, while out
// operands are operands that represent ids generated from operations (result
// type id or result id). For example, for "OpIAdd %rtype %rid %inop1 %inop2",
// "%inop1" and "%inop2" are in operands, while "%rtype" and "%rid" are out
// operands.

// A *logical* operand to a SPIR-V instruction. It can be the type id, result
// id, or other additional operands carried in an instruction.
struct Operand {
  Operand(spv_operand_type_t t, std::vector<uint32_t>&& w)
      : type(t), words(std::move(w)) {}

  spv_operand_type_t type;      // Type of this logical operand.
  std::vector<uint32_t> words;  // Binary segments of this logical operand.

  // TODO(antiagainst): create fields for literal number kind, width, etc.
};

// A SPIR-V instruction. It contains the opcode and any additional logical
// operand. It may also contain line-related debug instruction (OpLine,
// OpNoLine) directly appearing before this instruction.
class Instruction {
 public:
  // Creates a default OpNop instruction.
  Instruction() : opcode_(SpvOpNop), type_id_(0), result_id_(0) {}
  // Creates an instruction with the given opcode |op| and no additional logical
  // operands.
  Instruction(SpvOp op) : opcode_(op), type_id_(0), result_id_(0) {}
  // Creates an instruction using the given spv_parsed_instruction_t |inst|. All
  // the data inside |inst| will be copied and owned in this instance. And keep
  // record of line-related debug instructions |dbg_line| ahead of this
  // instruction, if any.
  Instruction(const spv_parsed_instruction_t& inst,
              std::vector<Instruction>&& dbg_line = {});

  SpvOp opcode() const { return opcode_; }
  uint32_t type_id() const { return type_id_; }
  uint32_t result_id() const { return result_id_; }
  // Returns the vector of line-related debug instructions attached to this
  // instruction and the caller can directly modify them.
  std::vector<Instruction>& dbg_line_insts() { return dbg_line_insts_; }
  const std::vector<Instruction>& dbg_line_insts() const {
    return dbg_line_insts_;
  }

  // Gets the number of logical operands.
  uint32_t NumOperands() const {
    return static_cast<uint32_t>(operands_.size());
  }
  // Gets the number of SPIR-V words occupied by all logical operands.
  uint32_t NumOperandWords() const {
    return NumInOperandWords() + TypeResultIdCount();
  }
  // Gets the |index|-th logical operand.
  inline const Operand& GetOperand(uint32_t index) const;
  // Gets the |index|-th logical operand as a single SPIR-V word. This method is
  // not expected to be used with logical operands consisting of multiple SPIR-V
  // words.
  uint32_t GetSingleWordOperand(uint32_t index) const;
  // Sets the |index|-th operand's data to the given |data|.
  inline void SetOperand(uint32_t index, std::vector<uint32_t>&& data);

  // The following methods are similar to the above, but are for in operands.
  uint32_t NumInOperands() const {
    return static_cast<uint32_t>(operands_.size() - TypeResultIdCount());
  }
  uint32_t NumInOperandWords() const;
  const Operand& GetInOperand(uint32_t index) const {
    return GetOperand(index + TypeResultIdCount());
  }
  uint32_t GetSingleWordInOperand(uint32_t index) const {
    return GetSingleWordOperand(index + TypeResultIdCount());
  }

  // Returns true if this instruction is OpNop.
  inline bool IsNop() const;
  // Turns this instruction to OpNop. This does not clear out all preceding
  // line-related debug instructions.
  inline void ToNop();

  // Runs the given function |f| on this instruction and preceding debug
  // instructions.
  inline void ForEachInst(const std::function<void(Instruction*)>& f);

  // Pushes the binary segments for this instruction into the back of *|binary|.
  // If |skip_nop| is true and this is a OpNop, do nothing.
  void ToBinary(std::vector<uint32_t>* binary, bool skip_nop) const;

 private:
  // Returns the toal count of result type id and result id.
  uint32_t TypeResultIdCount() const {
    return (type_id_ != 0) + (result_id_ != 0);
  }

  SpvOp opcode_;        // Opcode
  uint32_t type_id_;    // Result type id. A value of 0 means no result type id.
  uint32_t result_id_;  // Result id. A value of 0 means no result id.
  // All logical operands, including result type id and result id.
  std::vector<Operand> operands_;
  // Opline and OpNoLine instructions preceding this instruction. Note that for
  // Instructions representing OpLine or OpNonLine itself, this field should be
  // empty.
  std::vector<Instruction> dbg_line_insts_;
};

inline const Operand& Instruction::GetOperand(uint32_t index) const {
  assert(index < operands_.size() && "operand index out of bound");
  return operands_[index];
};

inline void Instruction::SetOperand(uint32_t index,
                                    std::vector<uint32_t>&& data) {
  assert(index < operands_.size() && "operand index out of bound");
  operands_[index].words = std::move(data);
}

inline bool Instruction::IsNop() const {
  return opcode_ == SpvOpNop && type_id_ == 0 && result_id_ == 0 &&
         operands_.empty();
}

inline void Instruction::ToNop() {
  opcode_ = SpvOpNop;
  type_id_ = result_id_ = 0;
  operands_.clear();
}

inline void Instruction::ForEachInst(
    const std::function<void(Instruction*)>& f) {
  for (auto& dbg_line : dbg_line_insts_) f(&dbg_line);
  f(this);
}

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INSTRUCTION_H_
