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

#include "instruction.h"

#include <cassert>

#include "reflect.h"

namespace spvtools {
namespace ir {

Instruction::Instruction(const spv_parsed_instruction_t& inst,
                         std::vector<Instruction>&& dbg_line)
    : opcode_(static_cast<SpvOp>(inst.opcode)),
      type_id_(inst.type_id),
      result_id_(inst.result_id),
      dbg_line_insts_(std::move(dbg_line)) {
  assert((!IsDebugLineInst(opcode_) || dbg_line.empty()) &&
         "Op(No)Line attaching to Op(No)Line found");
  for (uint32_t i = 0; i < inst.num_operands; ++i) {
    const auto& current_payload = inst.operands[i];
    std::vector<uint32_t> words(
        inst.words + current_payload.offset,
        inst.words + current_payload.offset + current_payload.num_words);
    operands_.emplace_back(current_payload.type, std::move(words));
  }
}

uint32_t Instruction::GetSingleWordOperand(uint32_t index) const {
  const auto& words = GetOperand(index).words;
  assert(words.size() == 1 && "expected the operand only taking one word");
  return words.front();
}

uint32_t Instruction::NumInOperandWords() const {
  uint32_t size = 0;
  for (uint32_t i = TypeResultIdCount(); i < operands_.size(); ++i)
    size += static_cast<uint32_t>(operands_[i].words.size());
  return size;
}

void Instruction::ToBinary(std::vector<uint32_t>* binary, bool skip_nop) const {
  for (const auto& dbg_line : dbg_line_insts_) {
    dbg_line.ToBinary(binary, skip_nop);
  }

  if (skip_nop && IsNop()) return;

  const uint32_t num_words = 1 + NumOperandWords();
  binary->push_back((num_words << 16) | static_cast<uint16_t>(opcode_));
  for (const auto& operand : operands_)
    binary->insert(binary->end(), operand.words.begin(), operand.words.end());
}

}  // namespace ir
}  // namespace spvtools
