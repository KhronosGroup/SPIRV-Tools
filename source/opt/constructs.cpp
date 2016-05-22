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

#include <cassert>

#include "constructs.h"

namespace spvtools {
namespace opt {
namespace ir {

Inst::Inst(const spv_parsed_instruction_t& inst)
    : opcode_(static_cast<SpvOp>(inst.opcode)),
      type_id_(inst.type_id),
      result_id_(inst.result_id),
      words_(inst.words, inst.words + inst.num_words),
      operands_(inst.operands, inst.operands + inst.num_operands) {}

uint32_t Inst::GetOperandWord(size_t index) const {
  // March towards the start of "real" operands.
  index += 1 + (type_id_ != 0) + (result_id_ != 0);
  if (index >= words_.size()) {
    // TODO(antiagainst): do better than panicking.
    assert(0 && "operand index out of bound");
  }

  return words_[index];
}

uint32_t Inst::NumOperandWord() const {
  // March towards the start of "real" operands.
  const uint32_t count = static_cast<uint32_t>(words_.size());
  const uint32_t pre_operand = 1 + (type_id_ != 0) + (result_id_ != 0);
  assert(count >= pre_operand && "invalid instruction");
  return count - pre_operand;
}

void BasicBlock::ToBinary(std::vector<uint32_t>* binary) const {
  label_.ToBinary(binary);
  for (const auto& inst : insts_) inst.ToBinary(binary);
}

void Function::ToBinary(std::vector<uint32_t>* binary) const {
  def_inst_.ToBinary(binary);
  for (const auto& param : params_) param.ToBinary(binary);
  for (const auto& bb : blocks_) bb.ToBinary(binary);
  end_inst_.ToBinary(binary);
}

void Module::ToBinary(std::vector<uint32_t>* binary) const {
  binary->push_back(header_.magic_number);
  binary->push_back(header_.version);
  // TODO(antiagainst): should we change the generator number?
  binary->push_back(header_.generator);
  binary->push_back(header_.bound);
  binary->push_back(header_.reserved);

  for (const auto& c : capabilities_) c.ToBinary(binary);
  for (const auto& e : extensions_) e.ToBinary(binary);
  for (const auto& e : ext_inst_sets_) e.ToBinary(binary);
  memory_model_.ToBinary(binary);
  for (const auto& e : entry_points_) e.ToBinary(binary);
  for (const auto& e : execution_modes_) e.ToBinary(binary);
  for (const auto& d : debugs_) d.ToBinary(binary);
  for (const auto& a : annotations_) a.ToBinary(binary);
  for (const auto& t : types_) t.ToBinary(binary);
  for (const auto& c : constants_) c.ToBinary(binary);
  for (const auto& v : variables_) v.ToBinary(binary);
  for (const auto& f : functions_) f.ToBinary(binary);
}

}  // namespace ir
}  // namespace opt
}  // namespace spvtools
