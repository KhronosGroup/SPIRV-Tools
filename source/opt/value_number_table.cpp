// Copyright (c) 2017 Google Inc.
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

#include "value_number_table.h"

#include <algorithm>

namespace spvtools {
namespace opt {

uint32_t ValueNumberTable::GetValueNumber(spvtools::ir::Instruction* inst) {
  // TODO: Need to implement the substitution of operands by their value number
  // before hashing.
  // TODO: Implement a normal form for opcodes that commute like integer
  // addition.  This will let us know that a+b is the same value as b+a.
  assert(inst->result_id() != 0 &&
         "inst must have a result id to get a value number.");

  // Check if this instruction already has a value.
  auto id_to_val = id_to_value_.find(inst->result_id());
  if (id_to_val != id_to_value_.end()) {
    return id_to_val->second;
  }

  // If the instruction has other side effects, then it must
  // have its own value number.
  if (!context()->IsCombinatorInstruction(inst)) {
    uint32_t value_number = TakeNextValueNumber();
    id_to_value_[inst->result_id()] = value_number;
    return value_number;
  }

  // If it is a load from memory that can be modified, we have to assume the
  // memory has been modified, so we give it a new value number.
  //
  // Note that this test will also handle volatile loads because they are not
  // read only.  However, if this is ever relaxed because we analyze stores, we
  // will have to add a new case for volatile loads.
  if (inst->IsLoad() && !inst->IsReadOnlyLoad()) {
    uint32_t value_number = TakeNextValueNumber();
    id_to_value_[inst->result_id()] = value_number;
    return value_number;
  }

  // Otherwise, we check if this value has been computed before.
  auto value = instruction_to_value_.find(*inst);
  if (value != instruction_to_value_.end()) {
    uint32_t value_number = id_to_value_[value->first.result_id()];
    id_to_value_[inst->result_id()] = value_number;
    return value_number;
  }

  // If not, assign it a new value number.
  uint32_t value_number = TakeNextValueNumber();
  id_to_value_[inst->result_id()] = value_number;
  instruction_to_value_[*inst] = value_number;
  return value_number;
}

bool ComputeSameValue::operator()(const ir::Instruction& lhs,
                                  const ir::Instruction& rhs) const {
  if (lhs.result_id() == 0 || rhs.result_id() == 0) {
    return false;
  }

  if (lhs.opcode() != rhs.opcode()) {
    return false;
  }

  if (lhs.type_id() != rhs.type_id()) {
    return false;
  }

  if (lhs.NumInOperands() != rhs.NumInOperands()) {
    return false;
  }

  for (uint32_t i = 0; i < lhs.NumInOperands(); ++i) {
    if (lhs.GetInOperand(i) != rhs.GetInOperand(i)) {
      return false;
    }
  }

  return lhs.context()->get_decoration_mgr()->HaveTheSameDecorations(
      lhs.result_id(), rhs.result_id());
}

std::size_t ValueTableHash::operator()(
    const spvtools::ir::Instruction& inst) const {
  // We hash the opcode and in-operands, not the result, because we want
  // instructions that are the same except for the result to hash to the same
  // value.
  std::u32string h;
  h.push_back(inst.opcode());
  h.push_back(inst.type_id());
  for (uint32_t i = 0; i < inst.NumInOperands(); ++i) {
    const auto& opnd = inst.GetInOperand(i);
    for (uint32_t word : opnd.words) {
      h.push_back(word);
    }
  }
  return std::hash<std::u32string>()(h);
}
}  // namespace opt
}  // namespace spvtools
