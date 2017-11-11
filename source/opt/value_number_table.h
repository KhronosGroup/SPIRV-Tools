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

#ifndef LIBSPIRV_OPT_VALUE_NUMBER_TABLE_H_
#define LIBSPIRV_OPT_VALUE_NUMBER_TABLE_H_

#include <cstdint>
#include <unordered_map>
#include "instruction.h"
#include "ir_context.h"

namespace spvtools {
namespace opt {

// Returns true if the two instructions compute the same value.  Used by the
// value number table to compare two instructions.
class ComputeSameValue {
 public:
  bool operator()(const ir::Instruction& lhs, const ir::Instruction& rhs) const;
};

// The hash function used in the value number table.
class ValueTableHash {
 public:
  std::size_t operator()(const spvtools::ir::Instruction& inst) const;
};

// This class implements the value number analysis.  It is using a hash-based
// approach to value numbering.  For now, it is very simple, and computes value
// numbers for instructions when they are asked for via |GetValueNumber|.  This
// may change in the future and should not be relied on.
class ValueNumberTable {
 public:
  ValueNumberTable(ir::IRContext* ctx) : context_(ctx), next_value_number_(1) {}

  // Returns the value number of the value computed by |inst|.  |inst| must have
  // a result id that will hold the computed value.
  uint32_t GetValueNumber(spvtools::ir::Instruction* inst);

  // Returns the value number of the value contain in |id|.
  inline uint32_t GetValueNumber(uint32_t id);

  ir::IRContext* context() { return context_; }

 private:
  std::unordered_map<spvtools::ir::Instruction, uint32_t, ValueTableHash,
                     ComputeSameValue>
      instruction_to_value_;
  std::unordered_map<uint32_t, uint32_t> id_to_value_;
  ir::IRContext* context_;
  uint32_t next_value_number_;

  uint32_t TakeNextValueNumber() { return next_value_number_++; }
};

uint32_t ValueNumberTable::GetValueNumber(uint32_t id) {
  return GetValueNumber(context()->get_def_use_mgr()->GetDef(id));
}

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_VALUE_NUMBER_TABLE_H_
