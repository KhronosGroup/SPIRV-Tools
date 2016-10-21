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

#ifndef LIBSPIRV_VAL_TYPE_H_
#define LIBSPIRV_VAL_TYPE_H_

#include <cstdint>

#include <algorithm>
#include <functional>
#include <vector>

#include "spirv-tools/libspirv.h"
#include "spirv/1.1/spirv.h"

namespace libspirv {

spv_type_category_t OpcodeToTypeFlag(SpvOp opcode);

class Type {
 public:
  explicit Type(const spv_parsed_instruction_t& inst)
      : id_(inst.result_id),
        category_(OpcodeToTypeFlag(static_cast<SpvOp>(inst.opcode))),
        operands_(inst.words + inst.operands->offset + 1,  // skip result_id
                  inst.words + inst.num_words) {}

  bool IsAlias(const spv_parsed_instruction_t& inst) const {
    spv_type_category_t cat =
        OpcodeToTypeFlag(static_cast<SpvOp>(inst.opcode));

    // TODO(umar): comment about the magic numbers
    if (cat == category_ && operands().size() ==
        size_t(inst.num_words - 2) &&
        equal(begin(operands()), end(operands()),
              inst.words + inst.operands->offset + 1)) {
      return true;
    }
    return false;
  }

  bool IsType(spv_type_category_t cat) { return cat & category_; }
  bool IsTypeAny(const std::vector<spv_type_category_t>& possibilities) {
    return any_of(
        begin(possibilities), end(possibilities),
        [this](spv_type_category_t cat) { return cat & category_; });
  }

  uint32_t id() const { return id_; }
  spv_type_category_t category() const { return category_; }
  const std::vector<uint32_t>& operands() const { return operands_; }

 private:
  uint32_t id_;
  spv_type_category_t category_;
  std::vector<uint32_t> operands_;
};

}  // namespace libspirv

#endif  // LIBSPIRV_VAL_TYPE_H_
