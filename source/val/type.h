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

#include <vector>

#include "spirv-tools/libspirv.h"
#include "spirv/1.1/spirv.h"

namespace libspirv {

class Type {
 public:
  explicit Type(const spv_parsed_instruction_t& inst)
      : id_(inst.result_id),
        type_(static_cast<SpvOp>(inst.opcode)),
        operands_(inst.words + inst.operands->offset + 1,  // skip result_id
                  inst.words + inst.num_words) {}

  bool IsAlias(const spv_parsed_instruction_t& inst) const {
    if (inst.opcode == type_
        && equal(begin(operands()), end(operands()),
              inst.words + inst.operands->offset + 1)) {
      return true;
    }
    return false;
  }

  uint32_t id() const { return id_; }
  SpvOp type() const { return type_; }
  const std::vector<uint32_t>& operands() const { return operands_; }

 private:
  uint32_t id_;
  SpvOp type_;
  std::vector<uint32_t> operands_;
};

}  // namespace libspirv

#endif  /// LIBSPIRV_VAL_CONSTRUCT_H_
