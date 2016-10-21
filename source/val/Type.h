// Copyright (c) 2015-2016 The Khronos Group Inc.
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
