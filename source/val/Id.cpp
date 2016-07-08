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

#include "val/Id.h"

namespace libspirv {
#define OPERATOR(OP)                               \
  bool operator OP(const Id& lhs, const Id& rhs) { \
    return lhs.id_ OP rhs.id_;                     \
  }                                                \
  bool operator OP(const Id& lhs, uint32_t rhs) { return lhs.id_ OP rhs; }

OPERATOR(<)
OPERATOR(==)
#undef OPERATOR

Id::Id(const uint32_t result_id)
    : id_(result_id),
      type_id_(0),
      opcode_(SpvOpNop),
      defining_function_(nullptr),
      defining_block_(nullptr),
      uses_(),
      words_(0) {}

Id::Id(const spv_parsed_instruction_t* inst, Function* function,
       BasicBlock* block)
    : id_(inst->result_id),
      type_id_(inst->type_id),
      opcode_(static_cast<SpvOp>(inst->opcode)),
      defining_function_(function),
      defining_block_(block),
      uses_(),
      words_(inst->words, inst->words + inst->num_words) {}

void Id::RegisterUse(const BasicBlock* block) {
  if (block) { uses_.insert(block); }
}
}  // namespace libspirv
