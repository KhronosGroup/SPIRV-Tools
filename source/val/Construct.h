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

#ifndef LIBSPIRV_VAL_CONSTRUCT_H_
#define LIBSPIRV_VAL_CONSTRUCT_H_

#include <cstdint>

namespace libspirv {

class BasicBlock;

/// @brief This class tracks the CFG constructs as defined in the SPIR-V spec
class Construct {
 public:
  Construct(BasicBlock* header_block, BasicBlock* merge_block,
            BasicBlock* continue_block = nullptr);

  const BasicBlock* get_header() const;
  const BasicBlock* get_merge() const;
  const BasicBlock* get_continue() const;

  BasicBlock* get_header();
  BasicBlock* get_merge();
  BasicBlock* get_continue();

 private:
  BasicBlock* header_block_;    ///< The header block of a loop or selection
  BasicBlock* merge_block_;     ///< The merge block of a loop or selection
  BasicBlock* continue_block_;  ///< The continue block of a loop block
};

}  /// namespace libspirv

#endif  /// LIBSPIRV_VAL_CONSTRUCT_H_
