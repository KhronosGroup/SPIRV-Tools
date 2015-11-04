// Copyright (c) 2015 The Khronos Group Inc.
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

#include "UnitSPIRV.h"

namespace {

TEST(OperandTableGet, Default) {
  spv_operand_table table;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&table));
  ASSERT_NE(0, table->count);
  ASSERT_NE(nullptr, table->types);
}

TEST(OperandTableGet, InvalidPointerTable) {
  ASSERT_EQ(SPV_ERROR_INVALID_POINTER, spvOperandTableGet(nullptr));
}

TEST(OperandString, AllAreDefinedExceptVariable) {
  EXPECT_EQ(0, SPV_OPERAND_TYPE_NONE);  // None has no string, so don't test it.
  // Start testing at enum with value 1, skipping None.
  for (int i = 1; i < int(SPV_OPERAND_TYPE_FIRST_VARIABLE_TYPE); i++) {
    EXPECT_NE(nullptr, spvOperandTypeStr(static_cast<spv_operand_type_t>(i)))
        << " Operand type " << i;
  }
}

}  // anonymous namespace
