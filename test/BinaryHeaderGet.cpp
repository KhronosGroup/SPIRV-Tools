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

class BinaryHeaderGet : public ::testing::Test {
 public:
  BinaryHeaderGet() { memset(code, 0, sizeof(code)); }

  virtual void SetUp() {
    code[0] = SpvMagicNumber;
    code[1] = SpvVersion;
    code[2] = SPV_GENERATOR_CODEPLAY;
    code[3] = 1;  // NOTE: Bound
    code[4] = 0;  // NOTE: Schema; reserved
    code[5] = 0;  // NOTE: Instructions

    binary.code = code;
    binary.wordCount = 6;
  }

  virtual void TearDown() {}

  uint32_t code[6];
  spv_binary_t binary;
};

TEST_F(BinaryHeaderGet, Default) {
  spv_endianness_t endian;
  ASSERT_EQ(SPV_SUCCESS, spvBinaryEndianness(&binary, &endian));

  spv_header_t header;
  ASSERT_EQ(SPV_SUCCESS, spvBinaryHeaderGet(&binary, endian, &header));

  ASSERT_EQ(static_cast<uint32_t>(SpvMagicNumber), header.magic);
  ASSERT_EQ(99u, header.version);
  ASSERT_EQ(static_cast<uint32_t>(SPV_GENERATOR_CODEPLAY), header.generator);
  ASSERT_EQ(1u, header.bound);
  ASSERT_EQ(0u, header.schema);
  ASSERT_EQ(&code[5], header.instructions);
}

TEST_F(BinaryHeaderGet, InvalidCode) {
  spv_binary_t binary = {nullptr, 0};
  spv_header_t header;
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY,
            spvBinaryHeaderGet(&binary, SPV_ENDIANNESS_LITTLE, &header));
}

TEST_F(BinaryHeaderGet, InvalidPointerHeader) {
  ASSERT_EQ(SPV_ERROR_INVALID_POINTER,
            spvBinaryHeaderGet(&binary, SPV_ENDIANNESS_LITTLE, nullptr));
}

TEST_F(BinaryHeaderGet, TruncatedHeader) {
  for (int i = 1; i < SPV_INDEX_INSTRUCTION; i++) {
    binary.wordCount = i;
    ASSERT_EQ(SPV_ERROR_INVALID_BINARY,
              spvBinaryHeaderGet(&binary, SPV_ENDIANNESS_LITTLE, nullptr));
  }
}

}  // anonymous namespace
