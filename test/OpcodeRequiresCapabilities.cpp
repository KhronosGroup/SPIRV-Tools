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

class Requires : public ::testing::TestWithParam<Capability> {
 public:
  Requires()
      : entry({nullptr,
               0,
               (Op)0,
               SPV_OPCODE_FLAGS_CAPABILITIES,
               GetParam(),
               {}}) {}

  virtual void SetUp() {}

  virtual void TearDown() {}

  spv_opcode_desc_t entry;
};

TEST_P(Requires, Capabilityabilities) {
  ASSERT_NE(0, spvOpcodeRequiresCapabilities(&entry));
}

INSTANTIATE_TEST_CASE_P(Op, Requires,
                        ::testing::Values(CapabilityMatrix, CapabilityShader,
                                          CapabilityGeometry,
                                          CapabilityTessellation,
                                          CapabilityAddresses,
                                          CapabilityLinkage, CapabilityKernel));

TEST(OpcodeRequiresCapabilityaspvities, None) {
  spv_opcode_desc_t entry = {nullptr, 0, (Op)0, SPV_OPCODE_FLAGS_NONE, 0, {}};
  ASSERT_EQ(0, spvOpcodeRequiresCapabilities(&entry));
}
