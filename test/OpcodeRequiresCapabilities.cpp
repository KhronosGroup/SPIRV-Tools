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

class Requires : public ::testing::TestWithParam<Capability> {
 public:
  Requires()
      : entry({nullptr,
               (Op)0,
               SPV_OPCODE_FLAGS_CAPABILITIES,
               GetParam(),
               0,
               {},
               false,
               false,
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

TEST(OpcodeRequiresCapability, None) {
  spv_opcode_desc_t entry = {
      nullptr, (Op)0, SPV_OPCODE_FLAGS_NONE, 0, 0, {}, false, false, {}};
  ASSERT_EQ(0, spvOpcodeRequiresCapabilities(&entry));
}

/// Test SPV_CAPBILITY_AS_MASK

TEST(CapabilityAsMaskMacro, Sample) {
  EXPECT_EQ(uint64_t(1), SPV_CAPABILITY_AS_MASK(spv::CapabilityMatrix));
  EXPECT_EQ(uint64_t(0x10000), SPV_CAPABILITY_AS_MASK(spv::CapabilityImageSRGBWrite));
  EXPECT_EQ(uint64_t(0x100000000ULL), SPV_CAPABILITY_AS_MASK(spv::CapabilityClipDistance));
  EXPECT_EQ(uint64_t(1) << 53, SPV_CAPABILITY_AS_MASK(spv::CapabilityTransformFeedback));
};

/// Capabilities required by an Opcode.
struct ExpectedOpCodeCapabilities {
  spv::Op opcode;
  uint64_t capabilities;  //< Bitfield of spv::Capability.
};

using OpcodeTableCapabilitiesTest =
    ::testing::TestWithParam<ExpectedOpCodeCapabilities>;

TEST_P(OpcodeTableCapabilitiesTest, TableEntryMatchesExpectedCapabilities) {
  spv_opcode_table opcodeTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
  spv_opcode_desc entry;
  ASSERT_EQ(SPV_SUCCESS,
            spvOpcodeTableValueLookup(opcodeTable, GetParam().opcode, &entry));
  EXPECT_EQ(GetParam().capabilities, entry->capabilities);
}

/// Translates a spv::Capability into a bitfield.
inline uint64_t mask(spv::Capability c) { return SPV_CAPABILITY_AS_MASK(c); }

/// Combines two spv::Capabilities into a bitfield.
inline uint64_t mask(spv::Capability c1, spv::Capability c2) {
  return SPV_CAPABILITY_AS_MASK(c1) | SPV_CAPABILITY_AS_MASK(c2);
}

INSTANTIATE_TEST_CASE_P(
    TableRowTest, OpcodeTableCapabilitiesTest,
    // Spot-check a few opcodes.
    ::testing::Values(
        ExpectedOpCodeCapabilities{
            spv::OpImageQuerySize,
            mask(spv::CapabilityKernel, spv::CapabilityImageQuery)},
        ExpectedOpCodeCapabilities{
            spv::OpImageQuerySizeLod,
            mask(spv::CapabilityKernel, spv::CapabilityImageQuery)},
        ExpectedOpCodeCapabilities{
            spv::OpImageQueryLevels,
            mask(spv::CapabilityKernel, spv::CapabilityImageQuery)},
        ExpectedOpCodeCapabilities{
            spv::OpImageQuerySamples,
            mask(spv::CapabilityKernel, spv::CapabilityImageQuery)},
        ExpectedOpCodeCapabilities{spv::OpImageSparseSampleImplicitLod,
                                   mask(spv::CapabilitySparseResidency)},
        ExpectedOpCodeCapabilities{spv::OpCopyMemorySized,
                                   mask(spv::CapabilityAddresses)},
        ExpectedOpCodeCapabilities{spv::OpArrayLength,
                                   mask(spv::CapabilityShader)},
        ExpectedOpCodeCapabilities{spv::OpFunction, 0},
        ExpectedOpCodeCapabilities{spv::OpConvertFToS, 0}));

// TODO(deki): test operand-table capabilities.

}  // anonymous namespace
