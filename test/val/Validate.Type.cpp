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

#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "UnitSPIRV.h"
#include "ValidateFixtures.h"

#include "gmock/gmock.h"


using std::make_pair;
using std::get;
using std::make_tuple;
using std::pair;
using std::string;
using std::stringstream;
using std::tie;
using std::tuple;
using std::vector;

using testing::MatchesRegex;

using spvtest::Concatenate;
using spvtest::MakeInstruction;
using spvtest::MakeVector;

namespace {
using Possible = vector<spv_type_category_t>;
using ValidateType = spvtest::ValidateBase<
    tuple<pair<spv_type_category_t, vector<uint32_t>>,
          pair<vector<uint32_t>, vector<spv_type_category_t>>>>;
using ValidateTypeAlias1 = spvtest::ValidateBase<tuple<string, string, string>>;
using ValidateTypeAlias2 = spvtest::ValidateBase<tuple<string, string, string>>;

INSTANTIATE_TEST_CASE_P(
    NonAliasableTypes, ValidateTypeAlias1,
    ::testing::Values(
        make_tuple("OpTypeVoid", "", ""), make_tuple("OpTypeBool", "", ""),
        make_tuple("OpTypeInt 32 0", "OpTypeInt 32 1", ""),
        make_tuple("OpTypeFloat 32", "OpTypeFloat 64", ""),
        make_tuple("OpTypeFloat 64", "OpTypeFloat 32", ""),
        make_tuple("OpTypeVector %uint 4", "OpTypeVector %uint 3",
                   "%uint = OpTypeInt 32 0"),
        make_tuple("OpTypeMatrix %vec 4", "OpTypeMatrix %vec 3",
                   "%uint = OpTypeInt 32 0\n%vec = OpTypeVector %uint 3"),
        make_tuple("OpTypeImage %float 2D 0 0 0 0 Unknown ReadOnly",
                   "OpTypeImage %float 2D 0 0 0 0 Unknown WriteOnly",
                   "%float = OpTypeFloat 32"),
        make_tuple("OpTypeSampler", "", ""),
        make_tuple("OpTypeSampledImage %img", "",
                   "%float = OpTypeFloat 32\n"
                   "%img = OpTypeImage %float 2D 0 0 0 0 Unknown WriteOnly"),
        make_tuple("OpTypePointer UniformConstant %float",
                   "OpTypePointer UniformConstant %int",
                   "%int = OpTypeInt 32 1\n"
                   "%float = OpTypeFloat 32\n"),
        make_tuple("OpTypeFunction %void %float", "OpTypeFunction %void %int",
                   "%void = OpTypeVoid\n"
                   "%float = OpTypeFloat 32\n"
                   "%int = OpTypeInt 32 1\n"),
        make_tuple("OpTypeEvent", "", ""),
        make_tuple("OpTypeDeviceEvent", "", ""),
        make_tuple("OpTypeReserveId", "", ""),
        make_tuple("OpTypeQueue", "", ""),
        make_tuple("OpTypePipe ReadOnly", "OpTypePipe WriteOnly", "")
        // TODO(umar) OpTypeForwardPointer
        ));

string MakeModule(tuple<string, string, string> param) {
  const string type = get<0>(param);
  const string alt_type = get<1>(param);
  const string prefix = get<2>(param);

  string out = R"(
      OpCapability Kernel
      OpCapability Matrix
      OpCapability Addresses
      OpCapability DeviceEnqueue
      OpCapability Pipes
      OpMemoryModel Physical64 OpenCL
      OpName %type "type"
      OpName %alias "alias"
)";

  if (!prefix.empty()) {
    out += prefix + "\n";
  }
  out += "%type = " + type + "\n";
  if (!alt_type.empty()) {
    out += "%alt = " + alt_type + "\n";
  }
  out += "%alias = " + type;

  return out;
}

TEST_P(ValidateTypeAlias1, AliasedNonAggregateTypeBad) {
  const string str = MakeModule(GetParam());

  CompileSuccessfully(str.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_TYPE, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      MatchesRegex("Type .\\[alias\\] is an alias to type .\\[type\\]"));
}

INSTANTIATE_TEST_CASE_P(
    AliasableTypes, ValidateTypeAlias2,
    ::testing::Values(make_tuple("OpTypeArray %uint %ten",
                                 "OpTypeArray %uint %nine",
                                 "%uint = OpTypeInt 32 0\n"
                                 "%nine = OpConstant %uint 9\n"
                                 "%ten = OpConstant %uint 10\n")
                      // TODO(umar): Add tests for runtime arrays and structs
                      ));

TEST_P(ValidateTypeAlias2, AliasedAggregateTypeGood) {
  const string str = MakeModule(GetParam());

  CompileSuccessfully(str.c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateType, AliasedAggregateTypeGood) {
  const char str[] = R"(
     OpCapability Shader
     OpMemoryModel Logical GLSL450
%uint  = OpTypeInt 32 0
%two   = OpConstant %uint 3
%iarr  = OpTypeArray %uint %two
%iarr2 = OpTypeArray %uint %two
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// Returns the expected SPIR-V module header words for the Khronos
// Assembler generator, and with a given Id bound.
std::vector<uint32_t> ExpectedHeaderForBound(uint32_t bound) {
  return {SpvMagicNumber, 0x10000,
          SPV_GENERATOR_WORD(SPV_GENERATOR_KHRONOS_ASSEMBLER, 0), bound, 0};
}

bool HasType(spv_type_category_t type, Possible types) {
  return count_if(begin(types), end(types), [type](spv_type_category_t val) {
      return (val & type);
    }) > 0;
}

INSTANTIATE_TEST_CASE_P(
    ResultTypes, ValidateType,
    ::testing::Combine(
        ::testing::Values(
            make_pair(SPV_TYPE_CATEGORY_TYPE_VOID,
                      MakeInstruction(SpvOpTypeVoid, {1})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_BOOL,
                      MakeInstruction(SpvOpTypeBool, {1})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_INT,
                      MakeInstruction(SpvOpTypeInt, {1, 32, 1})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_FLOAT,
                      MakeInstruction(SpvOpTypeFloat, {1, 32})),
            make_pair(
                SPV_TYPE_CATEGORY_TYPE_IMAGE,
                Concatenate({MakeInstruction(SpvOpTypeInt, {100, 32, 1}),
                             MakeInstruction(SpvOpTypeImage,
                                             {1, 100, SpvDim2D, 0, 0, 0, 0,
                                              SpvImageFormatUnknown,
                                              SpvAccessQualifierReadOnly})})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_SAMPLER,
                      MakeInstruction(SpvOpTypeSampler, {1})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_SAMPLEDIMAGE,
                      Concatenate(
                          {MakeInstruction(SpvOpTypeInt, {100, 32, 1}),
                           MakeInstruction(SpvOpTypeImage,
                                           {101, 100, SpvDim2D, 0, 0, 0, 0,
                                            SpvImageFormatUnknown,
                                            SpvAccessQualifierReadOnly}),
                           MakeInstruction(SpvOpTypeSampledImage, {1, 101})})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_OPAQUE,
                      MakeInstruction(SpvOpTypeOpaque,
                                      Concatenate({{1}, MakeVector("blah")}))),
            make_pair(
                SPV_TYPE_CATEGORY_TYPE_POINTER,
                Concatenate({MakeInstruction(SpvOpTypeInt, {100, 32, 1}),
                             MakeInstruction(SpvOpTypePointer,
                                             {1, SpvStorageClassUniformConstant,
                                              100})})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_EVENT,
                      MakeInstruction(SpvOpTypeEvent, {1})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_DEVICEEVENT,
                      MakeInstruction(SpvOpTypeDeviceEvent, {1})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_RESERVEID,
                      MakeInstruction(SpvOpTypeReserveId, {1})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_QUEUE,
                      MakeInstruction(SpvOpTypeQueue, {1})),
            make_pair(SPV_TYPE_CATEGORY_TYPE_PIPE,
                      MakeInstruction(SpvOpTypePipe,
                                      {1, SpvAccessQualifierReadOnly}))),

        ::testing::Values(
            // make_pair(MakeInstruction(SpvOpUndef, {1, 2}), SpvOpTypeBool),
            make_pair(MakeInstruction(SpvOpConstantTrue, {1, 2}),
                      Possible{SPV_TYPE_CATEGORY_TYPE_BOOL}),
            make_pair(MakeInstruction(SpvOpConstantFalse, {1, 2}),
                      Possible{SPV_TYPE_CATEGORY_TYPE_BOOL}),
            make_pair(
                Concatenate({MakeInstruction(SpvOpTypeInt, {200, 32, 0}),
                             MakeInstruction(SpvOpConstant, {200, 201, 0}),
                             MakeInstruction(SpvOpConstantComposite,
                                             {1, 2, 201, 201, 201, 201})}),
                Possible{SPV_TYPE_CATEGORY_COMPOSITE}),
            make_pair(MakeInstruction(SpvOpConstantSampler,
                                      {1, 2, SpvSamplerAddressingModeNone, 0,
                                       SpvSamplerFilterModeNearest}),
                      Possible{SPV_TYPE_CATEGORY_TYPE_SAMPLER}),
            make_pair(MakeInstruction(SpvOpConstantNull, {1, 2}),
                      Possible{SPV_TYPE_CATEGORY_SCALAR,
                               SPV_TYPE_CATEGORY_TYPE_VECTOR,
                               SPV_TYPE_CATEGORY_TYPE_POINTER,
                               SPV_TYPE_CATEGORY_TYPE_EVENT,
                               SPV_TYPE_CATEGORY_TYPE_DEVICEEVENT,
                               SPV_TYPE_CATEGORY_TYPE_RESERVEID,
                               SPV_TYPE_CATEGORY_TYPE_QUEUE,
                            SPV_TYPE_CATEGORY_COMPOSITE})
                )));

TEST_P(ValidateType, ResultType) {
  pair<vector<uint32_t>, Possible> test;
  Possible accept;
  vector<uint32_t> op;
  pair<spv_type_category_t, vector<uint32_t>> type;

  tie(type, test) = GetParam();
  tie(op, accept) = test;

  spv_result_t result = SPV_ERROR_INVALID_TYPE;
  // If the type is one of the acceptable types then set result to SPV_SUCCESS
  if (HasType(type.first, accept)) {
    result = SPV_SUCCESS;
  }

  auto bytecode = Concatenate(
      {ExpectedHeaderForBound(2),
       MakeInstruction(SpvOpCapability, {SpvCapabilityKernel}),
       MakeInstruction(SpvOpCapability, {SpvCapabilityAddresses}),
       MakeInstruction(SpvOpCapability, {SpvCapabilityDeviceEnqueue}),
       MakeInstruction(SpvOpCapability, {SpvCapabilityPipes}),
       MakeInstruction(SpvOpCapability, {SpvCapabilityLiteralSampler}),
       MakeInstruction(SpvOpMemoryModel,
                       {SpvAddressingModelPhysical64, SpvMemoryModelOpenCL}),
       type.second, op});

  SetBinary(std::move(bytecode));
  auto actual_result = ValidateInstructions();
  EXPECT_EQ(result, actual_result);
  if (actual_result != SPV_SUCCESS) {
    EXPECT_THAT(
        getDiagnosticString(),
        MatchesRegex("Opcode .+ requires the result type to be .+ found .+"));
  }
}

TEST_F(ValidateType, ConstantTrueBad) {
  const char str[] = R"(
     OpCapability Shader
     OpMemoryModel Logical GLSL450
%uint  = OpTypeInt 32 0
%true  = OpConstantTrue %uint
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_TYPE, ValidateInstructions());
}

// TODO(umar): Enable once data rules have been implemented
TEST_F(ValidateType, DISABLED_IntType8CapabilityGood) {
  const char str[] = R"(
     OpCapability Kernel
     OpCapability Addresses
     OpCapability Int8
     OpMemoryModel Physical64 OpenCL
     OpName %uint8 "uint8"
%uint8 = OpTypeInt 8 0
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// TODO(umar): Enable once data rules have been implemented
TEST_F(ValidateType, DISABLED_IntType8CapabilityBad) {
  const char str[] = R"(
     OpCapability Kernel
     OpCapability Addresses
     OpMemoryModel Physical64 OpenCL
     OpName %uint8 "uint8"
%uint8 = OpTypeInt 8 0
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_TYPE, ValidateInstructions());
}

}  // namespace
