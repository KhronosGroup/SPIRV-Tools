// Copyright (c) 2016 Google Inc.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "source/opt/constructs.h"
#include "source/opt/type_builder.h"

namespace {

using namespace spvtools::opt;
using ::testing::ContainerEq;

spv_result_t ProcessHead(void*, spv_endianness_t, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t) {
  return SPV_SUCCESS;
}

spv_result_t ProcessInst(void* insts, const spv_parsed_instruction_t* inst) {
  reinterpret_cast<std::vector<ir::Inst>*>(insts)->push_back(ir::Inst(*inst));
  return SPV_SUCCESS;
}

std::vector<ir::Inst> AssemblyTextToInsts(const std::string& text) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_binary binary;
  spv_diagnostic diagnostic;

  std::vector<ir::Inst> insts;
  spv_result_t result =
      spvTextToBinary(context, text.data(), text.size(), &binary, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, result) << "assemble text to binary failed";
  result = spvBinaryParse(context, &insts, binary->code, binary->wordCount,
                          ProcessHead, ProcessInst, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, result) << "build ir::Insts from binary failed";

  spvDiagnosticDestroy(diagnostic);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);

  return insts;
}

TEST(TypeBuilder, Types) {
  const std::string text = R"(
    %void  = OpTypeVoid
    %bool  = OpTypeBool
    %u32   = OpTypeInt 32 0
    %id4   = OpConstant %u32 4
    %s32   = OpTypeInt 32 1
    %f64   = OpTypeFloat 64
    %v3u32 = OpTypeVector %u32 3
    %m3x3  = OpTypeMatrix %v3u32 3
    %a5u32 = OpTypeArray %u32 %id4
    %af64  = OpTypeRuntimeArray %f64
    %st1   = OpTypeStruct %u32
    %st2   = OpTypeStruct %f64 %s32 %v3u32
    %p     = OpTypePointer Uniform %st1
    %f     = OpTypeFunction %void %u32 %u32
  )";
  const auto insts = AssemblyTextToInsts(text);

  type::IdToTypeMap type_map;
  type::TypeBuilder builder(&type_map);

  std::vector<std::string> strs, expected_strs;
  for (const auto& inst : insts) {
    if (auto* type = builder.CreateType(inst)) strs.push_back(type->str());
  }
  expected_strs = {
      "void",
      "bool",
      "uint32",
      "sint32",
      "float64",
      "<uint32, 3>",
      "<<uint32, 3>, 3>",
      "[uint32, id(4)]",
      "[float64]",
      "{uint32}",
      "{float64, sint32, <uint32, 3>}",
      "{uint32}*",
      "(uint32, uint32) -> void",
  };

  EXPECT_EQ(expected_strs.size(), type_map.size());
  EXPECT_THAT(strs, ContainerEq(expected_strs));
}

}  // anonymous namespace
