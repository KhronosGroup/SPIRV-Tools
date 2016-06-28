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

#include <gtest/gtest.h>

#include "opt/libspirv.hpp"

namespace {

using namespace spvtools;

TEST(IrBuilder, RoundTrip) {
  // #version 310 es
  // int add(int a, int b) { return a + b; }
  // void main() { add(1, 2); }
  const std::string text =
      "OpCapability Shader\n"
      "%1 = OpExtInstImport \"GLSL.std.450\"\n"
      "OpMemoryModel Logical GLSL450\n"
      "OpEntryPoint Vertex %2 \"main\"\n"
      "OpSource ESSL 310\n"
      "OpSourceExtension \"GL_GOOGLE_cpp_style_line_directive\"\n"
      "OpSourceExtension \"GL_GOOGLE_include_directive\"\n"
      "OpName %2 \"main\"\n"
      "OpName %3 \"add(i1;i1;\"\n"
      "OpName %4 \"a\"\n"
      "OpName %5 \"b\"\n"
      "OpName %6 \"param\"\n"
      "OpName %7 \"param\"\n"
      "%8 = OpTypeVoid\n"
      "%9 = OpTypeFunction %8\n"
      "%10 = OpTypeInt 32 1\n"
      "%11 = OpTypePointer Function %10\n"
      "%12 = OpTypeFunction %10 %11 %11\n"
      "%13 = OpConstant %10 1\n"
      "%14 = OpConstant %10 2\n"
      "%2 = OpFunction %8 None %9\n"
      "%15 = OpLabel\n"
      "%6 = OpVariable %11 Function\n"
      "%7 = OpVariable %11 Function\n"
      "OpStore %6 %13\n"
      "OpStore %7 %14\n"
      "%16 = OpFunctionCall %10 %3 %6 %7\n"
      "OpReturn\n"
      "OpFunctionEnd\n"
      "%3 = OpFunction %10 None %12\n"
      "%4 = OpFunctionParameter %11\n"
      "%5 = OpFunctionParameter %11\n"
      "%17 = OpLabel\n"
      "%18 = OpLoad %10 %4\n"
      "%19 = OpLoad %10 %5\n"
      "%20 = OpIAdd %10 %18 %19\n"
      "OpReturnValue %20\n"
      "OpFunctionEnd\n";

  SpvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::unique_ptr<ir::Module> module = t.BuildModule(text);
  ASSERT_NE(nullptr, module);

  std::vector<uint32_t> binary;
  module->ToBinary(&binary, /* skip_nop = */ false);

  std::string disassembled_text;
  EXPECT_EQ(SPV_SUCCESS, t.Disassemble(binary, &disassembled_text));
  EXPECT_EQ(text, disassembled_text);
}

TEST(IrBuilder, KeepLineDebugInfo) {
  // #version 310 es
  // void main() {}
  const std::string text =
      "OpCapability Shader\n"
      "%1 = OpExtInstImport \"GLSL.std.450\"\n"
      "OpMemoryModel Logical GLSL450\n"
      "OpEntryPoint Vertex %2 \"main\"\n"
      "%3 = OpString \"minimal.vert\"\n"
      "OpSource ESSL 310\n"
      "OpName %2 \"main\"\n"
      "OpLine %3 10 10\n"
      "%4 = OpTypeVoid\n"
      "OpLine %3 100 100\n"
      "%5 = OpTypeFunction %4\n"
      "%2 = OpFunction %4 None %5\n"
      "OpLine %3 1 1\n"
      "OpNoLine\n"
      "OpLine %3 2 2\n"
      "OpLine %3 3 3\n"
      "%6 = OpLabel\n"
      "OpLine %3 4 4\n"
      "OpNoLine\n"
      "OpReturn\n"
      "OpFunctionEnd\n";

  SpvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::unique_ptr<ir::Module> module = t.BuildModule(text);
  ASSERT_NE(nullptr, module);

  std::vector<uint32_t> binary;
  module->ToBinary(&binary, /* skip_nop = */ false);

  std::string disassembled_text;
  EXPECT_EQ(SPV_SUCCESS, t.Disassemble(binary, &disassembled_text));
  EXPECT_EQ(text, disassembled_text);
}

TEST(IrBuilder, LocalGlobalVariables) {
  // #version 310 es
  //
  // float gv1 = 10.;
  // float gv2 = 100.;
  //
  // float f() {
  //   float lv1 = gv1 + gv2;
  //   float lv2 = gv1 * gv2;
  //   return lv1 / lv2;
  // }
  //
  // void main() {
  //   float lv1 = gv1 - gv2;
  // }
  const std::string text =
      "OpCapability Shader\n"
      "%1 = OpExtInstImport \"GLSL.std.450\"\n"
      "OpMemoryModel Logical GLSL450\n"
      "OpEntryPoint Vertex %2 \"main\"\n"
      "OpSource ESSL 310\n"
      "OpName %2 \"main\"\n"
      "OpName %3 \"f(\"\n"
      "OpName %4 \"gv1\"\n"
      "OpName %5 \"gv2\"\n"
      "OpName %6 \"lv1\"\n"
      "OpName %7 \"lv2\"\n"
      "OpName %8 \"lv1\"\n"
      "%9 = OpTypeVoid\n"
      "%10 = OpTypeFunction %9\n"
      "%11 = OpTypeFloat 32\n"
      "%12 = OpTypeFunction %11\n"
      "%13 = OpTypePointer Private %11\n"
      "%4 = OpVariable %13 Private\n"
      "%14 = OpConstant %11 10\n"
      "%5 = OpVariable %13 Private\n"
      "%15 = OpConstant %11 100\n"
      "%16 = OpTypePointer Function %11\n"
      "%2 = OpFunction %9 None %10\n"
      "%17 = OpLabel\n"
      "%8 = OpVariable %16 Function\n"
      "OpStore %4 %14\n"
      "OpStore %5 %15\n"
      "%18 = OpLoad %11 %4\n"
      "%19 = OpLoad %11 %5\n"
      "%20 = OpFSub %11 %18 %19\n"
      "OpStore %8 %20\n"
      "OpReturn\n"
      "OpFunctionEnd\n"
      "%3 = OpFunction %11 None %12\n"
      "%21 = OpLabel\n"
      "%6 = OpVariable %16 Function\n"
      "%7 = OpVariable %16 Function\n"
      "%22 = OpLoad %11 %4\n"
      "%23 = OpLoad %11 %5\n"
      "%24 = OpFAdd %11 %22 %23\n"
      "OpStore %6 %24\n"
      "%25 = OpLoad %11 %4\n"
      "%26 = OpLoad %11 %5\n"
      "%27 = OpFMul %11 %25 %26\n"
      "OpStore %7 %27\n"
      "%28 = OpLoad %11 %6\n"
      "%29 = OpLoad %11 %7\n"
      "%30 = OpFDiv %11 %28 %29\n"
      "OpReturnValue %30\n"
      "OpFunctionEnd\n";

  SpvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::unique_ptr<ir::Module> module = t.BuildModule(text);
  ASSERT_NE(nullptr, module);

  std::vector<uint32_t> binary;
  module->ToBinary(&binary, /* skip_nop = */ false);

  std::string disassembled_text;
  EXPECT_EQ(SPV_SUCCESS, t.Disassemble(binary, &disassembled_text));
  EXPECT_EQ(text, disassembled_text);
}

}  // anonymous namespace
