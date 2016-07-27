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
  const std::string text = R"asm(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpSource ESSL 310
OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
OpSourceExtension "GL_GOOGLE_include_directive"
OpName %main "main"
OpName %add_i1_i1_ "add(i1;i1;"
OpName %a "a"
OpName %b "b"
OpName %param "param"
OpName %param_0 "param"
%void = OpTypeVoid
%9 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%12 = OpTypeFunction %int %_ptr_Function_int %_ptr_Function_int
%13 = OpConstant %int 1
%14 = OpConstant %int 2
%main = OpFunction %void None %9
%15 = OpLabel
%param = OpVariable %_ptr_Function_int Function
%param_0 = OpVariable %_ptr_Function_int Function
OpStore %param %13
OpStore %param_0 %14
%16 = OpFunctionCall %int %add_i1_i1_ %param %param_0
OpReturn
OpFunctionEnd
%add_i1_i1_ = OpFunction %int None %12
%a = OpFunctionParameter %_ptr_Function_int
%b = OpFunctionParameter %_ptr_Function_int
%17 = OpLabel
%18 = OpLoad %int %a
%19 = OpLoad %int %b
%20 = OpIAdd %int %18 %19
OpReturnValue %20
OpFunctionEnd
)asm";

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
  const std::string text = R"asm(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
%3 = OpString "minimal.vert"
OpSource ESSL 310
OpName %main "main"
OpLine %3 10 10
%void = OpTypeVoid
OpLine %3 100 100
%5 = OpTypeFunction %void
%main = OpFunction %void None %5
OpLine %3 1 1
OpNoLine
OpLine %3 2 2
OpLine %3 3 3
%6 = OpLabel
OpLine %3 4 4
OpNoLine
OpReturn
OpFunctionEnd
)asm";

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
  const std::string text = R"asm(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpSource ESSL 310
OpName %main "main"
OpName %f_ "f("
OpName %gv1 "gv1"
OpName %gv2 "gv2"
OpName %lv1 "lv1"
OpName %lv2 "lv2"
OpName %lv1_0 "lv1"
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%12 = OpTypeFunction %float
%_ptr_Private_float = OpTypePointer Private %float
%gv1 = OpVariable %_ptr_Private_float Private
%14 = OpConstant %float 10
%gv2 = OpVariable %_ptr_Private_float Private
%15 = OpConstant %float 100
%_ptr_Function_float = OpTypePointer Function %float
%main = OpFunction %void None %10
%17 = OpLabel
%lv1_0 = OpVariable %_ptr_Function_float Function
OpStore %gv1 %14
OpStore %gv2 %15
%18 = OpLoad %float %gv1
%19 = OpLoad %float %gv2
%20 = OpFSub %float %18 %19
OpStore %lv1_0 %20
OpReturn
OpFunctionEnd
%f_ = OpFunction %float None %12
%21 = OpLabel
%lv1 = OpVariable %_ptr_Function_float Function
%lv2 = OpVariable %_ptr_Function_float Function
%22 = OpLoad %float %gv1
%23 = OpLoad %float %gv2
%24 = OpFAdd %float %22 %23
OpStore %lv1 %24
%25 = OpLoad %float %gv1
%26 = OpLoad %float %gv2
%27 = OpFMul %float %25 %26
OpStore %lv2 %27
%28 = OpLoad %float %lv1
%29 = OpLoad %float %lv2
%30 = OpFDiv %float %28 %29
OpReturnValue %30
OpFunctionEnd
)asm";

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
