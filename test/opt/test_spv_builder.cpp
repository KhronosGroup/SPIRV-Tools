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

#include "opt_test_common.h"

namespace {

using namespace spvtools::opt;

TEST(SpvBuilder, KeepLineDebugInfo) {
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

  std::unique_ptr<ir::Module> module = BuildSpv(Assemble(text));
  std::vector<uint32_t> binary;
  module->ToBinary(&binary, /*keep_nop = */ true);

  EXPECT_EQ(text, Disassemble(binary));
}

}  // anonymous namespace
