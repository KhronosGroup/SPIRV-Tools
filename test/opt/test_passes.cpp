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

#include "source/opt/pass_manager.h"
#include "source/opt/passes.h"

namespace {

using namespace spvtools::opt;

TEST(Passes, StripDebugInfo) {
  std::vector<std::string> text = {
      "OpCapability Shader",
      "%1 = OpExtInstImport \"GLSL.std.450\"",
      "OpMemoryModel Logical GLSL450",
      "OpEntryPoint Vertex %2 \"main\"",
      "%3 = OpString \"minimal.vert\"",
      "OpSource ESSL 310",
      "OpSourceContinued \"I'm a happy shader! Yay! ;)\"",
      "OpSourceExtension \"save-the-world-extension\"",
      "OpModuleProcessed \"42\"",
      "OpName %2 \"main\"",
      "OpNoLine",
      "OpLine %3 10 10",
      "%4 = OpTypeVoid",
      "OpLine %3 100 100",
      "%5 = OpTypeFunction %4",
      "%2 = OpFunction %4 None %5",
      "OpLine %3 1 1",
      "OpNoLine",
      "OpLine %3 2 2",
      "OpLine %3 3 3",
      "%6 = OpLabel",
      "OpLine %3 4 4",
      "OpNoLine",
      "OpReturn",
      "OpFunctionEnd",
  };
  std::string input_text, expected_text;
  for (const auto& line : text) {
    input_text += line + "\n";
    if (line.find("OpLine") == std::string::npos &&
        line.find("OpNoLine") == std::string::npos &&
        line.find("OpName") == std::string::npos &&
        line.find("OpString") == std::string::npos &&
        line.find("OpSource") == std::string::npos &&
        line.find("OpSourceContinued") == std::string::npos &&
        line.find("OpSourceExtension") == std::string::npos &&
        line.find("OpModuleProcessed") == std::string::npos) {
      expected_text += line + "\n";
    }
  }

  std::unique_ptr<ir::Module> module = BuildSpv(Assemble(input_text));

  PassManager manager;
  {
    std::unique_ptr<Pass> pass(new DebugInfoRemovalPass);
    manager.AddPass(std::move(pass));
  }
  manager.run(module.get());

  std::vector<uint32_t> binary;
  module->ToBinary(&binary);

  EXPECT_EQ(expected_text, Disassemble(binary));
}

}  // anonymous namespace
