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

#include <gtest/gtest.h>

#include "source/opt/ir_loader.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace opt {

namespace {

// Sets the module header. Meets the interface requirement of spvBinaryParse().
spv_result_t SetSpvHeader(void* builder, spv_endianness_t, uint32_t magic,
                          uint32_t version, uint32_t generator,
                          uint32_t id_bound, uint32_t reserved) {
  reinterpret_cast<ir::IrLoader*>(builder)->SetModuleHeader(
      magic, version, generator, id_bound, reserved);
  return SPV_SUCCESS;
};

// Processes a parsed instruction. Meets the interface requirement of
// spvBinaryParse().
spv_result_t SetSpvInst(void* builder, const spv_parsed_instruction_t* inst) {
  reinterpret_cast<ir::IrLoader*>(builder)->AddInstruction(inst);
  return SPV_SUCCESS;
};

}  // annoymous namespace

// Assembles the given assembly |text| and returns the binary.
std::vector<uint32_t> Assemble(const std::string& text) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;

  spv_result_t status =
      spvTextToBinary(context, text.data(), text.size(), &binary, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, status) << "assemble text to binary failed";
  std::vector<uint32_t> result(binary->code, binary->code + binary->wordCount);

  spvDiagnosticDestroy(diagnostic);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);

  return result;
}

// Disassembles the given SPIR-V |binary| and returns the assembly.
std::string Disassemble(const std::vector<uint32_t>& binary) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_text text = nullptr;
  spv_diagnostic diagnostic = nullptr;

  spv_result_t status =
      spvBinaryToText(context, binary.data(), binary.size(),
                      SPV_BINARY_TO_TEXT_OPTION_NO_HEADER, &text, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, status) << "disassemble binary to text failed";
  std::string result(text->str, text->str + text->length);

  spvDiagnosticDestroy(diagnostic);
  spvTextDestroy(text);
  spvContextDestroy(context);

  return result;
}

// Builds and returns a Module for the given SPIR-V |binary|.
std::unique_ptr<ir::Module> BuildModule(const std::vector<uint32_t>& binary) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_diagnostic diagnostic = nullptr;

  std::unique_ptr<ir::Module> module(new ir::Module);
  ir::IrLoader builder(module.get());

  spv_result_t status =
      spvBinaryParse(context, &builder, binary.data(), binary.size(),
                     SetSpvHeader, SetSpvInst, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, status) << "build ir::Module from binary failed";

  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);

  builder.EndModule();

  return module;
}

}  // namespace opt
}  // namespace spvtools
