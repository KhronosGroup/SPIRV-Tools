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

#ifndef LIBSPIRV_TEST_OPT_OPT_TEST_COMMON_H_
#define LIBSPIRV_TEST_OPT_OPT_TEST_COMMON_H_

#include <memory>

#include <gtest/gtest.h>

#include "source/opt/constructs.h"
#include "source/opt/spv_builder.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace opt {

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

std::string Disassemble(const std::vector<uint32_t>& binary) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_text text = nullptr;
  spv_diagnostic diagnostic = nullptr;

  spv_result_t status = spvBinaryToText(context, binary.data(), binary.size(),
                                        0, &text, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, status) << "disassemble binary to text failed";
  std::string result(text->str, text->str + text->length);

  spvDiagnosticDestroy(diagnostic);
  spvTextDestroy(text);
  spvContextDestroy(context);

  // Strip comments at the begining of the disassembled text. We are sure that
  // they exist; so assert()s are used here.
  auto pos = result.find_last_of(';');
  assert(pos != std::string::npos);
  result = result.substr(pos + 1);
  pos = result.find('\n');
  assert(pos != std::string::npos);
  result = result.substr(pos + 1);

  return result;
}

spv_result_t SetSpvHeader(void* builder, spv_endianness_t, uint32_t magic,
                          uint32_t version, uint32_t generator,
                          uint32_t id_bound, uint32_t reserved) {
  reinterpret_cast<ir::SpvBuilder*>(builder)->SetModuleHeader(
      magic, version, generator, id_bound, reserved);
  return SPV_SUCCESS;
};

spv_result_t SetSpvInst(void* builder, const spv_parsed_instruction_t* inst) {
  reinterpret_cast<ir::SpvBuilder*>(builder)->AddInstruction(inst);
  return SPV_SUCCESS;
};

std::unique_ptr<ir::Module> BuildSpv(const std::vector<uint32_t>& binary) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_diagnostic diagnostic = nullptr;

  std::unique_ptr<ir::Module> module(new ir::Module);
  ir::SpvBuilder spv_builder(module.get());

  spv_result_t status =
      spvBinaryParse(context, &spv_builder, binary.data(), binary.size(),
                     SetSpvHeader, SetSpvInst, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, status) << "build ir::Module from binary failed";

  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);

  return module;
}

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_TEST_OPT_OPT_TEST_COMMON_H_
