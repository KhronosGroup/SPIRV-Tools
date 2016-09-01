// Copyright (c) 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspirv.hpp"

#include "ir_loader.h"
#include "make_unique.h"

namespace spvtools {

namespace {

// Sets the module header. Meets the interface requirement of spvBinaryParse().
spv_result_t SetSpvHeader(void* builder, spv_endianness_t, uint32_t magic,
                          uint32_t version, uint32_t generator,
                          uint32_t id_bound, uint32_t reserved) {
  reinterpret_cast<ir::IrLoader*>(builder)
      ->SetModuleHeader(magic, version, generator, id_bound, reserved);
  return SPV_SUCCESS;
};

// Processes a parsed instruction. Meets the interface requirement of
// spvBinaryParse().
spv_result_t SetSpvInst(void* builder, const spv_parsed_instruction_t* inst) {
  reinterpret_cast<ir::IrLoader*>(builder)->AddInstruction(inst);
  return SPV_SUCCESS;
};

}  // annoymous namespace

spv_result_t SpvTools::Assemble(const std::string& text,
                                std::vector<uint32_t>* binary) {
  spv_binary spvbinary = nullptr;
  spv_diagnostic diagnostic = nullptr;

  spv_result_t status = spvTextToBinary(context_, text.data(), text.size(),
                                        &spvbinary, &diagnostic);
  if (status == SPV_SUCCESS) {
    binary->assign(spvbinary->code, spvbinary->code + spvbinary->wordCount);
  }

  spvDiagnosticDestroy(diagnostic);
  spvBinaryDestroy(spvbinary);

  return status;
}

spv_result_t SpvTools::Disassemble(const std::vector<uint32_t>& binary,
                                   std::string* text, uint32_t options) {
  spv_text spvtext = nullptr;
  spv_diagnostic diagnostic = nullptr;

  spv_result_t status = spvBinaryToText(context_, binary.data(), binary.size(),
                                        options, &spvtext, &diagnostic);
  if (status == SPV_SUCCESS) {
    text->assign(spvtext->str, spvtext->str + spvtext->length);
  }

  spvDiagnosticDestroy(diagnostic);
  spvTextDestroy(spvtext);

  return status;
}

std::unique_ptr<ir::Module> SpvTools::BuildModule(
    const std::vector<uint32_t>& binary) {
  spv_diagnostic diagnostic = nullptr;

  auto module = MakeUnique<ir::Module>();
  ir::IrLoader loader(module.get());

  spv_result_t status =
      spvBinaryParse(context_, &loader, binary.data(), binary.size(),
                     SetSpvHeader, SetSpvInst, &diagnostic);

  spvDiagnosticDestroy(diagnostic);

  loader.EndModule();

  if (status == SPV_SUCCESS) return module;
  return nullptr;
}

std::unique_ptr<ir::Module> SpvTools::BuildModule(const std::string& text) {
  std::vector<uint32_t> binary;
  if (Assemble(text, &binary) != SPV_SUCCESS) return nullptr;
  return BuildModule(binary);
}

}  // namespace spvtools
