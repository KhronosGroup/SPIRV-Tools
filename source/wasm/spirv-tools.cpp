// Copyright (c) 2020 The Khronos Group Inc.
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

#include "spirv-tools/libspirv.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>
using namespace emscripten;

void print_msg_to_stderr (spv_message_level_t, const char*,
                          const spv_position_t&, const char* m) {
  std::cerr << "error: " << m << std::endl;
};

std::string dis(std::string const& buffer, uint32_t env, uint32_t options) {
  spvtools::SpirvTools core(static_cast<spv_target_env>(env));
  core.SetMessageConsumer(print_msg_to_stderr);

  std::vector<uint32_t> spirv;
  const uint32_t* ptr = reinterpret_cast<const uint32_t*>(buffer.data());
  spirv.assign(ptr, ptr + buffer.size() / 4);
  std::string disassembly;
  if (!core.Disassemble(spirv, &disassembly, options)) return "Error";
  return disassembly;
}

std::string as(std::string const& source, uint32_t env, uint32_t options) {
  spvtools::SpirvTools core(static_cast<spv_target_env>(env));
  core.SetMessageConsumer(print_msg_to_stderr);

  std::vector<uint32_t> spirv;
  if (!core.Assemble(source, &spirv, options)) spirv.clear();
  // Copy the data out.
  const auto* ptr = reinterpret_cast<const char*>(spirv.data());
  return std::string(ptr,spirv.size()*4);
}

EMSCRIPTEN_BINDINGS(my_module) {
  function("dis", &dis);
  function("as", &as);

#include "spv_env.inc"

  constant("SPV_BINARY_TO_TEXT_OPTION_NONE", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NONE));
  constant("SPV_BINARY_TO_TEXT_OPTION_PRINT", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_PRINT));
  constant("SPV_BINARY_TO_TEXT_OPTION_COLOR", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
  constant("SPV_BINARY_TO_TEXT_OPTION_INDENT", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT));
  constant("SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET));
  constant("SPV_BINARY_TO_TEXT_OPTION_NO_HEADER", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER));
  constant("SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES", static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES));

  constant("SPV_TEXT_TO_BINARY_OPTION_NONE", static_cast<uint32_t>(SPV_TEXT_TO_BINARY_OPTION_NONE));
  constant("SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS", static_cast<uint32_t>(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS));
}
