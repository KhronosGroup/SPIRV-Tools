// Copyright (c) 2023 Google LLC.
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

#include "extract_source.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "source/opt/log.h"
#include "spirv-tools/libspirv.hpp"
#include "tools/util/cli_consumer.h"

namespace {
constexpr auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_6;
}  // namespace

bool extract_source_from_module(
    const std::vector<uint32_t>& binary,
    std::unordered_map<std::string, std::string>* output) {
  auto context = spvtools::SpirvTools(kDefaultEnvironment);
  context.SetMessageConsumer(spvtools::utils::CLIMessageConsumer);

  spvtools::HeaderParser headerParser =
      [](const spv_endianness_t endianess,
         const spv_parsed_header_t& instruction) {
        (void)endianess;
        (void)instruction;
        return SPV_SUCCESS;
      };

  spvtools::InstructionParser instructionParser =
      [](const spv_parsed_instruction_t& instruction) {
        (void)instruction;
        return SPV_SUCCESS;
      };

  if (!context.Parse(binary, headerParser, instructionParser)) {
    return false;
  }

  // FIXME
  (void)output;
  return true;
}
