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

#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

#include "source/latest_version_spirv_header.h"
#include "source/opt/log.h"
#include "source/util/string_utils.h"
#include "spirv-tools/libspirv.hpp"
#include "tools/util/cli_consumer.h"

namespace {

constexpr auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_6;

// Extract a string literal from a given range of SPIR-V words.
// Copies all the characters up to the first '\0', while removing escape
// patterns. Not finding a '\0' before reaching `end` fails the extraction.
spv_result_t ExtractStringLiteral(const spv_position_t& loc,
                                  const uint32_t* begin, const uint32_t* end,
                                  std::string* output) {
  const size_t source_length =
      static_cast<size_t>(std::distance(begin, end)) * sizeof(uint32_t);
  const std::string source = spvtools::utils::MakeString(
      begin, end, /*assert_found_terminating_null=*/false);
  if (source.size() == source_length) {
    spvtools::Error(spvtools::utils::CLIMessageConsumer, "", loc,
                    "Missing NULL terminator for literal string.");
    return SPV_ERROR_INVALID_BINARY;
  }

  std::string unescaped_source;
  unescaped_source.reserve(source.size());
  for (size_t read_index = 0; read_index < source.size(); ++read_index) {
    if (source[read_index] == '\\' && read_index + 1 < source.size()) {
      ++read_index;
    }
    unescaped_source.push_back(source[read_index]);
  }

  output->append(unescaped_source);
  return SPV_SUCCESS;
}

spv_result_t extractOpString(const spv_position_t& loc,
                             const spv_parsed_instruction_t& instruction,
                             std::string* output) {
  assert(output != nullptr);
  assert(instruction.opcode == static_cast<unsigned>(spv::Op::OpString));
  if (instruction.num_operands != 2) {
    spvtools::Error(spvtools::utils::CLIMessageConsumer, "", loc,
                    "Missing operands for OpString.");
    return SPV_ERROR_INVALID_BINARY;
  }

  const auto& operand = instruction.operands[1];
  const uint32_t* string_begin = instruction.words + operand.offset;
  const uint32_t* string_end = string_begin + operand.num_words;
  return ExtractStringLiteral(loc, string_begin, string_end, output);
}

spv_result_t extractOpSourceContinued(
    const spv_position_t& loc, const spv_parsed_instruction_t& instruction,
    std::string* output) {
  assert(output != nullptr);
  assert(instruction.opcode ==
         static_cast<unsigned>(spv::Op::OpSourceContinued));
  if (instruction.num_operands != 1) {
    spvtools::Error(spvtools::utils::CLIMessageConsumer, "", loc,
                    "Missing operands for OpSourceContinued.");
    return SPV_ERROR_INVALID_BINARY;
  }

  const auto& operand = instruction.operands[0];
  const uint32_t* string_begin = instruction.words + operand.offset;
  const uint32_t* string_end = string_begin + operand.num_words;
  return ExtractStringLiteral(loc, string_begin, string_end, output);
}

spv_result_t extractOpSource(const spv_position_t& loc,
                             const spv_parsed_instruction_t& instruction,
                             spv::Id* filename, std::string* code) {
  assert(filename != nullptr && code != nullptr);
  assert(instruction.opcode == static_cast<unsigned>(spv::Op::OpSource));
  // OpCode [ Source Language | Version | File (optional) | Source (optional) ]
  if (instruction.num_words < 3) {
    spvtools::Error(spvtools::utils::CLIMessageConsumer, "", loc,
                    "Missing operands for OpSource.");
    return SPV_ERROR_INVALID_BINARY;
  }

  *filename = 0;
  *code = "";
  if (instruction.num_words < 4) {
    return SPV_SUCCESS;
  }
  *filename = instruction.words[3];

  if (instruction.num_words < 5) {
    return SPV_SUCCESS;
  }

  const uint32_t* string_begin = instruction.words + 4;
  const uint32_t* string_end = instruction.words + instruction.num_words;
  return ExtractStringLiteral(loc, string_begin, string_end, code);
}

}  // namespace

bool ExtractSourceFromModule(
    const std::vector<uint32_t>& binary,
    std::unordered_map<std::string, std::string>* output) {
  auto context = spvtools::SpirvTools(kDefaultEnvironment);
  context.SetMessageConsumer(spvtools::utils::CLIMessageConsumer);

  // There is nothing valuable in the header.
  spvtools::HeaderParser headerParser = [](const spv_endianness_t,
                                           const spv_parsed_header_t&) {
    return SPV_SUCCESS;
  };

  std::unordered_map<uint32_t, std::string> stringMap;
  std::vector<std::pair<spv::Id, std::string>> sources;
  spv::Op lastOpcode = spv::Op::Max;
  size_t instructionIndex = 0;

  spvtools::InstructionParser instructionParser =
      [&stringMap, &sources, &lastOpcode,
       &instructionIndex](const spv_parsed_instruction_t& instruction) {
        const spv_position_t loc = {0, 0, instructionIndex + 1};
        spv_result_t result = SPV_SUCCESS;

        if (instruction.opcode == static_cast<unsigned>(spv::Op::OpString)) {
          std::string content;
          result = extractOpString(loc, instruction, &content);
          if (result == SPV_SUCCESS) {
            stringMap.emplace(instruction.result_id, std::move(content));
          }
        } else if (instruction.opcode ==
                   static_cast<unsigned>(spv::Op::OpSource)) {
          spv::Id filenameId;
          std::string code;
          result = extractOpSource(loc, instruction, &filenameId, &code);
          if (result == SPV_SUCCESS) {
            sources.emplace_back(std::make_pair(filenameId, std::move(code)));
          }
        } else if (instruction.opcode ==
                   static_cast<unsigned>(spv::Op::OpSourceContinued)) {
          if (lastOpcode != spv::Op::OpSource &&
              lastOpcode != spv::Op::OpSourceContinued) {
            spvtools::Error(spvtools::utils::CLIMessageConsumer, "", loc,
                            "OpSourceContinued MUST follow an OpSource or "
                            "OpSourceContinued.");
            return SPV_ERROR_INVALID_BINARY;
          }

          assert(sources.size() > 0);
          result = extractOpSourceContinued(loc, instruction,
                                            &sources.back().second);
        }

        ++instructionIndex;
        lastOpcode = static_cast<spv::Op>(instruction.opcode);
        return result;
      };

  if (!context.Parse(binary, headerParser, instructionParser)) {
    return false;
  }

  std::string defaultName = "unnamed-";
  size_t unnamedCount = 0;
  for (auto & [ id, code ] : sources) {
    std::string filename;
    const auto it = stringMap.find(id);
    if (it == stringMap.cend() || it->second.empty()) {
      filename = "unnamed-" + std::to_string(unnamedCount) + ".hlsl";
      ++unnamedCount;
    } else {
      filename = it->second;
    }

    if (output->count(filename) != 0) {
      spvtools::Error(spvtools::utils::CLIMessageConsumer, "", {},
                      "Source file name conflict.");
      return false;
    }
    output->insert({filename, code});
  }

  return true;
}
