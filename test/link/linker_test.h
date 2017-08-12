// Copyright (c) 2017 Pierre Moreau
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

#ifndef LIBSPIRV_TEST_LINK_LINK_TEST
#define LIBSPIRV_TEST_LINK_LINK_TEST

#include <iostream>

#include "source/spirv_constant.h"
#include "unit_spirv.h"

#include "spirv-tools/linker.hpp"

namespace spvtest {

using Binary = std::vector<uint32_t>;
using Binaries = std::vector<Binary>;

class LinkerTest : public ::testing::Test {
 public:
  LinkerTest() : tools_(SPV_ENV_UNIVERSAL_1_2), linker_(SPV_ENV_UNIVERSAL_1_2),
        assemble_options_(spvtools::SpirvTools::kDefaultAssembleOption),
        disassemble_options_(spvtools::SpirvTools::kDefaultDisassembleOption) {
    const auto consumer = [this](spv_message_level_t level, const char*,
                                const spv_position_t& position,
                                const char* message) {
      switch (level) {
        case SPV_MSG_FATAL:
        case SPV_MSG_INTERNAL_ERROR:
        case SPV_MSG_ERROR:
          error_message_ = "ERROR";
          break;
        case SPV_MSG_WARNING:
          error_message_ = "WARNING";
          break;
        case SPV_MSG_INFO:
          error_message_ = "INFO";
          break;
        case SPV_MSG_DEBUG:
          error_message_ = "DEBUG";
          break;
      }
      error_message_ += ": " + std::to_string(position.index) + ": " + message;
    };
    tools_.SetMessageConsumer(consumer);
    linker_.SetMessageConsumer(consumer);
  }

  virtual void SetUp() {}
  virtual void TearDown() { error_message_.clear(); }

  spv_result_t Link(const std::vector<std::string>& bodies, spvtest::Binary& linked_binary, spvtools::LinkerOptions options = spvtools::LinkerOptions()) {
    spvtest::Binaries binaries(bodies.size());
    for (size_t i = 0u; i < bodies.size(); ++i)
      if (!tools_.Assemble(bodies[i], binaries.data() + i, assemble_options_))
        return SPV_ERROR_INVALID_TEXT;

    return linker_.Link(binaries, linked_binary, options);
  }

  spv_result_t Link(const spvtest::Binaries& binaries, spvtest::Binary& linked_binary, spvtools::LinkerOptions options = spvtools::LinkerOptions()) {
    return linker_.Link(binaries, linked_binary, options);
  }

  spv_result_t Disassemble(const spvtest::Binary& binary, std::string& text) {
    return tools_.Disassemble(binary, &text, disassemble_options_) ? SPV_SUCCESS : SPV_ERROR_INVALID_BINARY;
  }

  void SetAssembleOptions(uint32_t assemble_options) {
    assemble_options_ = assemble_options;
  }

  void SetDisassembleOptions(uint32_t disassemble_options) {
    disassemble_options_ = disassemble_options;
  }

  std::string GetErrorMessage() const {
    return error_message_;
  }

 private:
  spvtools::SpirvTools tools_;  // An instance for calling SPIRV-Tools functionalities.
  spvtools::Linker linker_;
  uint32_t assemble_options_;
  uint32_t disassemble_options_;
  std::string error_message_;
};

}  // namespace spvtest

#endif // LIBSPIRV_TEST_LINK_LINK_TEST
