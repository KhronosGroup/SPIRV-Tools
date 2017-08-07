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

namespace spvtools {

using Binary = std::vector<uint32_t>;
using Binaries = std::vector<Binary>;

class LinkerTest : public ::testing::Test {
 public:
  LinkerTest() : linker(SPV_ENV_UNIVERSAL_1_2) {
    linker.SetMessageConsumer([](spv_message_level_t level, const char*,
                                const spv_position_t& position,
                                const char* message) {
      switch (level) {
        case SPV_MSG_FATAL:
        case SPV_MSG_INTERNAL_ERROR:
        case SPV_MSG_ERROR:
          std::cerr << "error: " << position.index << ": " << message
                    << std::endl;
          break;
        case SPV_MSG_WARNING:
          std::cout << "warning: " << position.index << ": " << message
                    << std::endl;
          break;
        case SPV_MSG_INFO:
          std::cout << "info: " << position.index << ": " << message << std::endl;
          break;
        default:
          break;
      }
    });
  }

  spvtools::Linker linker;
};

}  // namespace spvtools

#endif // LIBSPIRV_TEST_LINK_LINK_TEST
