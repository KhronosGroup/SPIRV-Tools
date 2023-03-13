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

#include "tools/objdump/extract_source.h"

#include <gtest/gtest.h>

#include <string>

#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "spirv-tools/libspirv.hpp"
#include "tools/util/cli_consumer.h"

namespace {

constexpr auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_6;

std::pair<bool, std::unordered_map<std::string, std::string>> extractSource(
    const std::string& spv_source) {
  std::unique_ptr<spvtools::opt::IRContext> ctx = spvtools::BuildModule(
      kDefaultEnvironment, spvtools::utils::CLIMessageConsumer, spv_source,
      spvtools::SpirvTools::kDefaultAssembleOption |
          SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  std::vector<uint32_t> binary;
  ctx->module()->ToBinary(&binary, /* skip_nop = */ false);
  std::unordered_map<std::string, std::string> output;
  bool result = extract_source_from_module(binary, &output);
  return std::make_pair(result, std::move(output));
}

}  // namespace

TEST(ExtractSourceTest, no_debug) {
  std::string source = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
   %void = OpTypeVoid
      %2 = OpTypeFunction %void
   %bool = OpTypeBool
      %4 = OpUndef %bool
      %5 = OpFunction %void None %2
      %6 = OpLabel
           OpReturn
           OpFunctionEnd
  )";

  auto[success, result] = extractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 0);
}
