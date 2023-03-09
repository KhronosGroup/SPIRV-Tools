#include "extract_source.h"

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
