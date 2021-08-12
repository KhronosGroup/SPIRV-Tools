// Copyright (c) 2021 Google LLC.
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

#include "source/lint/divergence_analysis.h"

#include <string>

#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace lint {
namespace {

void CLIMessageConsumer(spv_message_level_t level, const char*,
                        const spv_position_t& position, const char* message) {
  switch (level) {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
      std::cerr << "error: line " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_WARNING:
      std::cout << "warning: line " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_INFO:
      std::cout << "info: line " << position.index << ": " << message
                << std::endl;
      break;
    default:
      break;
  }
}

class DivergenceTest : public ::testing::Test {
 protected:
  std::unique_ptr<opt::IRContext> context_;
  std::unique_ptr<DivergenceAnalysis> divergence_;

  void Build(std::string text, uint32_t function_id = 1) {
    context_ = BuildModule(SPV_ENV_UNIVERSAL_1_0, CLIMessageConsumer, text,
                           SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    opt::Module* module = context_->module();
    EXPECT_NE(nullptr, module);
    // Module should have one function.
    ASSERT_NE(module->begin(), module->end());
    ASSERT_EQ(++module->begin(), module->end());
    opt::Function* function = &*module->begin();
    ASSERT_EQ(function->result_id(), function_id);
    divergence_.reset(new DivergenceAnalysis(*context_));
    divergence_->Run(function);
  }
};

// Makes assertions a bit shorter.
using Level = DivergenceAnalysis::DivergenceLevel;

TEST_F(DivergenceTest, SimpleTest) {
  Build(R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main" %x
	       OpExecutionMode %1 OriginLowerLeft
       %void = OpTypeVoid
     %void_f = OpTypeFunction %void
       %bool = OpTypeBool
      %float = OpTypeFloat 32
      %false = OpConstantFalse %bool
       %true = OpConstantTrue %bool
       %zero = OpConstant %float 0
        %x_t = OpTypePointer Input %float
          %x = OpVariable %x_t Input
          %1 = OpFunction %void None %void_f
         %10 = OpLabel
         %11 = OpLoad %float %x
         %12 = OpFOrdLessThan %bool %11 %zero
               OpSelectionMerge %14 None
               OpBranchConditional %12 %13 %14
         %13 = OpLabel
               OpBranch %14
         %14 = OpLabel
               OpReturn
               OpFunctionEnd
  )");
  // Control flow divergence.
  EXPECT_EQ(Level::kUniform, divergence_->GetDivergenceLevel(10));
  EXPECT_EQ(Level::kDivergent, divergence_->GetDivergenceLevel(13));
  EXPECT_EQ(12, divergence_->GetDivergenceSource(13));
  EXPECT_EQ(Level::kUniform, divergence_->GetDivergenceLevel(14));
  // Value divergence.
  EXPECT_EQ(Level::kDivergent, divergence_->GetDivergenceLevel(11));
  EXPECT_EQ(0, divergence_->GetDivergenceSource(11));
  EXPECT_EQ(Level::kDivergent, divergence_->GetDivergenceLevel(12));
  EXPECT_EQ(11, divergence_->GetDivergenceSource(12));
}

}  // namespace
}  // namespace lint
}  // namespace spvtools
