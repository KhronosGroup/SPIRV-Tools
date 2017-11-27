// Copyright (c) 2017 Google Inc.
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

#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "../assembly_builder.h"
#include "../pass_fixture.h"
#include "../pass_utils.h"
#include "opt/dominator_analysis_pass.h"
#include "opt/pass.h"

namespace {

using namespace spvtools;
using ::testing::UnorderedElementsAre;

using PassClassTest = PassTest<::testing::Test>;

const ir::Function* getFromModule(ir::Module* module, uint32_t id) {
  for (ir::Function& F : *module) {
    if (F.result_id() == id) {
      return &F;
    }
  }
  return nullptr;
}

TEST_F(PassClassTest, BasicVisitFromEntryPoint) {
  const std::string text = R"(
        OpCapability Shader
        OpMemoryModel Logical GLSL450
        OpEntryPoint Fragment %0 "main"
        %1 = OpTypeVoid
        %2 = OpTypeFunction %1
        %3 = OpTypeBool
        %4 = OpTypeInt 32 1
        %5 = OpConstant %4 0
        %6 = OpConstantFalse %3
        %7 = OpConstantTrue %3
        %8 = OpConstant %4 1
        %0 = OpFunction %1 None %2
        %9 = OpLabel
        OpBranch %10
        %11 = OpLabel
        OpBranch %10
        %10 = OpLabel
        OpSwitch %5 %12 1 %13
        %12 = OpLabel
        OpBranch %14
        %13 = OpLabel
        OpBranch %14
        %15 = OpLabel
        OpReturn
        %14 = OpLabel
        OpBranchConditional %7 %10 %15
        OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  opt::DominatorAnalysis testPass;
  testPass.InitializeTree(*module);

  const ir::Function* F = getFromModule(module, 4);

  EXPECT_TRUE(testPass.Dominates(5, 18, F));
  EXPECT_TRUE(testPass.Dominates(5, 53, F));
  EXPECT_TRUE(testPass.Dominates(5, 19, F));
  EXPECT_TRUE(testPass.Dominates(5, 25, F));
  EXPECT_TRUE(testPass.Dominates(5, 29, F));
  EXPECT_TRUE(testPass.Dominates(5, 27, F));
  EXPECT_TRUE(testPass.Dominates(5, 26, F));
  EXPECT_TRUE(testPass.Dominates(5, 28, F));

  EXPECT_TRUE(testPass.StrictlyDominates(5, 18, F));
  EXPECT_TRUE(testPass.StrictlyDominates(5, 53, F));
  EXPECT_TRUE(testPass.StrictlyDominates(5, 19, F));
  EXPECT_TRUE(testPass.StrictlyDominates(5, 25, F));
  EXPECT_TRUE(testPass.StrictlyDominates(5, 29, F));
  EXPECT_TRUE(testPass.StrictlyDominates(5, 27, F));
  EXPECT_TRUE(testPass.StrictlyDominates(5, 26, F));
  EXPECT_TRUE(testPass.StrictlyDominates(5, 28, F));
}

}  // namespace
