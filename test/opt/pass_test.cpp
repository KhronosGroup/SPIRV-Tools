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

#include "assembly_builder.h"
#include "opt/pass.h"
#include "pass_fixture.h"
#include "pass_utils.h"

namespace {
using namespace spvtools;
class DummyPass : public opt::Pass {
 public:
  const char* name() const override { return "dummy-pass"; }
  Status Process(ir::Module* module) override {
    return module ? Status::SuccessWithoutChange : Status::Failure;
  }
};
}  // namespace

namespace {

using namespace spvtools;
using ::testing::UnorderedElementsAre;

using PassClassTest = PassTest<::testing::Test>;

TEST_F(PassClassTest, BasicVisitFromEntryPoint) {
  // Make sure we visit the entry point, and the function it calls.
  // Do not visit Dead or Exported.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %10 "main" %gl_FragColor
               OpExecutionMode %10 OriginUpperLeft
               OpSource GLSL 150
               OpName %10 "main"
               OpName %Dead "Dead"
               OpName %11 "Constant"
               OpName %gl_FragColor "gl_FragColor"
               OpName %ExportedFunc "ExportedFunc"
               OpDecorate %gl_FragColor Location 0
               OpDecorate %ExportedFunc LinkageAttributes "ExportedFunc" Export
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
      %float = OpTypeFloat 32
          %8 = OpTypeFunction %float
  %float_0_5 = OpConstant %float 0.5
  %float_0_8 = OpConstant %float 0.8
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
    %float_1 = OpConstant %float 1
         %10 = OpFunction %void None %6
         %14 = OpLabel
         %15 = OpFunctionCall %float %11
         %16 = OpFunctionCall %float %11
         %17 = OpCompositeConstruct %v4float %15 %16 %float_0_8 %float_1
               OpStore %gl_FragColor %17
               OpReturn
               OpFunctionEnd
   %11 = OpFunction %float None %8
         %18 = OpLabel
               OpReturnValue %float_0_8
               OpFunctionEnd
       %Dead = OpFunction %float None %8
         %19 = OpLabel
               OpReturnValue %float_0_5
               OpFunctionEnd
%ExportedFunc = OpFunction %float None %9
         %20 = OpLabel
         %21 = OpFunctionCall %float %11
               OpReturnValue %16
               OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<ir::Module> module =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  DummyPass testPass;
  std::vector<uint32_t> processed;
  opt::Pass::ProcessFunction mark_visited = [&processed](ir::Function* fp) {
    processed.push_back(fp->result_id());
    return false;
  };
  testPass.ProcessEntryPointCallTree(mark_visited, module.get());
  EXPECT_THAT(processed, UnorderedElementsAre(10, 11));
}

TEST_F(PassClassTest, BasicVisitReachable) {
  // Make sure we visit the entry point, exported function, and the function
  // they call. Do not visit Dead.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %10 "main" %gl_FragColor
               OpExecutionMode %10 OriginUpperLeft
               OpSource GLSL 150
               OpName %10 "main"
               OpName %Dead "Dead"
               OpName %11 "Constant"
               OpName %gl_FragColor "gl_FragColor"
               OpName %12 "ExportedFunc"
               OpName %13 "Constant2"
               OpDecorate %gl_FragColor Location 0
               OpDecorate %12 LinkageAttributes "ExportedFunc" Export
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
      %float = OpTypeFloat 32
          %8 = OpTypeFunction %float
  %float_0_5 = OpConstant %float 0.5
  %float_0_8 = OpConstant %float 0.8
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
    %float_1 = OpConstant %float 1
         %10 = OpFunction %void None %6
         %14 = OpLabel
         %15 = OpFunctionCall %float %11
         %16 = OpFunctionCall %float %11
         %17 = OpCompositeConstruct %v4float %15 %16 %float_0_8 %float_1
               OpStore %gl_FragColor %17
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %float None %8
         %18 = OpLabel
               OpReturnValue %float_0_8
               OpFunctionEnd
       %Dead = OpFunction %float None %8
         %19 = OpLabel
               OpReturnValue %float_0_5
               OpFunctionEnd
         %12 = OpFunction %float None %9
         %20 = OpLabel
         %21 = OpFunctionCall %float %13
               OpReturnValue %21
               OpFunctionEnd
         %13 = OpFunction %float None %8
         %22 = OpLabel
               OpReturnValue %float_0_8
               OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<ir::Module> module =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  DummyPass testPass;
  std::vector<uint32_t> processed;
  opt::Pass::ProcessFunction mark_visited = [&processed](ir::Function* fp) {
    processed.push_back(fp->result_id());
    return false;
  };
  testPass.ProcessReachableCallTree(mark_visited, module.get());
  EXPECT_THAT(processed, UnorderedElementsAre(10, 11, 12, 13));
}

TEST_F(PassClassTest, BasicVisitOnlyOnce) {
  // Make sure we visit %11 only once, even if it is called from two different
  // functions.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %10 "main" %gl_FragColor
               OpExecutionMode %10 OriginUpperLeft
               OpSource GLSL 150
               OpName %10 "main"
               OpName %Dead "Dead"
               OpName %11 "Constant"
               OpName %gl_FragColor "gl_FragColor"
               OpName %12 "ExportedFunc"
               OpDecorate %gl_FragColor Location 0
               OpDecorate %12 LinkageAttributes "ExportedFunc" Export
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
      %float = OpTypeFloat 32
          %8 = OpTypeFunction %float
  %float_0_5 = OpConstant %float 0.5
  %float_0_8 = OpConstant %float 0.8
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
    %float_1 = OpConstant %float 1
         %10 = OpFunction %void None %6
         %14 = OpLabel
         %15 = OpFunctionCall %float %11
         %16 = OpFunctionCall %float %11
         %17 = OpCompositeConstruct %v4float %15 %16 %float_0_8 %float_1
               OpStore %gl_FragColor %17
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %float None %8
         %18 = OpLabel
               OpReturnValue %float_0_8
               OpFunctionEnd
       %Dead = OpFunction %float None %8
         %19 = OpLabel
               OpReturnValue %float_0_5
               OpFunctionEnd
         %12 = OpFunction %float None %9
         %20 = OpLabel
         %21 = OpFunctionCall %float %12
               OpReturnValue %21
               OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<ir::Module> module =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  DummyPass testPass;
  std::vector<uint32_t> processed;
  opt::Pass::ProcessFunction mark_visited = [&processed](ir::Function* fp) {
    processed.push_back(fp->result_id());
    return false;
  };
  testPass.ProcessReachableCallTree(mark_visited, module.get());
  EXPECT_THAT(processed, UnorderedElementsAre(10, 11, 12));
}

TEST_F(PassClassTest, BasicDontVisitExportedVariable) {
  // Make sure we only visit functions and not exported variables.
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %10 "main" %gl_FragColor
               OpExecutionMode %10 OriginUpperLeft
               OpSource GLSL 150
               OpName %10 "main"
               OpName %Dead "Dead"
               OpName %11 "Constant"
               OpName %12 "export_var"
               OpDecorate %12 LinkageAttributes "export_var" Export
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
      %float = OpTypeFloat 32
  %float_0_5 = OpConstant %float 0.5
         %12 = OpVariable %float Output
         %10 = OpFunction %void None %6
         %14 = OpLabel
               OpStore %12 %float_0_5
               OpReturn
               OpFunctionEnd
)";
  // clang-format on

  std::unique_ptr<ir::Module> module =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  DummyPass testPass;
  std::vector<uint32_t> processed;
  opt::Pass::ProcessFunction mark_visited = [&processed](ir::Function* fp) {
    processed.push_back(fp->result_id());
    return false;
  };
  testPass.ProcessReachableCallTree(mark_visited, module.get());
  EXPECT_THAT(processed, UnorderedElementsAre(10));
}
}  // namespace
