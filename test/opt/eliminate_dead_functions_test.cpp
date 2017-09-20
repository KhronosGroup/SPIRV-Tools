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
#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;
using ::testing::HasSubstr;

using EliminateDeadFunctionsBasicTest = PassTest<::testing::Test>;

TEST_F(EliminateDeadFunctionsBasicTest, BasicDeleteDeadFunction) {
  // The function Dead should be removed because it is never called.
  const std::vector<const char*> common_code = {
      // clang-format off
               "OpCapability Shader",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 150",
               "OpName %main \"main\"",
               "OpName %Dead \"Dead\"",
               "OpName %Constant \"Constant\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
               "OpDecorate %gl_FragColor Location 0",
       "%void = OpTypeVoid",
          "%7 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
          "%9 = OpTypeFunction %float",
  "%float_0_5 = OpConstant %float 0.5",
  "%float_0_8 = OpConstant %float 0.8",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
    "%float_1 = OpConstant %float 1",
       "%main = OpFunction %void None %7",
         "%15 = OpLabel",
         "%16 = OpFunctionCall %float %Constant",
         "%17 = OpFunctionCall %float %Constant",
         "%18 = OpCompositeConstruct %v4float %16 %17 %float_0_8 %float_1",
               "OpStore %gl_FragColor %18",
               "OpReturn",
               "OpFunctionEnd",
  "%Constant = OpFunction %float None %9",
         "%20 = OpLabel",
               "OpReturnValue %float_0_8",
               "OpFunctionEnd"
      // clang-format on
  };

  const std::vector<const char*> dead_function = {
      // clang-format off
      "%Dead = OpFunction %float None %9",
         "%19 = OpLabel",
               "OpReturnValue %float_0_5",
               "OpFunctionEnd",
      // clang-format on
  };

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<opt::EliminateDeadFunctionsPass>(
      JoinAllInsts(Concat(common_code, dead_function)),
      JoinAllInsts(common_code), /* skip_nop = */ true);
}

TEST_F(EliminateDeadFunctionsBasicTest, BasicKeepLiveFunction) {
  // Everything is reachable from an entry point, so no functions should be
  // deleted.
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 150",
               "OpName %main \"main\"",
               "OpName %Dead \"Dead\"",
               "OpName %Constant \"Constant\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
               "OpDecorate %gl_FragColor Location 0",
       "%void = OpTypeVoid",
          "%7 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
          "%9 = OpTypeFunction %float",
  "%float_0_5 = OpConstant %float 0.5",
  "%float_0_8 = OpConstant %float 0.8",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
    "%float_1 = OpConstant %float 1",
       "%main = OpFunction %void None %7",
         "%15 = OpLabel",
         "%16 = OpFunctionCall %float %Constant",
         "%17 = OpFunctionCall %float %Dead",
         "%18 = OpCompositeConstruct %v4float %16 %17 %float_0_8 %float_1",
               "OpStore %gl_FragColor %18",
               "OpReturn",
               "OpFunctionEnd",
      "%Dead = OpFunction %float None %9",
         "%19 = OpLabel",
               "OpReturnValue %float_0_5",
               "OpFunctionEnd",
  "%Constant = OpFunction %float None %9",
         "%20 = OpLabel",
               "OpReturnValue %float_0_8",
               "OpFunctionEnd"
      // clang-format on
  };

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  std::string assembly = JoinAllInsts(text);
  auto result = SinglePassRunAndDisassemble<opt::EliminateDeadFunctionsPass>(
      assembly, /* skip_nop = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(assembly, std::get<0>(result));
}

TEST_F(EliminateDeadFunctionsBasicTest, BasicKeepExportFunctions) {
  // All functions are reachable.  In particular, ExportedFunc and Constant are
  // reachable because ExportedFunc is exported.  Nothing should be removed.
  const std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
               "OpCapability Linkage",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 150",
               "OpName %main \"main\"",
               "OpName %ExportedFunc \"ExportedFunc\"",
               "OpName %Constant \"Constant\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
               "OpDecorate %gl_FragColor Location 0",
               "OpDecorate %ExportedFunc LinkageAttributes \"ExportedFunc\" Export",
       "%void = OpTypeVoid",
          "%7 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
          "%9 = OpTypeFunction %float",
  "%float_0_5 = OpConstant %float 0.5",
  "%float_0_8 = OpConstant %float 0.8",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
    "%float_1 = OpConstant %float 1",
       "%main = OpFunction %void None %7",
         "%15 = OpLabel",
         "%18 = OpCompositeConstruct %v4float %float_0_5 %float_0_5 %float_0_8 %float_1",
               "OpStore %gl_FragColor %18",
               "OpReturn",
               "OpFunctionEnd",
"%ExportedFunc = OpFunction %float None %9",
         "%19 = OpLabel",
         "%16 = OpFunctionCall %float %Constant",
               "OpReturnValue %16",
               "OpFunctionEnd",
  "%Constant = OpFunction %float None %9",
         "%20 = OpLabel",
               "OpReturnValue %float_0_8",
               "OpFunctionEnd"
      // clang-format on
  };

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  std::string assembly = JoinAllInsts(text);
  auto result = SinglePassRunAndDisassemble<opt::EliminateDeadFunctionsPass>(
      assembly, /* skip_nop = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(assembly, std::get<0>(result));
}
}  // anonymous namespace
