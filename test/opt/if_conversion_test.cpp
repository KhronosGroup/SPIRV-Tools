// Copyright (c) 2018 Google LLC
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

#include "assembly_builder.h"
#include "gmock/gmock.h"
#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using IfConversionTest = PassTest<::testing::Test>;

#ifdef SPIRV_EFFCEE
TEST_F(IfConversionTest, TestSimplePhi) {
  const std::string text = R"(
; CHECK: OpSelectionMerge [[merge:%\w+]]
; CHECK: [[merge]] = OpLabel
; CHECK-NOT: OpPhi
; CHECK: [[sel:%\w+]] = OpSelect %int %true %uint_0 %uint_1
; CHECK OpStore {{%\w+}} [[sel]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %1 "func" %2
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%_ptr_Output_uint = OpTypePointer Output %uint
%_ptr_Function_uint = OpTypePointer Function %uint
%2 = OpVariable %_ptr_Output_uint Output
%11 = OpTypeFunction %void
%1 = OpFunction %void None %11
%12 = OpLabel
OpSelectionMerge %14 None
OpBranchConditional %true %15 %16
%15 = OpLabel
OpBranch %14
%16 = OpLabel
OpBranch %14
%14 = OpLabel
%18 = OpPhi %uint %uint_0 %15 %uint_1 %16
OpStore %2 %18
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::IfConversion>(text, true);
}
#endif  // SPIRV_EFFCEE

}  // anonymous namespace
