// Copyright (c) 2020 Google LLC
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

#include "gmock/gmock.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using PropagateDebugvalueTest = PassTest<::testing::Test>;

TEST_F(PropagateDebugvalueTest, SameImmediateDominator) {
  const std::string text = R"(
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
%ext = OpExtInstImport "OpenCL.DebugInfo.100"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%name = OpString "test"
OpName %main "main"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%float_1 = OpConstant %float 1
%uint = OpTypeInt 32 0
%uint_32 = OpConstant %uint 32
%null_expr = OpExtInst %void %ext DebugExpression
%src = OpExtInst %void %ext DebugSource %name
%cu = OpExtInst %void %ext DebugCompilationUnit 1 4 %src HLSL
%ty = OpExtInst %void %ext DebugTypeFunction FlagIsProtected|FlagIsPrivate %void
%dbg_main = OpExtInst %void %ext DebugFunction %name %ty %src 0 0 %cu %name FlagIsProtected|FlagIsPrivate 0 %main
%dbg_f = OpExtInst %void %ext DebugTypeBasic %name %uint_32 Float
; CHECK:      [[dbg_foo:%\w+]] = OpExtInst %void {{%\w+}} DebugLocalVariable
%dbg_foo = OpExtInst %void %ext DebugLocalVariable %name %dbg_f %src 0 0 %dbg_main FlagIsLocal
%main = OpFunction %void None %5
; CHECK:      [[bb1:%\w+]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_1
; CHECK-NEXT: OpSelectionMerge [[bb4:%\w+]]
; CHECK-NEXT: OpBranchConditional %true [[bb2:%\w+]] [[bb3:%\w+]]
; CHECK-NEXT: [[bb2]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_1
; CHECK-NEXT: OpBranch [[bb4]]
; CHECK-NEXT: [[bb3]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_1
; CHECK-NEXT: OpBranch [[bb4]]
; CHECK-NEXT: [[bb4]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_1
%17 = OpLabel
%decl0 = OpExtInst %void %ext DebugValue %dbg_foo %float_1 %null_expr
OpSelectionMerge %18 None
OpBranchConditional %true %19 %20
%19 = OpLabel
OpBranch %18
%20 = OpLabel
OpBranch %18
%18 = OpLabel
OpReturn
OpFunctionEnd
  )";

  SinglePassRunAndMatch<PropagateDebugvalue>(text, true);
}

TEST_F(PropagateDebugvalueTest, MultipleDebugValues) {
  const std::string text = R"(
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
%ext = OpExtInstImport "OpenCL.DebugInfo.100"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%name = OpString "test"
OpName %main "main"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%uint = OpTypeInt 32 0
%uint_32 = OpConstant %uint 32
%null_expr = OpExtInst %void %ext DebugExpression
%src = OpExtInst %void %ext DebugSource %name
%cu = OpExtInst %void %ext DebugCompilationUnit 1 4 %src HLSL
%ty = OpExtInst %void %ext DebugTypeFunction FlagIsProtected|FlagIsPrivate %void
%dbg_main = OpExtInst %void %ext DebugFunction %name %ty %src 0 0 %cu %name FlagIsProtected|FlagIsPrivate 0 %main
%dbg_f = OpExtInst %void %ext DebugTypeBasic %name %uint_32 Float
; CHECK:      [[dbg_foo:%\w+]] = OpExtInst %void {{%\w+}} DebugLocalVariable
%dbg_foo = OpExtInst %void %ext DebugLocalVariable %name %dbg_f %src 0 0 %dbg_main FlagIsLocal
%main = OpFunction %void None %5
; CHECK:      [[bb1:%\w+]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_1
; CHECK-NEXT: OpBranch [[bb3:%\w+]]
; CHECK-NEXT: [[bb2:%\w+]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_2
; CHECK-NEXT: OpBranch [[bb4:%\w+]]
; CHECK-NEXT: [[bb3]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_1
; CHECK-NEXT: OpBranch [[bb2]]
; CHECK-NEXT: [[bb4]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_2
%17 = OpLabel
%decl0 = OpExtInst %void %ext DebugValue %dbg_foo %float_1 %null_expr
OpBranch %20
%19 = OpLabel
%decl1 = OpExtInst %void %ext DebugValue %dbg_foo %float_2 %null_expr
OpBranch %18
%20 = OpLabel
OpBranch %19
%18 = OpLabel
OpReturn
OpFunctionEnd
  )";

  SinglePassRunAndMatch<PropagateDebugvalue>(text, false);
}

TEST_F(PropagateDebugvalueTest, MultipleDebugValuesInBB) {
  const std::string text = R"(
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
%ext = OpExtInstImport "OpenCL.DebugInfo.100"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
%name = OpString "test"
OpName %main "main"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%float_3 = OpConstant %float 3
%uint = OpTypeInt 32 0
%uint_32 = OpConstant %uint 32
%null_expr = OpExtInst %void %ext DebugExpression
%src = OpExtInst %void %ext DebugSource %name
%cu = OpExtInst %void %ext DebugCompilationUnit 1 4 %src HLSL
%ty = OpExtInst %void %ext DebugTypeFunction FlagIsProtected|FlagIsPrivate %void
%dbg_main = OpExtInst %void %ext DebugFunction %name %ty %src 0 0 %cu %name FlagIsProtected|FlagIsPrivate 0 %main
%dbg_f = OpExtInst %void %ext DebugTypeBasic %name %uint_32 Float
; CHECK:      [[dbg_foo:%\w+]] = OpExtInst %void {{%\w+}} DebugLocalVariable
%dbg_foo = OpExtInst %void %ext DebugLocalVariable %name %dbg_f %src 0 0 %dbg_main FlagIsLocal
%main = OpFunction %void None %5
; CHECK:      [[bb1:%\w+]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_1
; CHECK-NEXT: OpBranch [[bb3:%\w+]]
; CHECK-NEXT: [[bb2:%\w+]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_2
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_3
; CHECK-NEXT: OpBranch [[bb4:%\w+]]
; CHECK-NEXT: [[bb3]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_1
; CHECK-NEXT: OpBranch [[bb2]]
; CHECK-NEXT: [[bb4]] = OpLabel
; CHECK-NEXT: DebugValue [[dbg_foo]] %float_3
%17 = OpLabel
%decl0 = OpExtInst %void %ext DebugValue %dbg_foo %float_1 %null_expr
OpBranch %20
%19 = OpLabel
%decl1 = OpExtInst %void %ext DebugValue %dbg_foo %float_2 %null_expr
%decl2 = OpExtInst %void %ext DebugValue %dbg_foo %float_3 %null_expr
OpBranch %18
%20 = OpLabel
OpBranch %19
%18 = OpLabel
OpReturn
OpFunctionEnd
  )";

  SinglePassRunAndMatch<PropagateDebugvalue>(text, false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
