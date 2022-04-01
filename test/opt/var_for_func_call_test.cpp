// Copyright (c) 2022 Google LLC.
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
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using VarFuncCallTest = PassTest<::testing::Test>;
TEST_F(VarFuncCallTest, Simple) {
  const std::string predef = R"(OpCapability SampledBuffer
OpCapability ImageBuffer
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpSource HLSL 630
OpName %type_buffer_image "type.buffer.image"
OpName %output "output"
OpName %main "main"
OpName %color "color"
OpName %bb_entry "bb.entry"
OpName %T "T"
OpMemberName %T 0 "t0"
OpMemberName %T 1 "t1"
OpName %t "t"
OpName %fn "fn"
OpName %p0 "p0"
OpName %p1 "p1"
OpName %bb_entry_0 "bb.entry"
OpDecorate %main LinkageAttributes "main" Export
OpDecorate %output DescriptorSet 0
OpDecorate %output Binding 1
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%float = OpTypeFloat 32
%type_buffer_image = OpTypeImage %float Buffer 2 0 0 2 Rgba32f
%_ptr_UniformConstant_type_buffer_image = OpTypePointer UniformConstant %type_buffer_image
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
)";

  const std::string before =
      R"(%9 = OpTypeFunction %int %_ptr_Function_v4float
%uint = OpTypeInt 32 0
%T = OpTypeStruct %float %uint
%_ptr_Function_T = OpTypePointer Function %T
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Function_uint = OpTypePointer Function %uint
%void = OpTypeVoid
%25 = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_uint
%output = OpVariable %_ptr_UniformConstant_type_buffer_image UniformConstant
%main = OpFunction %int None %9
%color = OpFunctionParameter %_ptr_Function_v4float
%bb_entry = OpLabel
%t = OpVariable %_ptr_Function_T Function
%19 = OpAccessChain %_ptr_Function_float %t %int_0
%21 = OpAccessChain %_ptr_Function_uint %t %int_1
%23 = OpFunctionCall %void %fn %19 %21
OpReturnValue %int_1
OpFunctionEnd
%fn = OpFunction %void DontInline %25
%p0 = OpFunctionParameter %_ptr_Function_float
%p1 = OpFunctionParameter %_ptr_Function_uint
%bb_entry_0 = OpLabel
OpReturn
OpFunctionEnd)";

  const std::string after = R"(%19 = OpTypeFunction %int %_ptr_Function_v4float
%uint = OpTypeInt 32 0
%T = OpTypeStruct %float %uint
%_ptr_Function_T = OpTypePointer Function %T
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Function_uint = OpTypePointer Function %uint
%void = OpTypeVoid
%25 = OpTypeFunction %void %_ptr_Function_float %_ptr_Function_uint
%output = OpVariable %_ptr_UniformConstant_type_buffer_image UniformConstant
%main = OpFunction %int None %19
%color = OpFunctionParameter %_ptr_Function_v4float
%bb_entry = OpLabel
%29 = OpVariable %_ptr_Function_float Function
%32 = OpVariable %_ptr_Function_uint Function
%t = OpVariable %_ptr_Function_T Function
%26 = OpAccessChain %_ptr_Function_float %t %int_0
%27 = OpAccessChain %_ptr_Function_uint %t %int_1
%30 = OpLoad %float %26
OpStore %29 %30
%33 = OpLoad %uint %27
OpStore %32 %33
%28 = OpFunctionCall %void %fn %29 %32
%31 = OpLoad %float %29
OpStore %26 %31
%34 = OpLoad %uint %32
OpStore %27 %34
OpReturnValue %int_1
OpFunctionEnd
%fn = OpFunction %void DontInline %25
%p0 = OpFunctionParameter %_ptr_Function_float
%p1 = OpFunctionParameter %_ptr_Function_uint
%bb_entry_0 = OpLabel
OpReturn
OpFunctionEnd
)";
  SinglePassRunAndCheck<VarsForFunctionCallPass>(predef+before, predef+after, true, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools