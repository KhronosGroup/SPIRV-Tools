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

#include <iostream>

#include <gmock/gmock.h>

#include "assembly_builder.h"
#include "gmock/gmock.h"
#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;
using ir::Instruction;
using ir::IRContext;
using opt::PassManager;

using CopyPropArrayPassTest = PassTest<::testing::Test>;

#ifdef SPIRV_EFFCEE
TEST_F(CopyPropArrayPassTest, BasicPropagateArray) {
  const std::string before =
      R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_INDEX %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 600
OpName %type_MyCBuffer "type.MyCBuffer"
OpMemberName %type_MyCBuffer 0 "Data"
OpName %MyCBuffer "MyCBuffer"
OpName %main "main"
OpName %in_var_INDEX "in.var.INDEX"
OpName %out_var_SV_Target "out.var.SV_Target"
OpDecorate %_arr_v4float_uint_8 ArrayStride 16
OpMemberDecorate %type_MyCBuffer 0 Offset 0
OpDecorate %type_MyCBuffer Block
OpDecorate %in_var_INDEX Flat
OpDecorate %in_var_INDEX Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %MyCBuffer DescriptorSet 0
OpDecorate %MyCBuffer Binding 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%_arr_v4float_uint_8 = OpTypeArray %v4float %uint_8
%type_MyCBuffer = OpTypeStruct %_arr_v4float_uint_8
%_ptr_Uniform_type_MyCBuffer = OpTypePointer Uniform %type_MyCBuffer
%void = OpTypeVoid
%13 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_arr_v4float_uint_8_0 = OpTypeArray %v4float %uint_8
%_ptr_Function__arr_v4float_uint_8_0 = OpTypePointer Function %_arr_v4float_uint_8_0
%int_0 = OpConstant %int 0
%_ptr_Uniform__arr_v4float_uint_8 = OpTypePointer Uniform %_arr_v4float_uint_8
%_ptr_Function_v4float = OpTypePointer Function %v4float
%MyCBuffer = OpVariable %_ptr_Uniform_type_MyCBuffer Uniform
%in_var_INDEX = OpVariable %_ptr_Input_int Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
; CHECK: OpFunction
%main = OpFunction %void None %13
; CHECK: OpLabel
%22 = OpLabel
; CHECK: OpVariable
%23 = OpVariable %_ptr_Function__arr_v4float_uint_8_0 Function
; CHECK: [[new_address:%\w+]] = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %uint_0
%24 = OpLoad %int %in_var_INDEX
%25 = OpAccessChain %_ptr_Uniform__arr_v4float_uint_8 %MyCBuffer %int_0
%26 = OpLoad %_arr_v4float_uint_8 %25
%27 = OpCompositeExtract %v4float %26 0
%28 = OpCompositeExtract %v4float %26 1
%29 = OpCompositeExtract %v4float %26 2
%30 = OpCompositeExtract %v4float %26 3
%31 = OpCompositeExtract %v4float %26 4
%32 = OpCompositeExtract %v4float %26 5
%33 = OpCompositeExtract %v4float %26 6
%34 = OpCompositeExtract %v4float %26 7
%35 = OpCompositeConstruct %_arr_v4float_uint_8_0 %27 %28 %29 %30 %31 %32 %33 %34
OpStore %23 %35
%36 = OpAccessChain %_ptr_Function_v4float %23 %24
; CHECK %37 = OpLoad %v4float [[new_address]]
%37 = OpLoad %v4float %36
OpStore %out_var_SV_Target %37
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  SinglePassRunAndMatch<opt::CopyPropagateArrays>(before, false);
}

#endif  // SPIRV_EFFCEE
}  // namespace
