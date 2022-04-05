// Copyright (c) 2022 Google LLC
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

#include "gmock/gmock.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using FlattenArrayMatrixStageVarTest = PassTest<::testing::Test>;

TEST_F(FlattenArrayMatrixStageVarTest, FlattenArrayMatrixStageVar) {
  // HLSL code:
  //
  //    struct HSPatchConstData {
  //      float tessFactor[3] : SV_TessFactor;
  //      float insideTessFactor[1] : SV_InsideTessFactor;
  //    };
  //
  //    struct HSCtrlPt {
  //      float2   a[3] : A;
  //      double   b    : B;
  //      float2   c[2] : C;
  //      float    d[2] : D;
  //      float2x3 e    : E;
  //    };
  //
  //    HSPatchConstData HSPatchConstantFunc(
  //        const OutputPatch<HSCtrlPt, 3> input) {
  //      HSPatchConstData data;
  //      data.tessFactor[0] = 3.0;
  //      data.tessFactor[1] = 3.0;
  //      data.tessFactor[2] = 3.0;
  //      data.insideTessFactor[0] = 3.0;
  //      return data;
  //    }
  //
  //    [domain("tri")]
  //    [partitioning("fractional_odd")]
  //    [outputtopology("triangle_cw")]
  //    [outputcontrolpoints(3)]
  //    [patchconstantfunc("HSPatchConstantFunc")]
  //    [maxtessfactor(15)]
  //    HSCtrlPt main(InputPatch<HSCtrlPt, 3> input,
  //                  uint CtrlPtID : SV_OutputControlPointID) {
  //      HSCtrlPt data;
  //      data.a[1] = input[CtrlPtID].c[CtrlPtID];
  //      data.d = input[CtrlPtID].d;
  //      return data;
  //    }
  const std::string header = R"(
               OpCapability Tessellation
               OpCapability Float64
               OpMemoryModel Logical GLSL450
               OpEntryPoint TessellationControl %main "main" %in_var_A %in_var_B %in_var_C %in_var_D %in_var_E %gl_InvocationID %out_var_A %out_var_B %out_var_C %out_var_D %out_var_E %gl_TessLevelOuter %gl_TessLevelInner
               OpExecutionMode %main Triangles
               OpExecutionMode %main SpacingFractionalOdd
               OpExecutionMode %main VertexOrderCw
               OpExecutionMode %main OutputVertices 4
               OpSource HLSL 600
               OpName %in_var_A "in.var.A"
               OpName %in_var_B "in.var.B"
               OpName %in_var_C "in.var.C"
               OpName %in_var_D "in.var.D"
               OpName %in_var_E "in.var.E"
               OpName %out_var_A "out.var.A"
               OpName %out_var_B "out.var.B"
               OpName %out_var_C "out.var.C"
               OpName %out_var_D "out.var.D"
               OpName %out_var_E "out.var.E"

; CHECK-DAG: OpName %in_var_A "in.var.A"
; CHECK-DAG: OpName %in_var_A_0 "in.var.A"
; CHECK-DAG: OpName %in_var_A_1 "in.var.A"
; CHECK-DAG: OpName %in_var_C "in.var.C"
; CHECK-DAG: OpName %in_var_C_0 "in.var.C"
; CHECK-DAG: OpName %in_var_D "in.var.D"
; CHECK-DAG: OpName %in_var_D_0 "in.var.D"
; CHECK-DAG: OpName %in_var_E "in.var.E"
; CHECK-DAG: OpName %in_var_E_0 "in.var.E"
; CHECK-DAG: OpName %out_var_A "out.var.A"
; CHECK-DAG: OpName %out_var_A_0 "out.var.A"
; CHECK-DAG: OpName %out_var_A_1 "out.var.A"
; CHECK-DAG: OpName %out_var_C "out.var.C"
; CHECK-DAG: OpName %out_var_C_0 "out.var.C"
; CHECK-DAG: OpName %out_var_D "out.var.D"
; CHECK-DAG: OpName %out_var_D_0 "out.var.D"
; CHECK-DAG: OpName %out_var_E "out.var.E"
; CHECK-DAG: OpName %out_var_E_0 "out.var.E"

               OpName %main "main"
               OpName %HSCtrlPt "HSCtrlPt"
               OpMemberName %HSCtrlPt 0 "a"
               OpMemberName %HSCtrlPt 1 "b"
               OpMemberName %HSCtrlPt 2 "c"
               OpMemberName %HSCtrlPt 3 "d"
               OpMemberName %HSCtrlPt 4 "e"
               OpName %param_var_input "param.var.input"
               OpName %param_var_CtrlPtID "param.var.CtrlPtID"
               OpName %temp_var_hullMainRetVal "temp.var.hullMainRetVal"
               OpName %if_true "if.true"
               OpName %HSPatchConstData "HSPatchConstData"
               OpMemberName %HSPatchConstData 0 "tessFactor"
               OpMemberName %HSPatchConstData 1 "insideTessFactor"
               OpName %if_merge "if.merge"
               OpName %HSPatchConstantFunc "HSPatchConstantFunc"
               OpName %input "input"
               OpName %bb_entry "bb.entry"
               OpName %data "data"
               OpName %src_main "src.main"
               OpName %input_0 "input"
               OpName %CtrlPtID "CtrlPtID"
               OpName %bb_entry_0 "bb.entry"
               OpName %data_0 "data"
               OpDecorate %gl_InvocationID BuiltIn InvocationId
               OpDecorate %gl_TessLevelOuter BuiltIn TessLevelOuter
               OpDecorate %gl_TessLevelOuter Patch
               OpDecorate %gl_TessLevelInner BuiltIn TessLevelInner
               OpDecorate %gl_TessLevelInner Patch
               OpDecorate %in_var_A Location 0
               OpDecorate %in_var_B Location 0
               OpDecorate %in_var_B Component 2
               OpDecorate %in_var_C Location 1
               OpDecorate %in_var_C Component 2
               OpDecorate %in_var_D Location 3
               OpDecorate %in_var_E Location 3
               OpDecorate %in_var_E Component 1
               OpDecorate %out_var_A Location 0
               OpDecorate %out_var_B Location 0
               OpDecorate %out_var_B Component 2
               OpDecorate %out_var_C Location 1
               OpDecorate %out_var_C Component 2
               OpDecorate %out_var_D Location 3
               OpDecorate %out_var_E Location 3
               OpDecorate %out_var_E Component 1

; CHECK-DAG: OpDecorate %in_var_A Location 0
; CHECK-DAG: OpDecorate %in_var_A Component 0
; CHECK-DAG: OpDecorate %in_var_A_0 Location 1
; CHECK-DAG: OpDecorate %in_var_A_0 Component 0
; CHECK-DAG: OpDecorate %in_var_A_1 Location 2
; CHECK-DAG: OpDecorate %in_var_A_1 Component 0

; CHECK-DAG: OpDecorate %in_var_C Location 1
; CHECK-DAG: OpDecorate %in_var_C Component 2
; CHECK-DAG: OpDecorate %in_var_C_0 Location 2
; CHECK-DAG: OpDecorate %in_var_C_0 Component 2

; CHECK-DAG: OpDecorate %in_var_D Location 3
; CHECK-DAG: OpDecorate %in_var_D Component 0
; CHECK-DAG: OpDecorate %in_var_D_0 Location 4
; CHECK-DAG: OpDecorate %in_var_D_0 Component 0

; CHECK-DAG: OpDecorate %in_var_E Location 3
; CHECK-DAG: OpDecorate %in_var_E Component 1
; CHECK-DAG: OpDecorate %in_var_E_0 Location 4
; CHECK-DAG: OpDecorate %in_var_E_0 Component 1

; CHECK-DAG: OpDecorate %out_var_A Location 0
; CHECK-DAG: OpDecorate %out_var_A Component 0
; CHECK-DAG: OpDecorate %out_var_A_0 Location 1
; CHECK-DAG: OpDecorate %out_var_A_0 Component 0
; CHECK-DAG: OpDecorate %out_var_A_1 Location 2
; CHECK-DAG: OpDecorate %out_var_A_1 Component 0

; CHECK-DAG: OpDecorate %out_var_C Location 1
; CHECK-DAG: OpDecorate %out_var_C Component 2
; CHECK-DAG: OpDecorate %out_var_C_0 Location 2
; CHECK-DAG: OpDecorate %out_var_C_0 Component 2

; CHECK-DAG: OpDecorate %out_var_D Location 3
; CHECK-DAG: OpDecorate %out_var_D Component 0
; CHECK-DAG: OpDecorate %out_var_D_0 Location 4
; CHECK-DAG: OpDecorate %out_var_D_0 Component 0

; CHECK-DAG: OpDecorate %out_var_E Location 3
; CHECK-DAG: OpDecorate %out_var_E Component 1
; CHECK-DAG: OpDecorate %out_var_E_0 Location 4
; CHECK-DAG: OpDecorate %out_var_E_0 Component 1

       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
     %uint_4 = OpConstant %uint 4
      %float = OpTypeFloat 32
    %float_3 = OpConstant %float 3
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
      %int_2 = OpConstant %int 2
      %int_3 = OpConstant %int 3
    %v2float = OpTypeVector %float 2
%_arr_v2float_uint_3 = OpTypeArray %v2float %uint_3
%_arr__arr_v2float_uint_3_uint_4 = OpTypeArray %_arr_v2float_uint_3 %uint_4
%_ptr_Input__arr__arr_v2float_uint_3_uint_4 = OpTypePointer Input %_arr__arr_v2float_uint_3_uint_4
     %double = OpTypeFloat 64
%_arr_double_uint_4 = OpTypeArray %double %uint_4
%_ptr_Input__arr_double_uint_4 = OpTypePointer Input %_arr_double_uint_4
%_arr_v2float_uint_2 = OpTypeArray %v2float %uint_2
%_arr__arr_v2float_uint_2_uint_4 = OpTypeArray %_arr_v2float_uint_2 %uint_4
%_ptr_Input__arr__arr_v2float_uint_2_uint_4 = OpTypePointer Input %_arr__arr_v2float_uint_2_uint_4
%_arr_float_uint_2 = OpTypeArray %float %uint_2
%_arr__arr_float_uint_2_uint_4 = OpTypeArray %_arr_float_uint_2 %uint_4
%_ptr_Input__arr__arr_float_uint_2_uint_4 = OpTypePointer Input %_arr__arr_float_uint_2_uint_4
    %v3float = OpTypeVector %float 3
%mat2v3float = OpTypeMatrix %v3float 2
%_arr_mat2v3float_uint_4 = OpTypeArray %mat2v3float %uint_4
%_ptr_Input__arr_mat2v3float_uint_4 = OpTypePointer Input %_arr_mat2v3float_uint_4
%_ptr_Input_uint = OpTypePointer Input %uint
%_ptr_Output__arr__arr_v2float_uint_3_uint_4 = OpTypePointer Output %_arr__arr_v2float_uint_3_uint_4
%_ptr_Output__arr_double_uint_4 = OpTypePointer Output %_arr_double_uint_4
%_ptr_Output__arr__arr_v2float_uint_2_uint_4 = OpTypePointer Output %_arr__arr_v2float_uint_2_uint_4
%_ptr_Output__arr__arr_float_uint_2_uint_4 = OpTypePointer Output %_arr__arr_float_uint_2_uint_4
%_ptr_Output__arr_mat2v3float_uint_4 = OpTypePointer Output %_arr_mat2v3float_uint_4
%_arr_float_uint_4 = OpTypeArray %float %uint_4
%_ptr_Output__arr_float_uint_4 = OpTypePointer Output %_arr_float_uint_4
%_ptr_Output__arr_float_uint_2 = OpTypePointer Output %_arr_float_uint_2
       %void = OpTypeVoid
         %55 = OpTypeFunction %void
   %HSCtrlPt = OpTypeStruct %_arr_v2float_uint_3 %double %_arr_v2float_uint_2 %_arr_float_uint_2 %mat2v3float
%_arr_HSCtrlPt_uint_4 = OpTypeArray %HSCtrlPt %uint_4
%_ptr_Function__arr_HSCtrlPt_uint_4 = OpTypePointer Function %_arr_HSCtrlPt_uint_4
%_ptr_Function_uint = OpTypePointer Function %uint
%_ptr_Output__arr_v2float_uint_3 = OpTypePointer Output %_arr_v2float_uint_3
%_ptr_Output_double = OpTypePointer Output %double
%_ptr_Output__arr_v2float_uint_2 = OpTypePointer Output %_arr_v2float_uint_2
%_ptr_Output_mat2v3float = OpTypePointer Output %mat2v3float
%_ptr_Function_HSCtrlPt = OpTypePointer Function %HSCtrlPt
%_ptr_Function__arr_v2float_uint_3 = OpTypePointer Function %_arr_v2float_uint_3
%_ptr_Function_double = OpTypePointer Function %double
%_ptr_Function__arr_v2float_uint_2 = OpTypePointer Function %_arr_v2float_uint_2
%_ptr_Function__arr_float_uint_2 = OpTypePointer Function %_arr_float_uint_2
%_ptr_Function_mat2v3float = OpTypePointer Function %mat2v3float
       %bool = OpTypeBool
%_arr_float_uint_3 = OpTypeArray %float %uint_3
%_arr_float_uint_1 = OpTypeArray %float %uint_1
%HSPatchConstData = OpTypeStruct %_arr_float_uint_3 %_arr_float_uint_1
%_ptr_Output_float = OpTypePointer Output %float
        %201 = OpTypeFunction %HSPatchConstData %_ptr_Function__arr_HSCtrlPt_uint_4
%_ptr_Function_HSPatchConstData = OpTypePointer Function %HSPatchConstData
%_ptr_Function_float = OpTypePointer Function %float
        %212 = OpTypeFunction %HSCtrlPt %_ptr_Function__arr_HSCtrlPt_uint_4 %_ptr_Function_uint
%_ptr_Function_v2float = OpTypePointer Function %v2float
   %in_var_A = OpVariable %_ptr_Input__arr__arr_v2float_uint_3_uint_4 Input
   %in_var_B = OpVariable %_ptr_Input__arr_double_uint_4 Input
   %in_var_C = OpVariable %_ptr_Input__arr__arr_v2float_uint_2_uint_4 Input
   %in_var_D = OpVariable %_ptr_Input__arr__arr_float_uint_2_uint_4 Input
   %in_var_E = OpVariable %_ptr_Input__arr_mat2v3float_uint_4 Input

; CHECK-DAG: %in_var_A = OpVariable %_ptr_Input__arr_v2float_uint_4 Input
; CHECK-DAG: %in_var_A_0 = OpVariable %_ptr_Input__arr_v2float_uint_4 Input
; CHECK-DAG: %in_var_A_1 = OpVariable %_ptr_Input__arr_v2float_uint_4 Input
; CHECK-DAG: %in_var_C = OpVariable %_ptr_Input__arr_v2float_uint_4 Input
; CHECK-DAG: %in_var_C_0 = OpVariable %_ptr_Input__arr_v2float_uint_4 Input
; CHECK-DAG: %in_var_D = OpVariable %_ptr_Input__arr_float_uint_4 Input
; CHECK-DAG: %in_var_D_0 = OpVariable %_ptr_Input__arr_float_uint_4 Input
; CHECK-DAG: %in_var_E = OpVariable %_ptr_Input__arr_v3float_uint_4 Input
; CHECK-DAG: %in_var_E_0 = OpVariable %_ptr_Input__arr_v3float_uint_4 Input

%gl_InvocationID = OpVariable %_ptr_Input_uint Input
  %out_var_A = OpVariable %_ptr_Output__arr__arr_v2float_uint_3_uint_4 Output
  %out_var_B = OpVariable %_ptr_Output__arr_double_uint_4 Output
  %out_var_C = OpVariable %_ptr_Output__arr__arr_v2float_uint_2_uint_4 Output
  %out_var_D = OpVariable %_ptr_Output__arr__arr_float_uint_2_uint_4 Output
  %out_var_E = OpVariable %_ptr_Output__arr_mat2v3float_uint_4 Output

; CHECK-DAG: %out_var_A = OpVariable %_ptr_Output__arr_v2float_uint_4 Output
; CHECK-DAG: %out_var_A_0 = OpVariable %_ptr_Output__arr_v2float_uint_4 Output
; CHECK-DAG: %out_var_A_1 = OpVariable %_ptr_Output__arr_v2float_uint_4 Output
; CHECK-DAG: %out_var_C = OpVariable %_ptr_Output__arr_v2float_uint_4 Output
; CHECK-DAG: %out_var_C_0 = OpVariable %_ptr_Output__arr_v2float_uint_4 Output
; CHECK-DAG: %out_var_D = OpVariable %_ptr_Output__arr_float_uint_4 Output
; CHECK-DAG: %out_var_D_0 = OpVariable %_ptr_Output__arr_float_uint_4 Output
; CHECK-DAG: %out_var_E = OpVariable %_ptr_Output__arr_v3float_uint_4 Output
; CHECK-DAG: %out_var_E_0 = OpVariable %_ptr_Output__arr_v3float_uint_4 Output

%gl_TessLevelOuter = OpVariable %_ptr_Output__arr_float_uint_4 Output
%gl_TessLevelInner = OpVariable %_ptr_Output__arr_float_uint_2 Output
  )";

  // Splitting string literal to avoid a compile error "C2026: string too big,
  // trailing characters truncated" in Windows.
  const std::string function = R"(
       %main = OpFunction %void None %55
         %56 = OpLabel
%param_var_input = OpVariable %_ptr_Function__arr_HSCtrlPt_uint_4 Function
%param_var_CtrlPtID = OpVariable %_ptr_Function_uint Function
%temp_var_hullMainRetVal = OpVariable %_ptr_Function__arr_HSCtrlPt_uint_4 Function

         %64 = OpLoad %_arr__arr_v2float_uint_3_uint_4 %in_var_A

; CHECK: [[ptr:%\w+]] = OpAccessChain %_ptr_Input_v2float %in_var_A %uint_0
; CHECK: [[load_A_0:%\w+]] = OpLoad %v2float [[ptr:%\w+]]
; CHECK: [[ptr:%\w+]] = OpAccessChain %_ptr_Input_v2float %in_var_A_0 %uint_0
; CHECK: [[load_A_0_0:%\w+]] = OpLoad %v2float [[ptr:%\w+]]
; CHECK: [[ptr:%\w+]] = OpAccessChain %_ptr_Input_v2float %in_var_A_1 %uint_0
; CHECK: [[load_A_1_0:%\w+]] = OpLoad %v2float [[ptr:%\w+]]
; CHECK: [[ptr:%\w+]] = OpAccessChain %_ptr_Input_v2float %in_var_A %uint_1
; CHECK: [[load_A_1:%\w+]] = OpLoad %v2float [[ptr:%\w+]]
; CHECK: [[ptr:%\w+]] = OpAccessChain %_ptr_Input_v2float %in_var_A_0 %uint_1
; CHECK: [[load_A_0_1:%\w+]] = OpLoad %v2float [[ptr:%\w+]]
; CHECK: [[ptr:%\w+]] = OpAccessChain %_ptr_Input_v2float %in_var_A_1 %uint_1
; CHECK: [[load_A_1_1:%\w+]] = OpLoad %v2float [[ptr:%\w+]]

; CHECK-DAG: [[construct_A_0:%\w+]] = OpCompositeConstruct %_arr_v2float_uint_3 [[load_A_0]] [[load_A_0_0]] [[load_A_1_0]]
; CHECK-DAG: [[construct_A_1:%\w+]] = OpCompositeConstruct %_arr_v2float_uint_3 [[load_A_1]] [[load_A_0_1]] [[load_A_1_1]]

; CHECK: OpCompositeConstruct %_arr__arr_v2float_uint_3_uint_4 [[construct_A_0]] [[construct_A_1]]

         %65 = OpLoad %_arr_double_uint_4 %in_var_B
         %66 = OpLoad %_arr__arr_v2float_uint_2_uint_4 %in_var_C
         %67 = OpLoad %_arr__arr_float_uint_2_uint_4 %in_var_D
         %68 = OpLoad %_arr_mat2v3float_uint_4 %in_var_E

         %69 = OpCompositeExtract %_arr_v2float_uint_3 %64 0
         %70 = OpCompositeExtract %double %65 0
         %71 = OpCompositeExtract %_arr_v2float_uint_2 %66 0
         %72 = OpCompositeExtract %_arr_float_uint_2 %67 0
         %73 = OpCompositeExtract %mat2v3float %68 0
         %74 = OpCompositeConstruct %HSCtrlPt %69 %70 %71 %72 %73
         %75 = OpCompositeExtract %_arr_v2float_uint_3 %64 1
         %76 = OpCompositeExtract %double %65 1
         %77 = OpCompositeExtract %_arr_v2float_uint_2 %66 1
         %78 = OpCompositeExtract %_arr_float_uint_2 %67 1
         %79 = OpCompositeExtract %mat2v3float %68 1
         %80 = OpCompositeConstruct %HSCtrlPt %75 %76 %77 %78 %79
         %81 = OpCompositeExtract %_arr_v2float_uint_3 %64 2
         %82 = OpCompositeExtract %double %65 2
         %83 = OpCompositeExtract %_arr_v2float_uint_2 %66 2
         %84 = OpCompositeExtract %_arr_float_uint_2 %67 2
         %85 = OpCompositeExtract %mat2v3float %68 2
         %86 = OpCompositeConstruct %HSCtrlPt %81 %82 %83 %84 %85
         %87 = OpCompositeExtract %_arr_v2float_uint_3 %64 3
         %88 = OpCompositeExtract %double %65 3
         %89 = OpCompositeExtract %_arr_v2float_uint_2 %66 3
         %90 = OpCompositeExtract %_arr_float_uint_2 %67 3
         %91 = OpCompositeExtract %mat2v3float %68 3
         %92 = OpCompositeConstruct %HSCtrlPt %87 %88 %89 %90 %91
         %93 = OpCompositeConstruct %_arr_HSCtrlPt_uint_4 %74 %80 %86 %92
               OpStore %param_var_input %93
         %94 = OpLoad %uint %gl_InvocationID
               OpStore %param_var_CtrlPtID %94
         %95 = OpFunctionCall %HSCtrlPt %src_main %param_var_input %param_var_CtrlPtID
         %97 = OpCompositeExtract %_arr_v2float_uint_3 %95 0
         %99 = OpAccessChain %_ptr_Output__arr_v2float_uint_3 %out_var_A %94
               OpStore %99 %97

; CHECK: [[invoc_id:%\w+]] = OpLoad %uint %gl_InvocationID
; CHECK: [[ret:%\w+]] = OpFunctionCall %HSCtrlPt %src_main %param_var_input %param_var_CtrlPtID
; CHECK: [[a:%\w+]] = OpCompositeExtract %_arr_v2float_uint_3 [[ret]] 0
; CHECK: [[ptr_A:%\w+]] = OpAccessChain %_ptr_Output_v2float %out_var_A [[invoc_id]]
; CHECK: [[ptr_A_0:%\w+]] = OpAccessChain %_ptr_Output_v2float %out_var_A_0 [[invoc_id]]
; CHECK: [[ptr_A_1:%\w+]] = OpAccessChain %_ptr_Output_v2float %out_var_A_1 [[invoc_id]]
; CHECK: [[a0:%\w+]] = OpCompositeExtract %v2float [[a]] 0
; CHECK: OpStore [[ptr_A]] [[a0]]
; CHECK: [[a1:%\w+]] = OpCompositeExtract %v2float [[a]] 1
; CHECK: OpStore [[ptr_A_0]] [[a1]]
; CHECK: [[a2:%\w+]] = OpCompositeExtract %v2float [[a]] 2
; CHECK: OpStore [[ptr_A_1]] [[a2]]

        %100 = OpCompositeExtract %double %95 1
        %102 = OpAccessChain %_ptr_Output_double %out_var_B %94
               OpStore %102 %100
        %103 = OpCompositeExtract %_arr_v2float_uint_2 %95 2
        %105 = OpAccessChain %_ptr_Output__arr_v2float_uint_2 %out_var_C %94
               OpStore %105 %103
        %106 = OpCompositeExtract %_arr_float_uint_2 %95 3
        %107 = OpAccessChain %_ptr_Output__arr_float_uint_2 %out_var_D %94
               OpStore %107 %106
        %108 = OpCompositeExtract %mat2v3float %95 4
        %110 = OpAccessChain %_ptr_Output_mat2v3float %out_var_E %94
               OpStore %110 %108
               OpControlBarrier %uint_2 %uint_4 %uint_0
        %112 = OpAccessChain %_ptr_Function_HSCtrlPt %temp_var_hullMainRetVal %uint_0
        %114 = OpAccessChain %_ptr_Function__arr_v2float_uint_3 %112 %uint_0
        %115 = OpAccessChain %_ptr_Output__arr_v2float_uint_3 %out_var_A %uint_0
        %116 = OpLoad %_arr_v2float_uint_3 %115
               OpStore %114 %116

; CHECK: [[ptr_outpatch:%\w+]] = OpAccessChain %_ptr_Function_HSCtrlPt %temp_var_hullMainRetVal %uint_0
; CHECK: [[ptr_outpatch_a:%\w+]] = OpAccessChain %_ptr_Function__arr_v2float_uint_3 [[ptr_outpatch]] %uint_0
; CHECK: [[ptr_A_0:%\w+]] = OpAccessChain %_ptr_Output_v2float %out_var_A %uint_0
; CHECK: [[ptr_A_0_0:%\w+]] = OpAccessChain %_ptr_Output_v2float %out_var_A_0 %uint_0
; CHECK: [[ptr_A_1_0:%\w+]] = OpAccessChain %_ptr_Output_v2float %out_var_A_1 %uint_0
; CHECK: [[A_0:%\w+]] = OpLoad %v2float [[ptr_A_0]]
; CHECK: [[A_0_0:%\w+]] = OpLoad %v2float [[ptr_A_0_0]]
; CHECK: [[A_1_0:%\w+]] = OpLoad %v2float [[ptr_A_1_0]]
; CHECK: [[A0:%\w+]] = OpCompositeConstruct %_arr_v2float_uint_3 [[A_0]] [[A_0_0]] [[A_1_0]]
; CHECK: OpStore [[ptr_outpatch_a]] [[A0]]

        %118 = OpAccessChain %_ptr_Function_double %112 %uint_1
        %119 = OpAccessChain %_ptr_Output_double %out_var_B %uint_0
        %120 = OpLoad %double %119
               OpStore %118 %120
        %122 = OpAccessChain %_ptr_Function__arr_v2float_uint_2 %112 %uint_2
        %123 = OpAccessChain %_ptr_Output__arr_v2float_uint_2 %out_var_C %uint_0
        %124 = OpLoad %_arr_v2float_uint_2 %123
               OpStore %122 %124
        %126 = OpAccessChain %_ptr_Function__arr_float_uint_2 %112 %uint_3
        %127 = OpAccessChain %_ptr_Output__arr_float_uint_2 %out_var_D %uint_0
        %128 = OpLoad %_arr_float_uint_2 %127
               OpStore %126 %128
        %130 = OpAccessChain %_ptr_Function_mat2v3float %112 %uint_4
        %131 = OpAccessChain %_ptr_Output_mat2v3float %out_var_E %uint_0
        %132 = OpLoad %mat2v3float %131
               OpStore %130 %132
        %133 = OpAccessChain %_ptr_Function_HSCtrlPt %temp_var_hullMainRetVal %uint_1
        %134 = OpAccessChain %_ptr_Function__arr_v2float_uint_3 %133 %uint_0
        %135 = OpAccessChain %_ptr_Output__arr_v2float_uint_3 %out_var_A %uint_1
        %136 = OpLoad %_arr_v2float_uint_3 %135
               OpStore %134 %136

; CHECK: [[ptr_outpatch:%\w+]] = OpAccessChain %_ptr_Function_HSCtrlPt %temp_var_hullMainRetVal %uint_1
; CHECK: [[ptr_outpatch_a:%\w+]] = OpAccessChain %_ptr_Function__arr_v2float_uint_3 [[ptr_outpatch]] %uint_0
; CHECK: [[ptr_A_1:%\w+]] = OpAccessChain %_ptr_Output_v2float %out_var_A %uint_1
; CHECK: [[ptr_A_0_1:%\w+]] = OpAccessChain %_ptr_Output_v2float %out_var_A_0 %uint_1
; CHECK: [[ptr_A_1_1:%\w+]] = OpAccessChain %_ptr_Output_v2float %out_var_A_1 %uint_1
; CHECK: [[A_1:%\w+]] = OpLoad %v2float [[ptr_A_1]]
; CHECK: [[A_0_1:%\w+]] = OpLoad %v2float [[ptr_A_0_1]]
; CHECK: [[A_1_1:%\w+]] = OpLoad %v2float [[ptr_A_1_1]]
; CHECK: [[A1:%\w+]] = OpCompositeConstruct %_arr_v2float_uint_3 [[A_1]] [[A_0_1]] [[A_1_1]]
; CHECK: OpStore [[ptr_outpatch_a]] [[A1]]

        %137 = OpAccessChain %_ptr_Function_double %133 %uint_1
        %138 = OpAccessChain %_ptr_Output_double %out_var_B %uint_1
        %139 = OpLoad %double %138
               OpStore %137 %139
        %140 = OpAccessChain %_ptr_Function__arr_v2float_uint_2 %133 %uint_2
        %141 = OpAccessChain %_ptr_Output__arr_v2float_uint_2 %out_var_C %uint_1
        %142 = OpLoad %_arr_v2float_uint_2 %141
               OpStore %140 %142
        %143 = OpAccessChain %_ptr_Function__arr_float_uint_2 %133 %uint_3
        %144 = OpAccessChain %_ptr_Output__arr_float_uint_2 %out_var_D %uint_1
        %145 = OpLoad %_arr_float_uint_2 %144
               OpStore %143 %145
        %146 = OpAccessChain %_ptr_Function_mat2v3float %133 %uint_4
        %147 = OpAccessChain %_ptr_Output_mat2v3float %out_var_E %uint_1
        %148 = OpLoad %mat2v3float %147
               OpStore %146 %148
        %149 = OpAccessChain %_ptr_Function_HSCtrlPt %temp_var_hullMainRetVal %uint_2
        %150 = OpAccessChain %_ptr_Function__arr_v2float_uint_3 %149 %uint_0
        %151 = OpAccessChain %_ptr_Output__arr_v2float_uint_3 %out_var_A %uint_2
        %152 = OpLoad %_arr_v2float_uint_3 %151
               OpStore %150 %152
        %153 = OpAccessChain %_ptr_Function_double %149 %uint_1
        %154 = OpAccessChain %_ptr_Output_double %out_var_B %uint_2
        %155 = OpLoad %double %154
               OpStore %153 %155
        %156 = OpAccessChain %_ptr_Function__arr_v2float_uint_2 %149 %uint_2
        %157 = OpAccessChain %_ptr_Output__arr_v2float_uint_2 %out_var_C %uint_2
        %158 = OpLoad %_arr_v2float_uint_2 %157
               OpStore %156 %158
        %159 = OpAccessChain %_ptr_Function__arr_float_uint_2 %149 %uint_3
        %160 = OpAccessChain %_ptr_Output__arr_float_uint_2 %out_var_D %uint_2
        %161 = OpLoad %_arr_float_uint_2 %160
               OpStore %159 %161
        %162 = OpAccessChain %_ptr_Function_mat2v3float %149 %uint_4
        %163 = OpAccessChain %_ptr_Output_mat2v3float %out_var_E %uint_2
        %164 = OpLoad %mat2v3float %163
               OpStore %162 %164
        %165 = OpAccessChain %_ptr_Function_HSCtrlPt %temp_var_hullMainRetVal %uint_3
        %166 = OpAccessChain %_ptr_Function__arr_v2float_uint_3 %165 %uint_0
        %167 = OpAccessChain %_ptr_Output__arr_v2float_uint_3 %out_var_A %uint_3
        %168 = OpLoad %_arr_v2float_uint_3 %167
               OpStore %166 %168
        %169 = OpAccessChain %_ptr_Function_double %165 %uint_1
        %170 = OpAccessChain %_ptr_Output_double %out_var_B %uint_3
        %171 = OpLoad %double %170
               OpStore %169 %171
        %172 = OpAccessChain %_ptr_Function__arr_v2float_uint_2 %165 %uint_2
        %173 = OpAccessChain %_ptr_Output__arr_v2float_uint_2 %out_var_C %uint_3
        %174 = OpLoad %_arr_v2float_uint_2 %173
               OpStore %172 %174
        %175 = OpAccessChain %_ptr_Function__arr_float_uint_2 %165 %uint_3
        %176 = OpAccessChain %_ptr_Output__arr_float_uint_2 %out_var_D %uint_3
        %177 = OpLoad %_arr_float_uint_2 %176
               OpStore %175 %177
        %178 = OpAccessChain %_ptr_Function_mat2v3float %165 %uint_4
        %179 = OpAccessChain %_ptr_Output_mat2v3float %out_var_E %uint_3
        %180 = OpLoad %mat2v3float %179
               OpStore %178 %180
        %182 = OpIEqual %bool %94 %uint_0
               OpSelectionMerge %if_merge None
               OpBranchConditional %182 %if_true %if_merge
    %if_true = OpLabel
        %188 = OpFunctionCall %HSPatchConstData %HSPatchConstantFunc %temp_var_hullMainRetVal
        %190 = OpCompositeExtract %_arr_float_uint_3 %188 0
        %192 = OpAccessChain %_ptr_Output_float %gl_TessLevelOuter %uint_0
        %193 = OpCompositeExtract %float %190 0
               OpStore %192 %193
        %194 = OpAccessChain %_ptr_Output_float %gl_TessLevelOuter %uint_1
        %195 = OpCompositeExtract %float %190 1
               OpStore %194 %195
        %196 = OpAccessChain %_ptr_Output_float %gl_TessLevelOuter %uint_2
        %197 = OpCompositeExtract %float %190 2
               OpStore %196 %197
        %198 = OpCompositeExtract %_arr_float_uint_1 %188 1
        %199 = OpAccessChain %_ptr_Output_float %gl_TessLevelInner %uint_0
        %200 = OpCompositeExtract %float %198 0
               OpStore %199 %200
               OpBranch %if_merge
   %if_merge = OpLabel
               OpReturn
               OpFunctionEnd
%HSPatchConstantFunc = OpFunction %HSPatchConstData None %201
      %input = OpFunctionParameter %_ptr_Function__arr_HSCtrlPt_uint_4
   %bb_entry = OpLabel
       %data = OpVariable %_ptr_Function_HSPatchConstData Function
        %207 = OpAccessChain %_ptr_Function_float %data %int_0 %int_0
               OpStore %207 %float_3
        %208 = OpAccessChain %_ptr_Function_float %data %int_0 %int_1
               OpStore %208 %float_3
        %209 = OpAccessChain %_ptr_Function_float %data %int_0 %int_2
               OpStore %209 %float_3
        %210 = OpAccessChain %_ptr_Function_float %data %int_1 %int_0
               OpStore %210 %float_3
        %211 = OpLoad %HSPatchConstData %data
               OpReturnValue %211
               OpFunctionEnd
   %src_main = OpFunction %HSCtrlPt None %212
    %input_0 = OpFunctionParameter %_ptr_Function__arr_HSCtrlPt_uint_4
   %CtrlPtID = OpFunctionParameter %_ptr_Function_uint
 %bb_entry_0 = OpLabel
     %data_0 = OpVariable %_ptr_Function_HSCtrlPt Function
        %217 = OpLoad %uint %CtrlPtID
        %218 = OpLoad %uint %CtrlPtID
        %220 = OpAccessChain %_ptr_Function_v2float %input_0 %217 %int_2 %218
        %221 = OpLoad %v2float %220
        %222 = OpAccessChain %_ptr_Function_v2float %data_0 %int_0 %int_1
               OpStore %222 %221
        %223 = OpLoad %uint %CtrlPtID
        %224 = OpAccessChain %_ptr_Function__arr_float_uint_2 %input_0 %223 %int_3
        %225 = OpLoad %_arr_float_uint_2 %224
        %226 = OpAccessChain %_ptr_Function__arr_float_uint_2 %data_0 %int_3
               OpStore %226 %225
        %227 = OpLoad %HSCtrlPt %data_0
               OpReturnValue %227
               OpFunctionEnd
  )";

  std::vector<StageVariableInfo> info({
      // location, component, extra_arrayness, is_input_var
      {0, 0, 4, true},
      {0, 0, 4, false},
      {1, 2, 4, true},
      {1, 2, 4, false},
      {3, 0, 4, true},
      {3, 0, 4, false},
      {3, 1, 4, true},
      {3, 1, 4, false},
  });
  SinglePassRunAndMatch<FlattenArrayMatrixStageVariable>(header + function,
                                                         true, info);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
