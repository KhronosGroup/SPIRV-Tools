// Copyright (c) 2016 Google Inc.
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

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace opt {
namespace {

void DoRoundTripCheck(const std::string& text) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  ASSERT_NE(nullptr, context) << "Failed to assemble\n" << text;

  std::vector<uint32_t> binary;
  context->module()->ToBinary(&binary, /* skip_nop = */ false);

  std::string disassembled_text;
  EXPECT_TRUE(t.Disassemble(binary, &disassembled_text));
  EXPECT_EQ(text, disassembled_text);
}

TEST(IrBuilder, RoundTrip) {
  // #version 310 es
  // int add(int a, int b) { return a + b; }
  // void main() { add(1, 2); }
  DoRoundTripCheck(
      // clang-format off
               "OpCapability Shader\n"
          "%1 = OpExtInstImport \"GLSL.std.450\"\n"
               "OpMemoryModel Logical GLSL450\n"
               "OpEntryPoint Vertex %main \"main\"\n"
               "OpSource ESSL 310\n"
               "OpSourceExtension \"GL_GOOGLE_cpp_style_line_directive\"\n"
               "OpSourceExtension \"GL_GOOGLE_include_directive\"\n"
               "OpName %main \"main\"\n"
               "OpName %add_i1_i1_ \"add(i1;i1;\"\n"
               "OpName %a \"a\"\n"
               "OpName %b \"b\"\n"
               "OpName %param \"param\"\n"
               "OpName %param_0 \"param\"\n"
       "%void = OpTypeVoid\n"
          "%9 = OpTypeFunction %void\n"
        "%int = OpTypeInt 32 1\n"
 "%_ptr_Function_int = OpTypePointer Function %int\n"
         "%12 = OpTypeFunction %int %_ptr_Function_int %_ptr_Function_int\n"
      "%int_1 = OpConstant %int 1\n"
      "%int_2 = OpConstant %int 2\n"
       "%main = OpFunction %void None %9\n"
         "%15 = OpLabel\n"
      "%param = OpVariable %_ptr_Function_int Function\n"
    "%param_0 = OpVariable %_ptr_Function_int Function\n"
               "OpStore %param %int_1\n"
               "OpStore %param_0 %int_2\n"
         "%16 = OpFunctionCall %int %add_i1_i1_ %param %param_0\n"
               "OpReturn\n"
               "OpFunctionEnd\n"
 "%add_i1_i1_ = OpFunction %int None %12\n"
          "%a = OpFunctionParameter %_ptr_Function_int\n"
          "%b = OpFunctionParameter %_ptr_Function_int\n"
         "%17 = OpLabel\n"
         "%18 = OpLoad %int %a\n"
         "%19 = OpLoad %int %b\n"
         "%20 = OpIAdd %int %18 %19\n"
               "OpReturnValue %20\n"
               "OpFunctionEnd\n");
  // clang-format on
}

TEST(IrBuilder, RoundTripIncompleteBasicBlock) {
  DoRoundTripCheck(
      "%2 = OpFunction %1 None %3\n"
      "%4 = OpLabel\n"
      "OpNop\n");
}

TEST(IrBuilder, RoundTripIncompleteFunction) {
  DoRoundTripCheck("%2 = OpFunction %1 None %3\n");
}

TEST(IrBuilder, KeepLineDebugInfo) {
  // #version 310 es
  // void main() {}
  DoRoundTripCheck(
      // clang-format off
               "OpCapability Shader\n"
          "%1 = OpExtInstImport \"GLSL.std.450\"\n"
               "OpMemoryModel Logical GLSL450\n"
               "OpEntryPoint Vertex %main \"main\"\n"
          "%3 = OpString \"minimal.vert\"\n"
               "OpSource ESSL 310\n"
               "OpName %main \"main\"\n"
               "OpLine %3 10 10\n"
       "%void = OpTypeVoid\n"
               "OpLine %3 100 100\n"
          "%5 = OpTypeFunction %void\n"
       "%main = OpFunction %void None %5\n"
               "OpLine %3 1 1\n"
               "OpNoLine\n"
               "OpLine %3 2 2\n"
               "OpLine %3 3 3\n"
          "%6 = OpLabel\n"
               "OpLine %3 4 4\n"
               "OpNoLine\n"
               "OpReturn\n"
               "OpFunctionEnd\n");
  // clang-format on
}

// TODO: This function is used to test if spirv-opt can load SPIR-V code with
// debug info extension without errors. After handling debug info instructions
// in optimization passes, we must drop it and use DoRoundTripCheck instead.
void CheckLoadingError(const std::string& text) {
  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  ASSERT_NE(nullptr, context) << "Failed to assemble\n" << text;

  std::vector<uint32_t> binary;
  context->module()->ToBinary(&binary, /* skip_nop = */ false);

  std::string disassembled_text;
  EXPECT_TRUE(t.Disassemble(binary, &disassembled_text));
}

TEST(IrBuilder, ConsumeDebugInfoInst) {
  // /* HLSL */
  //
  // struct VS_OUTPUT {
  //   float4 pos : SV_POSITION;
  //   float4 color : COLOR;
  // };
  //
  // VS_OUTPUT main(float4 pos : POSITION,
  //                float4 color : COLOR) {
  //   VS_OUTPUT vout;
  //   vout.pos = pos;
  //   vout.color = color;
  //   return vout;
  // }
  CheckLoadingError(R"(OpCapability Shader
%1 = OpExtInstImport "OpenCL.DebugInfo.100"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %pos %color %gl_Position %out_var_COLOR
%7 = OpString "simple_vs.hlsl"
%8 = OpString "#line 1 \"simple_vs.hlsl\"
struct VS_OUTPUT {
  float4 pos : SV_POSITION;
  float4 color : COLOR;
};

VS_OUTPUT main(float4 pos : POSITION,
               float4 color : COLOR) {
  VS_OUTPUT vout;
  vout.pos = pos;
  vout.color = color;
  return vout;
}
"
OpSource HLSL 600 %7 "#line 1 \"simple_vs.hlsl\"
struct VS_OUTPUT {
  float4 pos : SV_POSITION;
  float4 color : COLOR;
};

VS_OUTPUT main(float4 pos : POSITION,
               float4 color : COLOR) {
  VS_OUTPUT vout;
  vout.pos = pos;
  vout.color = color;
  return vout;
}
"
%9 = OpString "struct VS_OUTPUT"
%10 = OpString "float"
%11 = OpString "pos : SV_POSITION"
%12 = OpString "color : COLOR"
%13 = OpString "VS_OUTPUT"
%14 = OpString "main"
%15 = OpString "VS_OUTPUT_main_v4f_v4f"
%16 = OpString "pos : POSITION"
%17 = OpString "color : COLOR"
%18 = OpString "vout"
OpName %out_var_COLOR "out.var.COLOR"
OpName %main "main"
OpName %VS_OUTPUT "VS_OUTPUT"
OpMemberName %VS_OUTPUT 0 "pos"
OpMemberName %VS_OUTPUT 1 "color"
OpName %pos "pos"
OpName %color "color"
OpName %vout "vout"
OpDecorate %gl_Position BuiltIn Position
OpDecorate %pos Location 0
OpDecorate %color Location 1
OpDecorate %out_var_COLOR Location 0
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_32 = OpConstant %int 32
%int_128 = OpConstant %int 128
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%void = OpTypeVoid
%31 = OpTypeFunction %void
%_ptr_Function_v4float = OpTypePointer Function %v4float
%VS_OUTPUT = OpTypeStruct %v4float %v4float
%33 = OpTypeFunction %VS_OUTPUT %_ptr_Function_v4float %_ptr_Function_v4float
%_ptr_Function_VS_OUTPUT = OpTypePointer Function %VS_OUTPUT
OpLine %7 6 23
%pos = OpVariable %_ptr_Input_v4float Input
OpLine %7 7 23
%color = OpVariable %_ptr_Input_v4float Input
OpLine %7 2 16
%gl_Position = OpVariable %_ptr_Output_v4float Output
OpLine %7 3 18
%out_var_COLOR = OpVariable %_ptr_Output_v4float Output
%35 = OpExtInst %void %1 DebugSource %7 %8
%36 = OpExtInst %void %1 DebugCompilationUnit 2 4 %35 HLSL
%37 = OpExtInst %void %1 DebugTypeComposite %9 Structure %35 1 1 %36 %13 %int_128 FlagIsProtected|FlagIsPrivate %35 %35 ; %38 %39
%40 = OpExtInst %void %1 DebugTypeBasic %10 %int_32 Float
%41 = OpExtInst %void %1 DebugTypeVector %40 4
%38 = OpExtInst %void %1 DebugTypeMember %11 %41 %35 2 3 %37 %int_0 %int_128 FlagIsProtected|FlagIsPrivate
%39 = OpExtInst %void %1 DebugTypeMember %12 %41 %35 3 3 %37 %int_128 %int_128 FlagIsProtected|FlagIsPrivate
%42 = OpExtInst %void %1 DebugTypeFunction FlagIsProtected|FlagIsPrivate %37 %41 %41
%43 = OpExtInst %void %1 DebugExpression
%44 = OpExtInst %void %1 DebugFunction %14 %42 %35 6 1 %36 %15 FlagIsProtected|FlagIsPrivate 7 %35; %main
%45 = OpExtInst %void %1 DebugLocalVariable %16 %41 %35 6 16 %44 FlagIsLocal 0
%46 = OpExtInst %void %1 DebugLocalVariable %17 %41 %35 7 16 %44 FlagIsLocal 1
%47 = OpExtInst %void %1 DebugLocalVariable %18 %37 %35 8 3 %44 FlagIsLocal
%48 = OpExtInst %void %1 DebugDeclare %45 %pos %43
%49 = OpExtInst %void %1 DebugDeclare %46 %color %43
OpLine %7 6 1
%main = OpFunction %void None %31
%51 = OpLabel
%50 = OpExtInst %void %1 DebugScope %44
OpLine %7 8 13
%vout = OpVariable %_ptr_Function_VS_OUTPUT Function
%52 = OpExtInst %void %1 DebugDeclare %47 %vout %43
OpLine %7 9 14
%53 = OpLoad %v4float %pos
OpLine %7 9 3
%54 = OpAccessChain %_ptr_Function_v4float %vout %int_0
%55 = OpExtInst %void %1 DebugValue %47 %54 %43 %int_0
OpStore %54 %53
OpLine %7 10 16
%56 = OpLoad %v4float %color
OpLine %7 10 3
%57 = OpAccessChain %_ptr_Function_v4float %vout %int_1
%58 = OpExtInst %void %1 DebugValue %47 %57 %43 %int_1
OpStore %57 %56
OpLine %7 11 10
%59 = OpLoad %VS_OUTPUT %vout
OpLine %7 11 3
%60 = OpCompositeExtract %v4float %59 0
OpStore %gl_Position %60
%61 = OpCompositeExtract %v4float %59 1
OpStore %out_var_COLOR %61
OpReturn
%62 = OpExtInst %void %1 DebugNoScope
OpFunctionEnd
)");
}

TEST(IrBuilder, ConsumeDebugInfoLexicalScopeInst) {
  // /* HLSL */
  //
  // float4 func2(float arg2) {   // func2_block
  //   return float4(arg2, 0, 0, 0);
  // }
  //
  // float4 func1(float arg1) {   // func1_block
  //   if (arg1 > 1) {       // if_true_block
  //     return float4(0, 0, 0, 0);
  //   }
  //   return func2(arg1);   // if_merge_block
  // }
  //
  // float4 main(float pos : POSITION) : SV_POSITION {  // main
  //   return func1(pos);
  // }
  CheckLoadingError(R"(
               OpCapability Shader
     %DbgExt = OpExtInstImport "OpenCL.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %pos %gl_Position
          %src = OpString "block/block.hlsl"
         %code = OpString "#line 1 \"block/block.hlsl\"
float4 func2(float arg2) {
  return float4(arg2, 0, 0, 0);
}

float4 func1(float arg1) {
  if (arg1 > 1) {
    return float4(0, 0, 0, 0);
  }
  return func2(arg1);
}

float4 main(float pos : POSITION) : SV_POSITION {
  return func1(pos);
}
"
               OpSource HLSL 600 %src "#line 1 \"block/block.hlsl\"
float4 func2(float arg2) {
  return float4(arg2, 0, 0, 0);
}

float4 func1(float arg1) {
  if (arg1 > 1) {
    return float4(0, 0, 0, 0);
  }
  return func2(arg1);
}

float4 main(float pos : POSITION) : SV_POSITION {
  return func1(pos);
}
"

; Type names
%float_name = OpString "float"
%main_name = OpString "main"
%main_linkage_name = OpString "v4f_main_f"
%func1_linkage_name = OpString "v4f_func1_f"
%func2_linkage_name = OpString "v4f_func2_f"
%pos_name = OpString "pos : POSITION"
%func1_name = OpString "func1"
%func2_name = OpString "func2"

               OpName %main "main"
               OpName %pos "pos"
               OpName %bb_entry "bb.entry"
               OpName %param_var_arg1 "param.var.arg1"
               OpName %func1 "func1"
               OpName %arg1 "arg1"
               OpName %bb_entry_0 "bb.entry"
               OpName %param_var_arg2 "param.var.arg2"
               OpName %if_true "if.true"
               OpName %if_merge "if.merge"
               OpName %func2 "func2"
               OpName %arg2 "arg2"
               OpName %bb_entry_1 "bb.entry"
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %pos Location 0
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
    %float_1 = OpConstant %float 1
    %float_0 = OpConstant %float 0

; Type sizes in bit unit. For example, 32 means "32 bits"
%int_32 = OpConstant %int 32

    %v4float = OpTypeVector %float 4
          %9 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_Input_float = OpTypePointer Input %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %13 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
         %20 = OpTypeFunction %v4float %_ptr_Function_float
       %bool = OpTypeBool
               OpLine %src 12 25
%pos = OpVariable %_ptr_Input_float Input
               OpLine %src 12 37
%gl_Position = OpVariable %_ptr_Output_v4float Output

; Compilation Unit
%dbg_src = OpExtInst %void %DbgExt DebugSource %src %code
%comp_unit = OpExtInst %void %DbgExt DebugCompilationUnit 2 4 %dbg_src HLSL

; Type information
; %VS_OUTPUT_info and %VS_OUTPUT_pos_info have cycling reference
%float_info = OpExtInst %void %DbgExt DebugTypeBasic %float_name %int_32 Float
%v4float_info = OpExtInst %void %DbgExt DebugTypeVector %float_info 4
%main_type_info = OpExtInst %void %DbgExt DebugTypeFunction FlagIsPublic %v4float_info %float_info
%func1_type_info = OpExtInst %void %DbgExt DebugTypeFunction FlagIsPublic %v4float_info %float_info
%func2_type_info = OpExtInst %void %DbgExt DebugTypeFunction FlagIsPublic %v4float_info %float_info

; Function information
%main_info = OpExtInst %void %DbgExt DebugFunction %main_name %main_type_info %dbg_src 12 1 %comp_unit %main_linkage_name FlagIsPublic 13 %main
%func1_info = OpExtInst %void %DbgExt DebugFunction %func1_name %func1_type_info %dbg_src 5 1 %comp_unit %func1_linkage_name FlagIsPublic 13 %func1
%func2_info = OpExtInst %void %DbgExt DebugFunction %func2_name %func2_type_info %dbg_src 1 1 %comp_unit %func2_linkage_name FlagIsPublic 13 %func2

; Block information
%if_true_block = OpExtInst %void %DbgExt DebugLexicalBlock %dbg_src 6 17 %func1_info
%if_merge_block = OpExtInst %void %DbgExt DebugLexicalBlock %dbg_src 9 3 %func1_info

               OpLine %src 12 1
       %main = OpFunction %void None %13
   %bb_entry = OpLabel

; Scope of function main()
%main_scope = OpExtInst %void %DbgExt DebugScope %main_info

               OpLine %src 13 16
%param_var_arg1 = OpVariable %_ptr_Function_float Function
         %24 = OpLoad %float %pos
               OpStore %param_var_arg1 %24
               OpLine %src 13 10
         %25 = OpFunctionCall %v4float %func1 %param_var_arg1
               OpLine %src 13 3
               OpStore %gl_Position %25
               OpReturn
               OpFunctionEnd

               OpLine %src 5 1
      %func1 = OpFunction %v4float None %20
               OpLine %src 5 20
       %arg1 = OpFunctionParameter %_ptr_Function_float
 %bb_entry_0 = OpLabel

; Scope of function func1()
%func1_scope = OpExtInst %void %DbgExt DebugScope %func1_info

               OpLine %src 9 16
%param_var_arg2 = OpVariable %_ptr_Function_float Function
               OpLine %src 6 7
         %30 = OpLoad %float %arg1
               OpLine %src 6 12
         %32 = OpFOrdGreaterThan %bool %30 %float_1
               OpLine %src 6 17
               OpSelectionMerge %if_merge None
               OpBranchConditional %32 %if_true %if_merge
    %if_true = OpLabel

; Scope of block %if_true
%if_true_scope = OpExtInst %void %DbgExt DebugScope %if_true_block

               OpLine %src 7 5
               OpReturnValue %9
   %if_merge = OpLabel

; Scope of block %if_merge
%if_merge_scope = OpExtInst %void %DbgExt DebugScope %if_merge_block

               OpLine %src 9 16
         %35 = OpLoad %float %arg1
               OpStore %param_var_arg2 %35
               OpLine %src 9 10
         %36 = OpFunctionCall %v4float %func2 %param_var_arg2
               OpLine %src 9 3
               OpReturnValue %36
               OpFunctionEnd

               OpLine %src 1 1
      %func2 = OpFunction %v4float None %20
               OpLine %src 1 20
       %arg2 = OpFunctionParameter %_ptr_Function_float
 %bb_entry_1 = OpLabel

; Scope of function func2()
%func2_scope = OpExtInst %void %DbgExt DebugScope %func2_info

               OpLine %src 2 17
         %40 = OpLoad %float %arg2
         %41 = OpCompositeConstruct %v4float %40 %float_0 %float_0 %float_0
               OpLine %src 2 3
               OpReturnValue %41
               OpFunctionEnd
)");
}

TEST(IrBuilder, ConsumeDebugInlinedAt) {
  // /* HLSL */
  //
  // float4 func2(float arg2) {   // func2_block
  //   return float4(arg2, 0, 0, 0);
  // }
  //
  // float4 func1(float arg1) {   // func1_block
  //   if (arg1 > 1) {       // if_true_block
  //     return float4(0, 0, 0, 0);
  //   }
  //   return func2(arg1);   // if_merge_block
  // }
  //
  // float4 main(float pos : POSITION) : SV_POSITION {  // main
  //   return func1(pos);
  // }
  CheckLoadingError(R"(
               OpCapability Shader
     %DbgExt = OpExtInstImport "OpenCL.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %pos %gl_Position
          %src = OpString "block/block.hlsl"
         %code = OpString "#line 1 \"block/block.hlsl\"
float4 func2(float arg2) {
  return float4(arg2, 0, 0, 0);
}

float4 func1(float arg1) {
  if (arg1 > 1) {
    return float4(0, 0, 0, 0);
  }
  return func2(arg1);
}

float4 main(float pos : POSITION) : SV_POSITION {
  return func1(pos);
}
"
               OpSource HLSL 600 %src "#line 1 \"block/block.hlsl\"
float4 func2(float arg2) {
  return float4(arg2, 0, 0, 0);
}

float4 func1(float arg1) {
  if (arg1 > 1) {
    return float4(0, 0, 0, 0);
  }
  return func2(arg1);
}

float4 main(float pos : POSITION) : SV_POSITION {
  return func1(pos);
}
"

; Type names
%float_name = OpString "float"
%main_name = OpString "main"
%main_linkage_name = OpString "v4f_main_f"
%func1_linkage_name = OpString "v4f_func1_f"
%func2_linkage_name = OpString "v4f_func2_f"
%pos_name = OpString "pos : POSITION"
%func1_name = OpString "func1"
%func2_name = OpString "func2"

               OpName %main "main"
               OpName %pos "pos"
               OpName %bb_entry "bb.entry"
               OpName %if_true "if.true"
               OpName %if_merge "if.merge"
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %pos Location 0
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
    %float_1 = OpConstant %float 1
    %float_0 = OpConstant %float 0

; Type sizes in bit unit. For example, 32 means "32"
%int_32 = OpConstant %int 32

    %v4float = OpTypeVector %float 4
          %9 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_Input_float = OpTypePointer Input %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %13 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
         %20 = OpTypeFunction %v4float %_ptr_Function_float
       %bool = OpTypeBool
               OpLine %src 12 25
%pos = OpVariable %_ptr_Input_float Input
               OpLine %src 12 37
%gl_Position = OpVariable %_ptr_Output_v4float Output

; Compilation Unit
%dbg_src = OpExtInst %void %DbgExt DebugSource %src %code
%comp_unit = OpExtInst %void %DbgExt DebugCompilationUnit 2 4 %dbg_src HLSL

; Type information
; %VS_OUTPUT_info and %VS_OUTPUT_pos_info have cycling reference
%float_info = OpExtInst %void %DbgExt DebugTypeBasic %float_name %int_32 Float
%v4float_info = OpExtInst %void %DbgExt DebugTypeVector %float_info 4
%main_type_info = OpExtInst %void %DbgExt DebugTypeFunction FlagIsPublic %v4float_info %float_info
%func1_type_info = OpExtInst %void %DbgExt DebugTypeFunction FlagIsPublic %v4float_info %float_info
%func2_type_info = OpExtInst %void %DbgExt DebugTypeFunction FlagIsPublic %v4float_info %float_info

; Function information
%opted_out = OpExtInst %void %DbgExt DebugInfoNone
%main_info = OpExtInst %void %DbgExt DebugFunction %main_name %main_type_info %dbg_src 12 1 %comp_unit %main_linkage_name FlagIsPublic 13 %main
%func1_info = OpExtInst %void %DbgExt DebugFunction %func1_name %func1_type_info %dbg_src 5 1 %comp_unit %func1_linkage_name FlagIsPublic 13 %opted_out
%func2_info = OpExtInst %void %DbgExt DebugFunction %func2_name %func2_type_info %dbg_src 1 1 %comp_unit %func2_linkage_name FlagIsPublic 13 %opted_out

; Block information
%main_block = OpExtInst %void %DbgExt DebugLexicalBlock %dbg_src 12 49 %main_info
%func1_block = OpExtInst %void %DbgExt DebugLexicalBlock %dbg_src 5 26 %func1_info
%func2_block = OpExtInst %void %DbgExt DebugLexicalBlock %dbg_src 1 26 %func2_info
%if_true_block = OpExtInst %void %DbgExt DebugLexicalBlock %dbg_src 6 17 %func1_block
%if_merge_block = OpExtInst %void %DbgExt DebugLexicalBlock %dbg_src 9 3 %func1_block

; Inlining information
%inline_func2_to_line9 = OpExtInst %void %DbgExt DebugInlinedAt 9 %if_merge_block
%inline_func1_to_line13 = OpExtInst %void %DbgExt DebugInlinedAt 13 %main_block
%inline_recursive = OpExtInst %void %DbgExt DebugInlinedAt 13 %main_block %inline_func2_to_line9

               OpLine %src 12 1
       %main = OpFunction %void None %13
   %bb_entry = OpLabel

; Inlining information: "Lines from line 6 originally included in %func1_block were inlined to line 13"
%scope_for_inline_func1 = OpExtInst %void %DbgExt DebugScope %func1_block %inline_func1_to_line13

               OpLine %src 6 7
         %30 = OpLoad %float %pos
               OpLine %src 6 12
         %32 = OpFOrdGreaterThan %bool %30 %float_1
               OpLine %src 6 17
               OpSelectionMerge %if_merge None
               OpBranchConditional %32 %if_true %if_merge
    %if_true = OpLabel

; Inlining information: "Lines from line 7 originally included in %if_true_block were inlined to line 13"
%scope_for_inline_if_true = OpExtInst %void %DbgExt DebugScope %if_true_block %inline_func1_to_line13

               OpLine %src 7 5
               OpStore %gl_Position %9
               OpReturn
   %if_merge = OpLabel

; Inlining information: "Line 2 originally included in %func2_block was recursively inlined to line 9 and line 13"
%scope_for_inline_func2 = OpExtInst %void %DbgExt DebugScope %func2_block %inline_recursive
               OpLine %src 2 17
         %40 = OpLoad %float %pos
               OpLine %src 2 10
         %41 = OpCompositeConstruct %v4float %40 %float_0 %float_0 %float_0

; End of inlining. Scope of function main()
%main_scope = OpExtInst %void %DbgExt DebugScope %main_block

               OpLine %src 13 3
               OpStore %gl_Position %41
               OpReturn
               OpFunctionEnd
)");
}

TEST(IrBuilder, LocalGlobalVariables) {
  // #version 310 es
  //
  // float gv1 = 10.;
  // float gv2 = 100.;
  //
  // float f() {
  //   float lv1 = gv1 + gv2;
  //   float lv2 = gv1 * gv2;
  //   return lv1 / lv2;
  // }
  //
  // void main() {
  //   float lv1 = gv1 - gv2;
  // }
  DoRoundTripCheck(
      // clang-format off
               "OpCapability Shader\n"
          "%1 = OpExtInstImport \"GLSL.std.450\"\n"
               "OpMemoryModel Logical GLSL450\n"
               "OpEntryPoint Vertex %main \"main\"\n"
               "OpSource ESSL 310\n"
               "OpName %main \"main\"\n"
               "OpName %f_ \"f(\"\n"
               "OpName %gv1 \"gv1\"\n"
               "OpName %gv2 \"gv2\"\n"
               "OpName %lv1 \"lv1\"\n"
               "OpName %lv2 \"lv2\"\n"
               "OpName %lv1_0 \"lv1\"\n"
       "%void = OpTypeVoid\n"
         "%10 = OpTypeFunction %void\n"
      "%float = OpTypeFloat 32\n"
         "%12 = OpTypeFunction %float\n"
 "%_ptr_Private_float = OpTypePointer Private %float\n"
        "%gv1 = OpVariable %_ptr_Private_float Private\n"
   "%float_10 = OpConstant %float 10\n"
        "%gv2 = OpVariable %_ptr_Private_float Private\n"
  "%float_100 = OpConstant %float 100\n"
 "%_ptr_Function_float = OpTypePointer Function %float\n"
       "%main = OpFunction %void None %10\n"
         "%17 = OpLabel\n"
      "%lv1_0 = OpVariable %_ptr_Function_float Function\n"
               "OpStore %gv1 %float_10\n"
               "OpStore %gv2 %float_100\n"
         "%18 = OpLoad %float %gv1\n"
         "%19 = OpLoad %float %gv2\n"
         "%20 = OpFSub %float %18 %19\n"
               "OpStore %lv1_0 %20\n"
               "OpReturn\n"
               "OpFunctionEnd\n"
         "%f_ = OpFunction %float None %12\n"
         "%21 = OpLabel\n"
        "%lv1 = OpVariable %_ptr_Function_float Function\n"
        "%lv2 = OpVariable %_ptr_Function_float Function\n"
         "%22 = OpLoad %float %gv1\n"
         "%23 = OpLoad %float %gv2\n"
         "%24 = OpFAdd %float %22 %23\n"
               "OpStore %lv1 %24\n"
         "%25 = OpLoad %float %gv1\n"
         "%26 = OpLoad %float %gv2\n"
         "%27 = OpFMul %float %25 %26\n"
               "OpStore %lv2 %27\n"
         "%28 = OpLoad %float %lv1\n"
         "%29 = OpLoad %float %lv2\n"
         "%30 = OpFDiv %float %28 %29\n"
               "OpReturnValue %30\n"
               "OpFunctionEnd\n");
  // clang-format on
}

TEST(IrBuilder, OpUndefOutsideFunction) {
  // #version 310 es
  // void main() {}
  const std::string text =
      // clang-format off
               "OpMemoryModel Logical GLSL450\n"
        "%int = OpTypeInt 32 1\n"
       "%uint = OpTypeInt 32 0\n"
      "%float = OpTypeFloat 32\n"
          "%4 = OpUndef %int\n"
     "%int_10 = OpConstant %int 10\n"
          "%6 = OpUndef %uint\n"
       "%bool = OpTypeBool\n"
          "%8 = OpUndef %float\n"
     "%double = OpTypeFloat 64\n";
  // clang-format on

  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  ASSERT_NE(nullptr, context);

  const auto opundef_count = std::count_if(
      context->module()->types_values_begin(),
      context->module()->types_values_end(),
      [](const Instruction& inst) { return inst.opcode() == SpvOpUndef; });
  EXPECT_EQ(3, opundef_count);

  std::vector<uint32_t> binary;
  context->module()->ToBinary(&binary, /* skip_nop = */ false);

  std::string disassembled_text;
  EXPECT_TRUE(t.Disassemble(binary, &disassembled_text));
  EXPECT_EQ(text, disassembled_text);
}

TEST(IrBuilder, OpUndefInBasicBlock) {
  DoRoundTripCheck(
      // clang-format off
               "OpMemoryModel Logical GLSL450\n"
               "OpName %main \"main\"\n"
       "%void = OpTypeVoid\n"
       "%uint = OpTypeInt 32 0\n"
     "%double = OpTypeFloat 64\n"
          "%5 = OpTypeFunction %void\n"
       "%main = OpFunction %void None %5\n"
          "%6 = OpLabel\n"
          "%7 = OpUndef %uint\n"
          "%8 = OpUndef %double\n"
               "OpReturn\n"
               "OpFunctionEnd\n");
  // clang-format on
}

TEST(IrBuilder, KeepLineDebugInfoBeforeType) {
  DoRoundTripCheck(
      // clang-format off
               "OpCapability Shader\n"
               "OpMemoryModel Logical GLSL450\n"
          "%1 = OpString \"minimal.vert\"\n"
               "OpLine %1 1 1\n"
               "OpNoLine\n"
       "%void = OpTypeVoid\n"
               "OpLine %1 2 2\n"
          "%3 = OpTypeFunction %void\n");
  // clang-format on
}

TEST(IrBuilder, KeepLineDebugInfoBeforeLabel) {
  DoRoundTripCheck(
      // clang-format off
               "OpCapability Shader\n"
               "OpMemoryModel Logical GLSL450\n"
          "%1 = OpString \"minimal.vert\"\n"
       "%void = OpTypeVoid\n"
          "%3 = OpTypeFunction %void\n"
       "%4 = OpFunction %void None %3\n"
          "%5 = OpLabel\n"
   "OpBranch %6\n"
               "OpLine %1 1 1\n"
               "OpLine %1 2 2\n"
          "%6 = OpLabel\n"
               "OpBranch %7\n"
               "OpLine %1 100 100\n"
          "%7 = OpLabel\n"
               "OpReturn\n"
               "OpFunctionEnd\n");
  // clang-format on
}

TEST(IrBuilder, KeepLineDebugInfoBeforeFunctionEnd) {
  DoRoundTripCheck(
      // clang-format off
               "OpCapability Shader\n"
               "OpMemoryModel Logical GLSL450\n"
          "%1 = OpString \"minimal.vert\"\n"
       "%void = OpTypeVoid\n"
          "%3 = OpTypeFunction %void\n"
       "%4 = OpFunction %void None %3\n"
               "OpLine %1 1 1\n"
               "OpLine %1 2 2\n"
               "OpFunctionEnd\n");
  // clang-format on
}

TEST(IrBuilder, KeepModuleProcessedInRightPlace) {
  DoRoundTripCheck(
      // clang-format off
               "OpCapability Shader\n"
               "OpMemoryModel Logical GLSL450\n"
          "%1 = OpString \"minimal.vert\"\n"
               "OpName %void \"void\"\n"
               "OpModuleProcessed \"Made it faster\"\n"
               "OpModuleProcessed \".. and smaller\"\n"
       "%void = OpTypeVoid\n");
  // clang-format on
}

// Checks the given |error_message| is reported when trying to build a module
// from the given |assembly|.
void DoErrorMessageCheck(const std::string& assembly,
                         const std::string& error_message, uint32_t line_num) {
  auto consumer = [error_message, line_num](spv_message_level_t, const char*,
                                            const spv_position_t& position,
                                            const char* m) {
    EXPECT_EQ(error_message, m);
    EXPECT_EQ(line_num, position.line);
  };

  SpirvTools t(SPV_ENV_UNIVERSAL_1_1);
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, std::move(consumer), assembly);
  EXPECT_EQ(nullptr, context);
}

TEST(IrBuilder, FunctionInsideFunction) {
  DoErrorMessageCheck("%2 = OpFunction %1 None %3\n%5 = OpFunction %4 None %6",
                      "function inside function", 2);
}

TEST(IrBuilder, MismatchOpFunctionEnd) {
  DoErrorMessageCheck("OpFunctionEnd",
                      "OpFunctionEnd without corresponding OpFunction", 1);
}

TEST(IrBuilder, OpFunctionEndInsideBasicBlock) {
  DoErrorMessageCheck(
      "%2 = OpFunction %1 None %3\n"
      "%4 = OpLabel\n"
      "OpFunctionEnd",
      "OpFunctionEnd inside basic block", 3);
}

TEST(IrBuilder, BasicBlockOutsideFunction) {
  DoErrorMessageCheck("OpCapability Shader\n%1 = OpLabel",
                      "OpLabel outside function", 2);
}

TEST(IrBuilder, OpLabelInsideBasicBlock) {
  DoErrorMessageCheck(
      "%2 = OpFunction %1 None %3\n"
      "%4 = OpLabel\n"
      "%5 = OpLabel",
      "OpLabel inside basic block", 3);
}

TEST(IrBuilder, TerminatorOutsideFunction) {
  DoErrorMessageCheck("OpReturn", "terminator instruction outside function", 1);
}

TEST(IrBuilder, TerminatorOutsideBasicBlock) {
  DoErrorMessageCheck("%2 = OpFunction %1 None %3\nOpReturn",
                      "terminator instruction outside basic block", 2);
}

TEST(IrBuilder, NotAllowedInstAppearingInFunction) {
  DoErrorMessageCheck("%2 = OpFunction %1 None %3\n%5 = OpVariable %4 Function",
                      "Non-OpFunctionParameter (opcode: 59) found inside "
                      "function but outside basic block",
                      2);
}

TEST(IrBuilder, UniqueIds) {
  const std::string text =
      // clang-format off
               "OpCapability Shader\n"
          "%1 = OpExtInstImport \"GLSL.std.450\"\n"
               "OpMemoryModel Logical GLSL450\n"
               "OpEntryPoint Vertex %main \"main\"\n"
               "OpSource ESSL 310\n"
               "OpName %main \"main\"\n"
               "OpName %f_ \"f(\"\n"
               "OpName %gv1 \"gv1\"\n"
               "OpName %gv2 \"gv2\"\n"
               "OpName %lv1 \"lv1\"\n"
               "OpName %lv2 \"lv2\"\n"
               "OpName %lv1_0 \"lv1\"\n"
       "%void = OpTypeVoid\n"
         "%10 = OpTypeFunction %void\n"
      "%float = OpTypeFloat 32\n"
         "%12 = OpTypeFunction %float\n"
 "%_ptr_Private_float = OpTypePointer Private %float\n"
        "%gv1 = OpVariable %_ptr_Private_float Private\n"
   "%float_10 = OpConstant %float 10\n"
        "%gv2 = OpVariable %_ptr_Private_float Private\n"
  "%float_100 = OpConstant %float 100\n"
 "%_ptr_Function_float = OpTypePointer Function %float\n"
       "%main = OpFunction %void None %10\n"
         "%17 = OpLabel\n"
      "%lv1_0 = OpVariable %_ptr_Function_float Function\n"
               "OpStore %gv1 %float_10\n"
               "OpStore %gv2 %float_100\n"
         "%18 = OpLoad %float %gv1\n"
         "%19 = OpLoad %float %gv2\n"
         "%20 = OpFSub %float %18 %19\n"
               "OpStore %lv1_0 %20\n"
               "OpReturn\n"
               "OpFunctionEnd\n"
         "%f_ = OpFunction %float None %12\n"
         "%21 = OpLabel\n"
        "%lv1 = OpVariable %_ptr_Function_float Function\n"
        "%lv2 = OpVariable %_ptr_Function_float Function\n"
         "%22 = OpLoad %float %gv1\n"
         "%23 = OpLoad %float %gv2\n"
         "%24 = OpFAdd %float %22 %23\n"
               "OpStore %lv1 %24\n"
         "%25 = OpLoad %float %gv1\n"
         "%26 = OpLoad %float %gv2\n"
         "%27 = OpFMul %float %25 %26\n"
               "OpStore %lv2 %27\n"
         "%28 = OpLoad %float %lv1\n"
         "%29 = OpLoad %float %lv2\n"
         "%30 = OpFDiv %float %28 %29\n"
               "OpReturnValue %30\n"
               "OpFunctionEnd\n";
  // clang-format on

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  ASSERT_NE(nullptr, context);

  std::unordered_set<uint32_t> ids;
  context->module()->ForEachInst([&ids](const Instruction* inst) {
    EXPECT_TRUE(ids.insert(inst->unique_id()).second);
  });
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
