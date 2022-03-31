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

class FlattenArrayMatrixStageVarTest : public PassTest<::testing::Test> {
 public:
  FlattenArrayMatrixStageVarTest()
      : consumer_([this](spv_message_level_t level, const char*,
                         const spv_position_t& position, const char* message) {
          if (!error_message_.empty()) error_message_ += "\n";
          switch (level) {
            case SPV_MSG_FATAL:
            case SPV_MSG_INTERNAL_ERROR:
            case SPV_MSG_ERROR:
              error_message_ += "ERROR";
              break;
            case SPV_MSG_WARNING:
              error_message_ += "WARNING";
              break;
            case SPV_MSG_INFO:
              error_message_ += "INFO";
              break;
            case SPV_MSG_DEBUG:
              error_message_ += "DEBUG";
              break;
          }
          error_message_ +=
              ": " + std::to_string(position.index) + ": " + message;
        }) {}

  std::tuple<std::string, Pass::Status> RunPass(
      const std::string& text, spv_target_env env, uint32_t disassemble_options,
      const std::vector<StageVariableInfo>& info) {
    std::unique_ptr<IRContext> context_ =
        spvtools::BuildModule(env, consumer_, text);
    std::string optimized_asm;
    if (!context_.get())
      return std::make_tuple(optimized_asm, Pass::Status::Failure);

    PassManager manager;
    manager.SetMessageConsumer(consumer_);
    manager.AddPass<FlattenArrayMatrixStageVariable>(info);

    const auto status = manager.Run(context_.get());

    if (status != Pass::Status::Failure) {
      std::vector<uint32_t> binary;
      context_->module()->ToBinary(&binary, false);

      SpirvTools tools(env);
      EXPECT_TRUE(
          tools.Disassemble(binary, &optimized_asm, disassemble_options))
          << "Disassembling failed for shader:\n"
          << text << std::endl;
    }
    return std::make_tuple(optimized_asm, status);
  }

  std::string GetErrorMessage() const { return error_message_; }

  void TearDown() override { error_message_.clear(); }

 private:
  spvtools::MessageConsumer consumer_;
  std::string error_message_;
};

TEST_F(FlattenArrayMatrixStageVarTest, Test) {
  const std::string text = R"(
               OpCapability Tessellation
               OpExtension "SPV_GOOGLE_hlsl_functionality1"
               OpMemoryModel Logical GLSL450
               OpEntryPoint TessellationControl %main_hs "main_hs" %in_var_POSITION %in_var_VERTCOLOR0 %gl_InvocationID %out_var_POSITION %out_var_VERTCOLOR0 %out_var_TESSFACTOR0 %gl_TessLevelOuter %gl_TessLevelInner
               OpExecutionMode %main_hs Triangles
               OpExecutionMode %main_hs SpacingFractionalOdd
               OpExecutionMode %main_hs VertexOrderCw
               OpExecutionMode %main_hs OutputVertices 3
               OpSource HLSL 600
               OpName %Positions "Positions"
               OpName %Colors "Colors"
               OpName %type__Globals "type.$Globals"
               OpMemberName %type__Globals 0 "Tessellation_factor"
               OpName %_Globals "$Globals"
               OpName %in_var_POSITION "in.var.POSITION"
               OpName %in_var_VERTCOLOR0 "in.var.VERTCOLOR0"
               OpName %out_var_POSITION "out.var.POSITION"
               OpName %out_var_VERTCOLOR0 "out.var.VERTCOLOR0"
               OpName %out_var_TESSFACTOR0 "out.var.TESSFACTOR0"
               OpName %main_hs "main_hs"
               OpName %VS_OUTPUT "VS_OUTPUT"
               OpMemberName %VS_OUTPUT 0 "pos"
               OpMemberName %VS_OUTPUT 1 "color"
               OpName %param_var_inputpoints "param.var.inputpoints"
               OpName %param_var_i "param.var.i"
               OpName %HS_OUTPUT "HS_OUTPUT"
               OpMemberName %HS_OUTPUT 0 "pos"
               OpMemberName %HS_OUTPUT 1 "color"
               OpMemberName %HS_OUTPUT 2 "tess_factor"
               OpName %temp_var_hullMainRetVal "temp.var.hullMainRetVal"
               OpName %if_true "if.true"
               OpName %HS_PATCH_OUTPUT "HS_PATCH_OUTPUT"
               OpMemberName %HS_PATCH_OUTPUT 0 "tess_factor"
               OpMemberName %HS_PATCH_OUTPUT 1 "inside_tess_factor"
               OpName %if_merge "if.merge"
               OpName %main_hs_patch "main_hs_patch"
               OpName %patch "patch"
               OpName %bb_entry "bb.entry"
               OpName %OUT "OUT"
               OpName %src_main_hs "src.main_hs"
               OpName %inputpoints "inputpoints"
               OpName %i "i"
               OpName %bb_entry_0 "bb.entry"
               OpName %OUT_0 "OUT"
               OpDecorateString %in_var_POSITION UserSemantic "POSITION"
               OpDecorateString %in_var_VERTCOLOR0 UserSemantic "VERTCOLOR0"
               OpDecorate %gl_InvocationID BuiltIn InvocationId
               OpDecorateString %gl_InvocationID UserSemantic "SV_OutputControlPointID"
               OpDecorateString %out_var_POSITION UserSemantic "POSITION"
               OpDecorateString %out_var_VERTCOLOR0 UserSemantic "VERTCOLOR0"
               OpDecorateString %out_var_TESSFACTOR0 UserSemantic "TESSFACTOR0"
               OpDecorate %gl_TessLevelOuter BuiltIn TessLevelOuter
               OpDecorateString %gl_TessLevelOuter UserSemantic "SV_TessFactor"
               OpDecorate %gl_TessLevelOuter Patch
               OpDecorate %gl_TessLevelInner BuiltIn TessLevelInner
               OpDecorateString %gl_TessLevelInner UserSemantic "SV_InsideTessFactor"
               OpDecorate %gl_TessLevelInner Patch
               OpDecorate %in_var_POSITION Location 0
               OpDecorate %in_var_VERTCOLOR0 Location 3
               OpDecorate %out_var_POSITION Location 0
               OpDecorate %out_var_VERTCOLOR0 Location 3
               OpDecorate %out_var_TESSFACTOR0 Location 0
               OpDecorate %out_var_TESSFACTOR0 Component 3
               OpDecorate %_Globals DescriptorSet 0
               OpDecorate %_Globals Binding 0
               OpMemberDecorate %type__Globals 0 Offset 0
               OpDecorate %type__Globals Block
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
 %float_n0_5 = OpConstant %float -0.5
  %float_0_5 = OpConstant %float 0.5
    %float_1 = OpConstant %float 1
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
        %int = OpTypeInt 32 1
      %int_2 = OpConstant %int 2
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
    %float_3 = OpConstant %float 3
     %uint_3 = OpConstant %uint 3
    %v2float = OpTypeVector %float 2
%_arr_v2float_uint_3 = OpTypeArray %v2float %uint_3
%_ptr_Private__arr_v2float_uint_3 = OpTypePointer Private %_arr_v2float_uint_3
    %v3float = OpTypeVector %float 3
%_arr_v3float_uint_3 = OpTypeArray %v3float %uint_3
%_ptr_Private__arr_v3float_uint_3 = OpTypePointer Private %_arr_v3float_uint_3
%type__Globals = OpTypeStruct %float
%_ptr_Uniform_type__Globals = OpTypePointer Uniform %type__Globals
%_arr__arr_v3float_uint_3_uint_3 = OpTypeArray %_arr_v3float_uint_3 %uint_3
%_ptr_Input__arr__arr_v3float_uint_3_uint_3 = OpTypePointer Input %_arr__arr_v3float_uint_3_uint_3
%_ptr_Input__arr_v3float_uint_3 = OpTypePointer Input %_arr_v3float_uint_3
%_ptr_Input_uint = OpTypePointer Input %uint
%_ptr_Output__arr__arr_v3float_uint_3_uint_3 = OpTypePointer Output %_arr__arr_v3float_uint_3_uint_3
%_ptr_Output__arr_v3float_uint_3 = OpTypePointer Output %_arr_v3float_uint_3
%_arr_float_uint_3 = OpTypeArray %float %uint_3
%_ptr_Output__arr_float_uint_3 = OpTypePointer Output %_arr_float_uint_3
     %uint_4 = OpConstant %uint 4
%_arr_float_uint_4 = OpTypeArray %float %uint_4
%_ptr_Output__arr_float_uint_4 = OpTypePointer Output %_arr_float_uint_4
%_arr_float_uint_2 = OpTypeArray %float %uint_2
%_ptr_Output__arr_float_uint_2 = OpTypePointer Output %_arr_float_uint_2
       %void = OpTypeVoid
         %50 = OpTypeFunction %void
  %VS_OUTPUT = OpTypeStruct %_arr_v3float_uint_3 %v3float
%_arr_VS_OUTPUT_uint_3 = OpTypeArray %VS_OUTPUT %uint_3
%_ptr_Function__arr_VS_OUTPUT_uint_3 = OpTypePointer Function %_arr_VS_OUTPUT_uint_3
%_ptr_Function_uint = OpTypePointer Function %uint
  %HS_OUTPUT = OpTypeStruct %_arr_v3float_uint_3 %v3float %float
%_arr_HS_OUTPUT_uint_3 = OpTypeArray %HS_OUTPUT %uint_3
%_ptr_Function__arr_HS_OUTPUT_uint_3 = OpTypePointer Function %_arr_HS_OUTPUT_uint_3
%_ptr_Output_v3float = OpTypePointer Output %v3float
%_ptr_Output_float = OpTypePointer Output %float
%_ptr_Function_HS_OUTPUT = OpTypePointer Function %HS_OUTPUT
%_ptr_Function__arr_v3float_uint_3 = OpTypePointer Function %_arr_v3float_uint_3
%_ptr_Function_v3float = OpTypePointer Function %v3float
%_ptr_Function_float = OpTypePointer Function %float
       %bool = OpTypeBool
%HS_PATCH_OUTPUT = OpTypeStruct %_arr_float_uint_3 %float
        %143 = OpTypeFunction %HS_PATCH_OUTPUT %_ptr_Function__arr_HS_OUTPUT_uint_3
%_ptr_Function_HS_PATCH_OUTPUT = OpTypePointer Function %HS_PATCH_OUTPUT
        %168 = OpTypeFunction %HS_OUTPUT %_ptr_Function__arr_VS_OUTPUT_uint_3 %_ptr_Function_uint
%_ptr_Uniform_float = OpTypePointer Uniform %float
  %Positions = OpVariable %_ptr_Private__arr_v2float_uint_3 Private
     %Colors = OpVariable %_ptr_Private__arr_v3float_uint_3 Private
   %_Globals = OpVariable %_ptr_Uniform_type__Globals Uniform
%in_var_POSITION = OpVariable %_ptr_Input__arr__arr_v3float_uint_3_uint_3 Input
%in_var_VERTCOLOR0 = OpVariable %_ptr_Input__arr_v3float_uint_3 Input
%gl_InvocationID = OpVariable %_ptr_Input_uint Input
%out_var_POSITION = OpVariable %_ptr_Output__arr__arr_v3float_uint_3_uint_3 Output
%out_var_VERTCOLOR0 = OpVariable %_ptr_Output__arr_v3float_uint_3 Output
%out_var_TESSFACTOR0 = OpVariable %_ptr_Output__arr_float_uint_3 Output
%gl_TessLevelOuter = OpVariable %_ptr_Output__arr_float_uint_4 Output
%gl_TessLevelInner = OpVariable %_ptr_Output__arr_float_uint_2 Output
    %main_hs = OpFunction %void None %50
         %51 = OpLabel
%param_var_inputpoints = OpVariable %_ptr_Function__arr_VS_OUTPUT_uint_3 Function
%param_var_i = OpVariable %_ptr_Function_uint Function
%temp_var_hullMainRetVal = OpVariable %_ptr_Function__arr_HS_OUTPUT_uint_3 Function
         %62 = OpCompositeConstruct %v2float %float_0 %float_n0_5
         %63 = OpCompositeConstruct %v2float %float_0_5 %float_0_5
         %64 = OpCompositeConstruct %v2float %float_n0_5 %float_0_5
         %65 = OpCompositeConstruct %_arr_v2float_uint_3 %62 %63 %64
               OpStore %Positions %65
         %66 = OpCompositeConstruct %v3float %float_1 %float_0 %float_0
         %67 = OpCompositeConstruct %v3float %float_0 %float_1 %float_0
         %68 = OpCompositeConstruct %v3float %float_0 %float_0 %float_1
         %69 = OpCompositeConstruct %_arr_v3float_uint_3 %66 %67 %68
               OpStore %Colors %69
         %70 = OpLoad %_arr__arr_v3float_uint_3_uint_3 %in_var_POSITION
         %71 = OpLoad %_arr_v3float_uint_3 %in_var_VERTCOLOR0
         %72 = OpCompositeExtract %_arr_v3float_uint_3 %70 0
         %73 = OpCompositeExtract %v3float %71 0
         %74 = OpCompositeConstruct %VS_OUTPUT %72 %73
         %75 = OpCompositeExtract %_arr_v3float_uint_3 %70 1
         %76 = OpCompositeExtract %v3float %71 1
         %77 = OpCompositeConstruct %VS_OUTPUT %75 %76
         %78 = OpCompositeExtract %_arr_v3float_uint_3 %70 2
         %79 = OpCompositeExtract %v3float %71 2
         %80 = OpCompositeConstruct %VS_OUTPUT %78 %79
         %81 = OpCompositeConstruct %_arr_VS_OUTPUT_uint_3 %74 %77 %80
               OpStore %param_var_inputpoints %81
         %82 = OpLoad %uint %gl_InvocationID
               OpStore %param_var_i %82
         %83 = OpFunctionCall %HS_OUTPUT %src_main_hs %param_var_inputpoints %param_var_i
         %85 = OpCompositeExtract %_arr_v3float_uint_3 %83 0
         %86 = OpAccessChain %_ptr_Output__arr_v3float_uint_3 %out_var_POSITION %82
               OpStore %86 %85
         %87 = OpCompositeExtract %v3float %83 1
         %89 = OpAccessChain %_ptr_Output_v3float %out_var_VERTCOLOR0 %82
               OpStore %89 %87
         %90 = OpCompositeExtract %float %83 2
         %92 = OpAccessChain %_ptr_Output_float %out_var_TESSFACTOR0 %82
               OpStore %92 %90
               OpControlBarrier %uint_2 %uint_4 %uint_0
         %94 = OpAccessChain %_ptr_Function_HS_OUTPUT %temp_var_hullMainRetVal %uint_0
         %96 = OpAccessChain %_ptr_Function__arr_v3float_uint_3 %94 %uint_0
         %97 = OpAccessChain %_ptr_Output__arr_v3float_uint_3 %out_var_POSITION %uint_0
         %98 = OpLoad %_arr_v3float_uint_3 %97
               OpStore %96 %98
        %100 = OpAccessChain %_ptr_Function_v3float %94 %uint_1
        %101 = OpAccessChain %_ptr_Output_v3float %out_var_VERTCOLOR0 %uint_0
        %102 = OpLoad %v3float %101
               OpStore %100 %102
        %104 = OpAccessChain %_ptr_Function_float %94 %uint_2
        %105 = OpAccessChain %_ptr_Output_float %out_var_TESSFACTOR0 %uint_0
        %106 = OpLoad %float %105
               OpStore %104 %106
        %107 = OpAccessChain %_ptr_Function_HS_OUTPUT %temp_var_hullMainRetVal %uint_1
        %108 = OpAccessChain %_ptr_Function__arr_v3float_uint_3 %107 %uint_0
        %109 = OpAccessChain %_ptr_Output__arr_v3float_uint_3 %out_var_POSITION %uint_1
        %110 = OpLoad %_arr_v3float_uint_3 %109
               OpStore %108 %110
        %111 = OpAccessChain %_ptr_Function_v3float %107 %uint_1
        %112 = OpAccessChain %_ptr_Output_v3float %out_var_VERTCOLOR0 %uint_1
        %113 = OpLoad %v3float %112
               OpStore %111 %113
        %114 = OpAccessChain %_ptr_Function_float %107 %uint_2
        %115 = OpAccessChain %_ptr_Output_float %out_var_TESSFACTOR0 %uint_1
        %116 = OpLoad %float %115
               OpStore %114 %116
        %117 = OpAccessChain %_ptr_Function_HS_OUTPUT %temp_var_hullMainRetVal %uint_2
        %118 = OpAccessChain %_ptr_Function__arr_v3float_uint_3 %117 %uint_0
        %119 = OpAccessChain %_ptr_Output__arr_v3float_uint_3 %out_var_POSITION %uint_2
        %120 = OpLoad %_arr_v3float_uint_3 %119
               OpStore %118 %120
        %121 = OpAccessChain %_ptr_Function_v3float %117 %uint_1
        %122 = OpAccessChain %_ptr_Output_v3float %out_var_VERTCOLOR0 %uint_2
        %123 = OpLoad %v3float %122
               OpStore %121 %123
        %124 = OpAccessChain %_ptr_Function_float %117 %uint_2
        %125 = OpAccessChain %_ptr_Output_float %out_var_TESSFACTOR0 %uint_2
        %126 = OpLoad %float %125
               OpStore %124 %126
        %128 = OpIEqual %bool %82 %uint_0
               OpSelectionMerge %if_merge None
               OpBranchConditional %128 %if_true %if_merge
    %if_true = OpLabel
        %132 = OpFunctionCall %HS_PATCH_OUTPUT %main_hs_patch %temp_var_hullMainRetVal
        %134 = OpCompositeExtract %_arr_float_uint_3 %132 0
        %135 = OpAccessChain %_ptr_Output_float %gl_TessLevelOuter %uint_0
        %136 = OpCompositeExtract %float %134 0
               OpStore %135 %136
        %137 = OpAccessChain %_ptr_Output_float %gl_TessLevelOuter %uint_1
        %138 = OpCompositeExtract %float %134 1
               OpStore %137 %138
        %139 = OpAccessChain %_ptr_Output_float %gl_TessLevelOuter %uint_2
        %140 = OpCompositeExtract %float %134 2
               OpStore %139 %140
        %141 = OpCompositeExtract %float %132 1
        %142 = OpAccessChain %_ptr_Output_float %gl_TessLevelInner %uint_0
               OpStore %142 %141
               OpBranch %if_merge
   %if_merge = OpLabel
               OpReturn
               OpFunctionEnd
%main_hs_patch = OpFunction %HS_PATCH_OUTPUT None %143
      %patch = OpFunctionParameter %_ptr_Function__arr_HS_OUTPUT_uint_3
   %bb_entry = OpLabel
        %OUT = OpVariable %_ptr_Function_HS_PATCH_OUTPUT Function
        %148 = OpAccessChain %_ptr_Function_float %patch %uint_0 %int_2
        %149 = OpLoad %float %148
        %150 = OpAccessChain %_ptr_Function_float %OUT %int_0 %int_0
               OpStore %150 %149
        %151 = OpAccessChain %_ptr_Function_float %patch %uint_1 %int_2
        %152 = OpLoad %float %151
        %153 = OpAccessChain %_ptr_Function_float %OUT %int_0 %int_1
               OpStore %153 %152
        %154 = OpAccessChain %_ptr_Function_float %patch %uint_2 %int_2
        %155 = OpLoad %float %154
        %156 = OpAccessChain %_ptr_Function_float %OUT %int_0 %int_2
               OpStore %156 %155
        %157 = OpAccessChain %_ptr_Function_float %patch %uint_0 %int_2
        %158 = OpLoad %float %157
        %159 = OpAccessChain %_ptr_Function_float %patch %uint_1 %int_2
        %160 = OpLoad %float %159
        %161 = OpFAdd %float %158 %160
        %162 = OpAccessChain %_ptr_Function_float %patch %uint_2 %int_2
        %163 = OpLoad %float %162
        %164 = OpFAdd %float %161 %163
        %165 = OpFDiv %float %164 %float_3
        %166 = OpAccessChain %_ptr_Function_float %OUT %int_1
               OpStore %166 %165
        %167 = OpLoad %HS_PATCH_OUTPUT %OUT
               OpReturnValue %167
               OpFunctionEnd
%src_main_hs = OpFunction %HS_OUTPUT None %168
%inputpoints = OpFunctionParameter %_ptr_Function__arr_VS_OUTPUT_uint_3
          %i = OpFunctionParameter %_ptr_Function_uint
 %bb_entry_0 = OpLabel
      %OUT_0 = OpVariable %_ptr_Function_HS_OUTPUT Function
        %173 = OpLoad %uint %i
        %174 = OpAccessChain %_ptr_Function__arr_v3float_uint_3 %inputpoints %173 %int_0
        %175 = OpLoad %_arr_v3float_uint_3 %174
        %176 = OpAccessChain %_ptr_Function__arr_v3float_uint_3 %OUT_0 %int_0
               OpStore %176 %175
        %177 = OpLoad %uint %i
        %178 = OpAccessChain %_ptr_Function_v3float %inputpoints %177 %int_1
        %179 = OpLoad %v3float %178
        %180 = OpAccessChain %_ptr_Function_v3float %OUT_0 %int_1
               OpStore %180 %179
        %182 = OpAccessChain %_ptr_Uniform_float %_Globals %int_0
        %183 = OpLoad %float %182
        %184 = OpAccessChain %_ptr_Function_float %OUT_0 %int_2
               OpStore %184 %183
        %185 = OpLoad %HS_OUTPUT %OUT_0
               OpReturnValue %185
               OpFunctionEnd
  )";

  std::vector<StageVariableInfo> info({{0, 0, 3, true}, {0, 0, 3, false}});
  std::string result;
  std::tie(result, std::ignore) =
      RunPass(text, SPV_ENV_UNIVERSAL_1_0,
              SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                  SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES,
              info);
  std::cout << result << std::endl;
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
