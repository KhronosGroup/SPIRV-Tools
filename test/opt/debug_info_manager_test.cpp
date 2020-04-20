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

#include "source/opt/debug_info_manager.h"

#include <memory>
#include <string>
#include <vector>

#include "effcee/effcee.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/instruction.h"
#include "spirv-tools/libspirv.hpp"

// Constants for OpenCL.DebugInfo.100 extension instructions.

static const uint32_t kDebugDeclareOperandVariableIndex = 5;
static const uint32_t kDebugFunctionOperandFunctionIndex = 13;
static const uint32_t kDebugInlinedAtOperandLineIndex = 4;
static const uint32_t kDebugInlinedAtOperandScopeIndex = 5;
static const uint32_t kDebugInlinedAtOperandInlinedIndex = 6;

namespace spvtools {
namespace opt {
namespace analysis {
namespace {

TEST(DebugInfoManager, CloneDebugDeclare) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "OpenCL.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_var_COLOR %out_var_SV_TARGET
               OpExecutionMode %main OriginUpperLeft
          %5 = OpString "ps.hlsl"
         %14 = OpString "#line 1 \"ps.hlsl\"
void main(float in_var_color : COLOR) {
  float color = in_var_color;
}
"
         %17 = OpString "float"
         %21 = OpString "main"
         %24 = OpString "color"
               OpName %in_var_COLOR "in.var.COLOR"
               OpName %main "main"
               OpDecorate %in_var_COLOR Location 0
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
      %float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
       %void = OpTypeVoid
         %27 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%in_var_COLOR = OpVariable %_ptr_Input_float Input
         %13 = OpExtInst %void %1 DebugExpression
         %15 = OpExtInst %void %1 DebugSource %5 %14
         %16 = OpExtInst %void %1 DebugCompilationUnit 1 4 %15 HLSL
         %18 = OpExtInst %void %1 DebugTypeBasic %17 %uint_32 Float
         %20 = OpExtInst %void %1 DebugTypeFunction FlagIsProtected|FlagIsPrivate %18 %18
         %22 = OpExtInst %void %1 DebugFunction %21 %20 %15 1 1 %16 %21 FlagIsProtected|FlagIsPrivate 1 %main
         %25 = OpExtInst %void %1 DebugLocalVariable %24 %18 %15 1 20 %22 FlagIsLocal 0
       %main = OpFunction %void None %27
         %28 = OpLabel
        %100 = OpVariable %_ptr_Function_float Function
        %150 = OpVariable %_ptr_Function_float Function
        %200 = OpVariable %_ptr_Function_float Function
         %31 = OpLoad %float %in_var_COLOR
               OpStore %100 %31
         %36 = OpExtInst %void %1 DebugDeclare %25 %100 %13
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  DebugInfoManager manager(context.get());

  EXPECT_EQ(manager.CloneDebugDeclare(150, 200), nullptr);

  Instruction* dbg_decl_with_150 = manager.CloneDebugDeclare(100, 150);

  EXPECT_EQ(dbg_decl_with_150->GetOpenCL100DebugOpcode(),
            OpenCLDebugInfo100DebugDeclare);
  EXPECT_EQ(dbg_decl_with_150->GetSingleWordOperand(
                kDebugDeclareOperandVariableIndex),
            150);

  Instruction* dbg_decl_with_200 = manager.CloneDebugDeclare(150, 200);

  EXPECT_EQ(dbg_decl_with_200->GetOpenCL100DebugOpcode(),
            OpenCLDebugInfo100DebugDeclare);
  EXPECT_EQ(dbg_decl_with_200->GetSingleWordOperand(
                kDebugDeclareOperandVariableIndex),
            200);
}

TEST(DebugInfoManager, GetDebugInlinedAt) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "OpenCL.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_var_COLOR %out_var_SV_TARGET
               OpExecutionMode %main OriginUpperLeft
          %5 = OpString "ps.hlsl"
         %14 = OpString "#line 1 \"ps.hlsl\"
void main(float in_var_color : COLOR) {
  float color = in_var_color;
}
"
         %17 = OpString "float"
         %21 = OpString "main"
         %24 = OpString "color"
               OpName %in_var_COLOR "in.var.COLOR"
               OpName %main "main"
               OpDecorate %in_var_COLOR Location 0
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
      %float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
       %void = OpTypeVoid
         %27 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%in_var_COLOR = OpVariable %_ptr_Input_float Input
         %13 = OpExtInst %void %1 DebugExpression
         %15 = OpExtInst %void %1 DebugSource %5 %14
         %16 = OpExtInst %void %1 DebugCompilationUnit 1 4 %15 HLSL
         %18 = OpExtInst %void %1 DebugTypeBasic %17 %uint_32 Float
         %20 = OpExtInst %void %1 DebugTypeFunction FlagIsProtected|FlagIsPrivate %18 %18
         %22 = OpExtInst %void %1 DebugFunction %21 %20 %15 1 1 %16 %21 FlagIsProtected|FlagIsPrivate 1 %main
        %100 = OpExtInst %void %1 DebugInlinedAt 7 %22
       %main = OpFunction %void None %27
         %28 = OpLabel
         %31 = OpLoad %float %in_var_COLOR
               OpStore %100 %31
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  DebugInfoManager manager(context.get());

  EXPECT_EQ(manager.GetDebugInlinedAt(150), nullptr);
  EXPECT_EQ(manager.GetDebugInlinedAt(31), nullptr);
  EXPECT_EQ(manager.GetDebugInlinedAt(22), nullptr);

  auto* inst = manager.GetDebugInlinedAt(100);
  EXPECT_EQ(inst->GetSingleWordOperand(kDebugInlinedAtOperandLineIndex), 7);
  EXPECT_EQ(inst->GetSingleWordOperand(kDebugInlinedAtOperandScopeIndex), 22);
}

TEST(DebugInfoManager, CreateDebugInlinedAt) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "OpenCL.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_var_COLOR %out_var_SV_TARGET
               OpExecutionMode %main OriginUpperLeft
          %5 = OpString "ps.hlsl"
         %14 = OpString "#line 1 \"ps.hlsl\"
void main(float in_var_color : COLOR) {
  float color = in_var_color;
}
"
         %17 = OpString "float"
         %21 = OpString "main"
         %24 = OpString "color"
               OpName %in_var_COLOR "in.var.COLOR"
               OpName %main "main"
               OpDecorate %in_var_COLOR Location 0
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
      %float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
       %void = OpTypeVoid
         %27 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%in_var_COLOR = OpVariable %_ptr_Input_float Input
         %13 = OpExtInst %void %1 DebugExpression
         %15 = OpExtInst %void %1 DebugSource %5 %14
         %16 = OpExtInst %void %1 DebugCompilationUnit 1 4 %15 HLSL
         %18 = OpExtInst %void %1 DebugTypeBasic %17 %uint_32 Float
         %20 = OpExtInst %void %1 DebugTypeFunction FlagIsProtected|FlagIsPrivate %18 %18
         %22 = OpExtInst %void %1 DebugFunction %21 %20 %15 1 1 %16 %21 FlagIsProtected|FlagIsPrivate 1 %main
        %100 = OpExtInst %void %1 DebugInlinedAt 7 %22
       %main = OpFunction %void None %27
         %28 = OpLabel
         %31 = OpLoad %float %in_var_COLOR
               OpStore %100 %31
               OpReturn
               OpFunctionEnd
  )";

  DebugScope scope(22U, 0U);

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  DebugInfoManager manager(context.get());

  uint32_t inlined_at_id = manager.CreateDebugInlinedAt(nullptr, scope);
  auto* inlined_at = manager.GetDebugInlinedAt(inlined_at_id);
  EXPECT_NE(inlined_at, nullptr);
  EXPECT_EQ(inlined_at->GetSingleWordOperand(kDebugInlinedAtOperandLineIndex),
            1);
  EXPECT_EQ(inlined_at->GetSingleWordOperand(kDebugInlinedAtOperandScopeIndex),
            22);
  EXPECT_EQ(inlined_at->NumOperands(), kDebugInlinedAtOperandScopeIndex + 1);

  const uint32_t line_number = 77U;
  Instruction line(context.get(), SpvOpLine);
  line.SetInOperands({
      {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {5U}},
      {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {line_number}},
      {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {0U}},
  });

  inlined_at_id = manager.CreateDebugInlinedAt(&line, scope);
  inlined_at = manager.GetDebugInlinedAt(inlined_at_id);
  EXPECT_NE(inlined_at, nullptr);
  EXPECT_EQ(inlined_at->GetSingleWordOperand(kDebugInlinedAtOperandLineIndex),
            line_number);
  EXPECT_EQ(inlined_at->GetSingleWordOperand(kDebugInlinedAtOperandScopeIndex),
            22);
  EXPECT_EQ(inlined_at->NumOperands(), kDebugInlinedAtOperandScopeIndex + 1);

  scope.SetInlinedAt(100U);
  inlined_at_id = manager.CreateDebugInlinedAt(&line, scope);
  inlined_at = manager.GetDebugInlinedAt(inlined_at_id);
  EXPECT_NE(inlined_at, nullptr);
  EXPECT_EQ(inlined_at->GetSingleWordOperand(kDebugInlinedAtOperandLineIndex),
            line_number);
  EXPECT_EQ(inlined_at->GetSingleWordOperand(kDebugInlinedAtOperandScopeIndex),
            22);
  EXPECT_EQ(inlined_at->NumOperands(), kDebugInlinedAtOperandInlinedIndex + 1);
  EXPECT_EQ(
      inlined_at->GetSingleWordOperand(kDebugInlinedAtOperandInlinedIndex),
      100U);
}

TEST(DebugInfoManager, CreateDebugInfoNone) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "OpenCL.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_var_COLOR %out_var_SV_TARGET
               OpExecutionMode %main OriginUpperLeft
          %5 = OpString "ps.hlsl"
         %14 = OpString "#line 1 \"ps.hlsl\"
void main(float in_var_color : COLOR) {
  float color = in_var_color;
}
"
         %17 = OpString "float"
         %21 = OpString "main"
         %24 = OpString "color"
               OpName %in_var_COLOR "in.var.COLOR"
               OpName %main "main"
               OpDecorate %in_var_COLOR Location 0
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
      %float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
       %void = OpTypeVoid
         %27 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%in_var_COLOR = OpVariable %_ptr_Input_float Input
         %13 = OpExtInst %void %1 DebugExpression
         %15 = OpExtInst %void %1 DebugSource %5 %14
         %16 = OpExtInst %void %1 DebugCompilationUnit 1 4 %15 HLSL
         %18 = OpExtInst %void %1 DebugTypeBasic %17 %uint_32 Float
         %20 = OpExtInst %void %1 DebugTypeFunction FlagIsProtected|FlagIsPrivate %18 %18
         %22 = OpExtInst %void %1 DebugFunction %21 %20 %15 1 1 %16 %21 FlagIsProtected|FlagIsPrivate 1 %main
         %25 = OpExtInst %void %1 DebugLocalVariable %24 %18 %15 1 20 %22 FlagIsLocal 0
       %main = OpFunction %void None %27
         %28 = OpLabel
        %100 = OpVariable %_ptr_Function_float Function
         %31 = OpLoad %float %in_var_COLOR
               OpStore %100 %31
         %36 = OpExtInst %void %1 DebugDeclare %25 %100 %13
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  DebugInfoManager manager(context.get());

  Instruction* debug_info_none_inst = manager.GetDebugInfoNone();
  EXPECT_NE(debug_info_none_inst, nullptr);
  EXPECT_EQ(debug_info_none_inst->GetOpenCL100DebugOpcode(),
            OpenCLDebugInfo100DebugInfoNone);
}

TEST(DebugInfoManager, GetDebugFunction) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "OpenCL.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %200 "200" %in_var_COLOR %out_var_SV_TARGET
               OpExecutionMode %200 OriginUpperLeft
          %5 = OpString "ps.hlsl"
         %14 = OpString "#line 1 \"ps.hlsl\"
void 200(float in_var_color : COLOR) {
  float color = in_var_color;
}
"
         %17 = OpString "float"
         %21 = OpString "200"
         %24 = OpString "color"
               OpName %in_var_COLOR "in.var.COLOR"
               OpName %200 "200"
               OpDecorate %in_var_COLOR Location 0
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
      %float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
       %void = OpTypeVoid
         %27 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%in_var_COLOR = OpVariable %_ptr_Input_float Input
         %13 = OpExtInst %void %1 DebugExpression
         %15 = OpExtInst %void %1 DebugSource %5 %14
         %16 = OpExtInst %void %1 DebugCompilationUnit 1 4 %15 HLSL
         %18 = OpExtInst %void %1 DebugTypeBasic %17 %uint_32 Float
         %20 = OpExtInst %void %1 DebugTypeFunction FlagIsProtected|FlagIsPrivate %18 %18
         %22 = OpExtInst %void %1 DebugFunction %21 %20 %15 1 1 %16 %21 FlagIsProtected|FlagIsPrivate 1 %200
         %25 = OpExtInst %void %1 DebugLocalVariable %24 %18 %15 1 20 %22 FlagIsLocal 0
       %200 = OpFunction %void None %27
         %28 = OpLabel
        %100 = OpVariable %_ptr_Function_float Function
         %31 = OpLoad %float %in_var_COLOR
               OpStore %100 %31
         %36 = OpExtInst %void %1 DebugDeclare %25 %100 %13
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  DebugInfoManager manager(context.get());

  EXPECT_EQ(manager.GetDebugFunction(100), nullptr);
  EXPECT_EQ(manager.GetDebugFunction(150), nullptr);

  Instruction* dbg_fn = manager.GetDebugFunction(200);

  EXPECT_EQ(dbg_fn->GetOpenCL100DebugOpcode(), OpenCLDebugInfo100DebugFunction);
  EXPECT_EQ(dbg_fn->GetSingleWordOperand(kDebugFunctionOperandFunctionIndex),
            200);
}

TEST(DebugInfoManager, CloneDebugInlinedAt) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "OpenCL.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_var_COLOR %out_var_SV_TARGET
               OpExecutionMode %main OriginUpperLeft
          %5 = OpString "ps.hlsl"
         %14 = OpString "#line 1 \"ps.hlsl\"
void main(float in_var_color : COLOR) {
  float color = in_var_color;
}
"
         %17 = OpString "float"
         %21 = OpString "main"
         %24 = OpString "color"
               OpName %in_var_COLOR "in.var.COLOR"
               OpName %main "main"
               OpDecorate %in_var_COLOR Location 0
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
      %float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
       %void = OpTypeVoid
         %27 = OpTypeFunction %void
%_ptr_Function_float = OpTypePointer Function %float
%in_var_COLOR = OpVariable %_ptr_Input_float Input
         %13 = OpExtInst %void %1 DebugExpression
         %15 = OpExtInst %void %1 DebugSource %5 %14
         %16 = OpExtInst %void %1 DebugCompilationUnit 1 4 %15 HLSL
         %18 = OpExtInst %void %1 DebugTypeBasic %17 %uint_32 Float
         %20 = OpExtInst %void %1 DebugTypeFunction FlagIsProtected|FlagIsPrivate %18 %18
         %22 = OpExtInst %void %1 DebugFunction %21 %20 %15 1 1 %16 %21 FlagIsProtected|FlagIsPrivate 1 %main
        %100 = OpExtInst %void %1 DebugInlinedAt 7 %22
       %main = OpFunction %void None %27
         %28 = OpLabel
         %31 = OpLoad %float %in_var_COLOR
               OpStore %100 %31
               OpReturn
               OpFunctionEnd
  )";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  DebugInfoManager manager(context.get());

  EXPECT_EQ(manager.CloneDebugInlinedAt(150), nullptr);
  EXPECT_EQ(manager.CloneDebugInlinedAt(22), nullptr);

  auto* inst = manager.CloneDebugInlinedAt(100);
  EXPECT_EQ(inst->GetSingleWordOperand(kDebugInlinedAtOperandLineIndex), 7);
  EXPECT_EQ(inst->GetSingleWordOperand(kDebugInlinedAtOperandScopeIndex), 22);
  EXPECT_EQ(inst->NumOperands(), kDebugInlinedAtOperandScopeIndex + 1);

  Instruction* before_100 = nullptr;
  for (auto it = context->module()->ext_inst_debuginfo_begin();
       it != context->module()->ext_inst_debuginfo_end(); ++it) {
    if (it->result_id() == 100) break;
    before_100 = &*it;
  }
  EXPECT_NE(inst, before_100);

  inst = manager.CloneDebugInlinedAt(100, manager.GetDebugInlinedAt(100));
  EXPECT_EQ(inst->GetSingleWordOperand(kDebugInlinedAtOperandLineIndex), 7);
  EXPECT_EQ(inst->GetSingleWordOperand(kDebugInlinedAtOperandScopeIndex), 22);
  EXPECT_EQ(inst->NumOperands(), kDebugInlinedAtOperandScopeIndex + 1);

  before_100 = nullptr;
  for (auto it = context->module()->ext_inst_debuginfo_begin();
       it != context->module()->ext_inst_debuginfo_end(); ++it) {
    if (it->result_id() == 100) break;
    before_100 = &*it;
  }
  EXPECT_EQ(inst, before_100);
}

}  // namespace
}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
