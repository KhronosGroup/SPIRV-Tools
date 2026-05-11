// Copyright (c) 2026 The Khronos Group Inc.
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

// Tests displaying ShaderDebugInfo in validation error messages

#include <string>

#include "gmock/gmock.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;

using ValidateShaderDebugInfo = spvtest::ValidateBase<bool>;

TEST_F(ValidateShaderDebugInfo, DebugLineSingleLine) {
  const std::string str = R"(
               OpCapability Shader
               OpExtension "SPV_KHR_non_semantic_info"
          %1 = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
          %2 = OpString "a.comp"
         %19 = OpString "#version 450
layout(set = 0, binding = 1, std430) buffer SSBO {
    vec4 data;
};

void main() {
    data = vec4(0.0);
}"
               OpDecorate %SSBO Block
               OpMemberDecorate %SSBO 0 Offset 0
               OpDecorate %_ Binding 1
               OpDecorate %_ DescriptorSet 0
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
     %uint_0 = OpConstant %uint 0
         %18 = OpExtInst %void %1 DebugSource %2 %19
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %SSBO = OpTypeStruct %v4float
%_ptr_StorageBuffer_SSBO = OpTypePointer StorageBuffer %SSBO
    %uint_12 = OpConstant %uint 12
          %_ = OpVariable %_ptr_StorageBuffer_SSBO StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %50 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_StorageBuffer_v4float = OpTypePointer StorageBuffer %v4float
     %uint_7 = OpConstant %uint 7
       %main = OpFunction %void None %5
         %15 = OpLabel
         %54 = OpExtInst %void %1 DebugLine %18 %uint_7 %uint_7 %uint_0 %uint_0
         ;; invalid here
         %53 = OpAccessChain %_ptr_StorageBuffer_v4float %_ %int_0 %int_0 %int_0
               OpStore %53 %50
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(str.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(), HasSubstr(R"(  --> a.comp:7:0
  |
7 |     data = vec4(0.0);
  |)"));
}

TEST_F(ValidateShaderDebugInfo, DebugLineMultieLine) {
  const std::string str = R"(
               OpCapability Shader
               OpExtension "SPV_KHR_non_semantic_info"
          %1 = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
          %2 = OpString "a.comp"
         %19 = OpString "#version 450
layout(set = 0, binding = 1, std430) buffer SSBO {
    vec4 data;
};

void main() {
    data = vec4(0.0);
}"
               OpDecorate %SSBO Block
               OpMemberDecorate %SSBO 0 Offset 0
               OpDecorate %_ Binding 1
               OpDecorate %_ DescriptorSet 0
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
     %uint_0 = OpConstant %uint 0
         %18 = OpExtInst %void %1 DebugSource %2 %19
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %SSBO = OpTypeStruct %v4float
%_ptr_StorageBuffer_SSBO = OpTypePointer StorageBuffer %SSBO
    %uint_12 = OpConstant %uint 12
          %_ = OpVariable %_ptr_StorageBuffer_SSBO StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %50 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_StorageBuffer_v4float = OpTypePointer StorageBuffer %v4float
     %uint_6 = OpConstant %uint 6
     %uint_8 = OpConstant %uint 8
       %main = OpFunction %void None %5
         %15 = OpLabel
         %54 = OpExtInst %void %1 DebugLine %18 %uint_6 %uint_8 %uint_0 %uint_0
         ;; invalid here
         %53 = OpAccessChain %_ptr_StorageBuffer_v4float %_ %int_0 %int_0 %int_0
               OpStore %53 %50
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(str.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(), HasSubstr(R"(  --> a.comp:6:0
  |
6 | void main() {
7 |     data = vec4(0.0);
8 | }
  |)"));
}

TEST_F(ValidateShaderDebugInfo, DebugSourceContinued) {
  const std::string str = R"(
               OpCapability Shader
               OpExtension "SPV_KHR_non_semantic_info"
          %1 = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
          %2 = OpString "a.comp"
         %text = OpString "#version 450
layout(set = 0, binding = 1, std430) buffer SSBO {
    vec4 data;
};

"
          %text1 = OpString "void main() {"
          %text2 = OpString "
    data = vec4"
          %text3 = OpString "(0.0);
}"
               OpDecorate %SSBO Block
               OpMemberDecorate %SSBO 0 Offset 0
               OpDecorate %_ Binding 1
               OpDecorate %_ DescriptorSet 0
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
     %uint_0 = OpConstant %uint 0

   %d_source = OpExtInst %void %1 DebugSource %2 %text
         %d1 = OpExtInst %void %1 DebugSourceContinued %text1
         %d2 = OpExtInst %void %1 DebugSourceContinued %text2
         %d3 = OpExtInst %void %1 DebugSourceContinued %text3

         %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %SSBO = OpTypeStruct %v4float
%_ptr_StorageBuffer_SSBO = OpTypePointer StorageBuffer %SSBO
    %uint_12 = OpConstant %uint 12
          %_ = OpVariable %_ptr_StorageBuffer_SSBO StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %50 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_StorageBuffer_v4float = OpTypePointer StorageBuffer %v4float
     %uint_6 = OpConstant %uint 6
     %uint_8 = OpConstant %uint 8
       %main = OpFunction %void None %5
         %15 = OpLabel
         %54 = OpExtInst %void %1 DebugLine %d_source %uint_6 %uint_8 %uint_0 %uint_0
         ;; invalid here
         %53 = OpAccessChain %_ptr_StorageBuffer_v4float %_ %int_0 %int_0 %int_0
               OpStore %53 %50
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(str.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(getDiagnosticString(), HasSubstr(R"(  --> a.comp:6:0
  |
6 | void main() {
7 |     data = vec4(0.0);
8 | }
  |)"));
}

// Make sure we can handle various instructions where there DebugInfo is unused
TEST_F(ValidateShaderDebugInfo, UnusedOpFunction) {
  const std::string str = R"(
     OpCapability Shader
     OpCapability Linkage
     OpExtension "SPV_KHR_non_semantic_info"
%d = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
     OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%4 = OpFunction %1 None %2
%5 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpFunction Function Type <id> '3[%uint]' is not a function type."));
}

TEST_F(ValidateShaderDebugInfo, VariableBasic) {
  const std::string str = R"(
               OpCapability Shader
               OpExtension "SPV_KHR_non_semantic_info"
          %1 = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %x
               OpExecutionMode %main LocalSize 1 1 1
          %2 = OpString "a.comp"
          %8 = OpString "uint"
         %16 = OpString "main"
         %19 = OpString "#version 450
layout(set = 0, binding = 1, std430) buffer SSBO {
    vec4 data;
} x;

void main() {
    x.data = vec4(0.0);
}"
         %25 = OpString "float"
         %31 = OpString "data"
         %34 = OpString "SSBO"
         %40 = OpString "x"
         %46 = OpString "int"
               OpDecorate %SSBO Block
               OpMemberDecorate %SSBO 0 Offset 0
               OpDecorate %x Binding 1
               OpDecorate %x DescriptorSet 0
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
     %uint_6 = OpConstant %uint 6
     %uint_0 = OpConstant %uint 0
          %9 = OpExtInst %void %1 DebugTypeBasic %8 %uint_32 %uint_6 %uint_0
     %uint_3 = OpConstant %uint 3
          %6 = OpExtInst %void %1 DebugTypeFunction %uint_3 %void
         %18 = OpExtInst %void %1 DebugSource %2 %19
     %uint_1 = OpConstant %uint 1
     %uint_4 = OpConstant %uint 4
     %uint_2 = OpConstant %uint 2
         %20 = OpExtInst %void %1 DebugCompilationUnit %uint_1 %uint_4 %18 %uint_2
         %17 = OpExtInst %void %1 DebugFunction %16 %6 %18 %uint_6 %uint_0 %20 %16 %uint_3 %uint_6
      %float = OpTypeFloat 32
         %26 = OpExtInst %void %1 DebugTypeBasic %25 %uint_32 %uint_3 %uint_0
    %v4float = OpTypeVector %float 4
         %28 = OpExtInst %void %1 DebugTypeVector %26 %uint_4
       %SSBO = OpTypeStruct %v4float
    %uint_10 = OpConstant %uint 10
         %30 = OpExtInst %void %1 DebugTypeMember %31 %28 %18 %uint_3 %uint_10 %uint_0 %uint_0 %uint_3
         %33 = OpExtInst %void %1 DebugTypeComposite %34 %uint_1 %18 %uint_2 %uint_0 %20 %34 %uint_0 %uint_3 %30
%_ptr_StorageBuffer_SSBO = OpTypePointer Uniform %SSBO
    %uint_12 = OpConstant %uint 12
         %37 = OpExtInst %void %1 DebugTypePointer %33 %uint_12 %uint_0
          %x = OpVariable %_ptr_StorageBuffer_SSBO StorageBuffer
     %uint_8 = OpConstant %uint 8
         %39 = OpExtInst %void %1 DebugGlobalVariable %40 %33 %18 %uint_2 %uint_0 %20 %40 %x %uint_8
    %float_0 = OpConstant %float 0
         %50 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_StorageBuffer_v4float = OpTypePointer StorageBuffer %v4float
       %main = OpFunction %void None %5
         %15 = OpLabel
         %53 = OpAccessChain %_ptr_StorageBuffer_v4float %x %uint_0
               OpStore %53 %50
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(str.c_str(), SPV_ENV_VULKAN_1_3);
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_3));
  EXPECT_THAT(getDiagnosticString(), HasSubstr(R"(  --> a.comp:2:0
  |
2 | layout(set = 0, binding = 1, std430) buffer SSBO {
  |)"));
}

// Test if no DebugGlobalVariable is found, things still work
TEST_F(ValidateShaderDebugInfo, NoDebugGlobalVariable) {
  const std::string str = R"(
               OpCapability Shader
               OpExtension "SPV_KHR_non_semantic_info"
          %1 = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %x
               OpExecutionMode %main LocalSize 1 1 1
          %2 = OpString "a.comp"
          %8 = OpString "uint"
         %19 = OpString "BAD"
         %25 = OpString "float"
               OpDecorate %SSBO Block
               OpMemberDecorate %SSBO 0 Offset 0
               OpDecorate %x Binding 1
               OpDecorate %x DescriptorSet 0
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
    %uint_32 = OpConstant %uint 32
     %uint_6 = OpConstant %uint 6
     %uint_0 = OpConstant %uint 0
          %9 = OpExtInst %void %1 DebugTypeBasic %8 %uint_32 %uint_6 %uint_0
     %uint_3 = OpConstant %uint 3
          %6 = OpExtInst %void %1 DebugTypeFunction %uint_3 %void
         %18 = OpExtInst %void %1 DebugSource %2 %19
     %uint_1 = OpConstant %uint 1
     %uint_4 = OpConstant %uint 4
     %uint_2 = OpConstant %uint 2
         %20 = OpExtInst %void %1 DebugCompilationUnit %uint_1 %uint_4 %18 %uint_2
      %float = OpTypeFloat 32
         %26 = OpExtInst %void %1 DebugTypeBasic %25 %uint_32 %uint_3 %uint_0
    %v4float = OpTypeVector %float 4
         %28 = OpExtInst %void %1 DebugTypeVector %26 %uint_4
       %SSBO = OpTypeStruct %v4float
    %uint_10 = OpConstant %uint 10
%_ptr_StorageBuffer_SSBO = OpTypePointer Uniform %SSBO
    %uint_12 = OpConstant %uint 12
          %x = OpVariable %_ptr_StorageBuffer_SSBO StorageBuffer
     %uint_8 = OpConstant %uint 8
    %float_0 = OpConstant %float 0
         %50 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_StorageBuffer_v4float = OpTypePointer StorageBuffer %v4float
       %main = OpFunction %void None %5
         %15 = OpLabel
         %53 = OpAccessChain %_ptr_StorageBuffer_v4float %x %uint_0
               OpStore %53 %50
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(str.c_str(), SPV_ENV_VULKAN_1_3);
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_3));
  EXPECT_THAT(getDiagnosticString(), Not(HasSubstr("a.comp")));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
