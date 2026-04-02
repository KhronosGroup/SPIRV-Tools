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

// Tests for OpExtension validator rules.

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/spirv_target_env.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Values;
using ::testing::ValuesIn;

using ValidateSpvKHRAbort = spvtest::ValidateBase<bool>;

TEST_F(ValidateSpvKHRAbort, Valid) {
  const std::string str = R"(
OpCapability Shader
OpCapability AbortKHR
OpExtension "SPV_KHR_abort"
OpMemoryModel Logical Simple
OpEntryPoint GLCompute %main "main"

%void    = OpTypeVoid
%void_fn = OpTypeFunction %void
%uint32_t = OpTypeInt 32 0
%payload = OpConstant %uint32_t 6
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpAbortKHR %uint32_t %payload
OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSpvKHRAbort, RequireFinalInstructionInBlock) {
  const std::string str = R"(
            OpCapability Shader
            OpCapability AbortKHR
            OpExtension "SPV_KHR_abort"
            OpMemoryModel Logical Simple
            OpEntryPoint GLCompute %main "main"

%void     = OpTypeVoid
%void_fn  = OpTypeFunction %void
%uint32_t = OpTypeInt 32 0
%payload  = OpConstant %uint32_t 6
%main     = OpFunction %void None %void_fn
%entry    = OpLabel
            OpAbortKHR %uint32_t %payload
            OpReturn
            OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Return must appear in a block"));
}

TEST_F(ValidateSpvKHRAbort, RequiresCapability) {
  const std::string str = R"(
            OpCapability Shader
            OpExtension "SPV_KHR_abort"
            OpMemoryModel Logical Simple
            OpEntryPoint GLCompute %main "main"

%void     = OpTypeVoid
%void_fn  = OpTypeFunction %void
%uint32_t = OpTypeInt 32 0
%payload  = OpConstant %uint32_t 6
%main     = OpFunction %void None %void_fn
%entry    = OpLabel
            OpAbortKHR %uint32_t %payload
            OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Opcode AbortKHR requires one of these capabilities: AbortKHR"));
}

TEST_F(ValidateSpvKHRAbort, RequiresExtension) {
  const std::string str = R"(
            OpCapability Shader
            OpCapability AbortKHR
            OpMemoryModel Logical Simple
            OpEntryPoint GLCompute %main "main"

%void     = OpTypeVoid
%void_fn  = OpTypeFunction %void
%uint32_t = OpTypeInt 32 0
%payload  = OpConstant %uint32_t 6
%main     = OpFunction %void None %void_fn
%entry    = OpLabel
            OpAbortKHR %uint32_t %payload
            OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("1st operand of Capability: operand AbortKHR(5120) "
                        "requires one of these extensions: SPV_KHR_abort"));
}

TEST_F(ValidateSpvKHRAbort, MismatchedOperandTypes) {
  const std::string str = R"(
            OpCapability Shader
            OpCapability AbortKHR
            OpExtension "SPV_KHR_abort"
            OpMemoryModel Logical Simple
            OpEntryPoint GLCompute %main "main"

%void     = OpTypeVoid
%void_fn  = OpTypeFunction %void
%uint32_t = OpTypeInt 32 0
%f32_t    = OpTypeFloat 32
%payload  = OpConstant %uint32_t 6
%main     = OpFunction %void None %void_fn
%entry    = OpLabel
            OpAbortKHR %f32_t %payload
            OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Type of Message operand does not logically match "
                        "the type of the Message Type operand"));
}

TEST_F(ValidateSpvKHRAbort, ValidCompositeOperandTypes) {
  const std::string str = R"(
             OpCapability Shader
             OpCapability Int8
             OpCapability AbortKHR
             OpCapability ConstantDataKHR
             OpExtension "SPV_KHR_abort"
             OpExtension "SPV_KHR_constant_data"
             OpMemoryModel Logical Simple
             OpEntryPoint GLCompute %main "main"

             OpDecorate %string1_t UTFEncodedKHR
             OpDecorate %string2_t UTFEncodedKHR

             OpDecorate %string1_x UTFEncodedKHR
             OpDecorate %string1_x ArrayStride 1
             OpDecorate %string2_x UTFEncodedKHR
             OpDecorate %string2_x ArrayStride 1
             OpMemberDecorate %message_x 0 Offset 0
             OpMemberDecorate %message_x 1 Offset 6
             OpMemberDecorate %message_x 2 Offset 8

%void      = OpTypeVoid
%void_fn   = OpTypeFunction %void

   %char_t = OpTypeInt 8 0
 %uint32_t = OpTypeInt 32 0
  %str1len = OpConstant %uint32_t 6
%string1_t = OpTypeArray %char_t %str1len
  %string1 = OpConstantDataKHR %string1_t "test: "
  %str2len = OpSpecConstant %uint32_t 2
%string2_t = OpTypeArray %char_t %str2len
  %string2 = OpSpecConstantDataKHR %string2_t "%u"
%message_t = OpTypeStruct %string1_t %string2_t %uint32_t
%uintval   = OpConstant %uint32_t 6

%string1_x = OpTypeArray %char_t %str1len
%string2_x = OpTypeArray %char_t %str2len
%message_x = OpTypeStruct %string1_t %string2_t %uint32_t

%main      = OpFunction %void None %void_fn
%entry     = OpLabel
  %message = OpCompositeConstruct %message_t %string1 %string2 %uintval
             OpAbortKHR %message_x %message
             OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSpvKHRAbort, MismatchedCompositeOperandTypes) {
  const std::string str = R"(
             OpCapability Shader
             OpCapability Int8
             OpCapability AbortKHR
             OpCapability ConstantDataKHR
             OpExtension "SPV_KHR_abort"
             OpExtension "SPV_KHR_constant_data"
             OpMemoryModel Logical Simple
             OpEntryPoint GLCompute %main "main"

             OpDecorate %string1_t UTFEncodedKHR
             OpDecorate %string2_t UTFEncodedKHR

             OpDecorate %string1_x UTFEncodedKHR
             OpDecorate %string1_x ArrayStride 1
             OpDecorate %string2_x UTFEncodedKHR
             OpDecorate %string2_x ArrayStride 1
             OpMemberDecorate %message_x 0 Offset 0
             OpMemberDecorate %message_x 1 Offset 6
             OpMemberDecorate %message_x 2 Offset 8

%void      = OpTypeVoid
%void_fn   = OpTypeFunction %void

   %char_t = OpTypeInt 8 0
 %uint32_t = OpTypeInt 32 0
  %str1len = OpConstant %uint32_t 6
%string1_t = OpTypeArray %char_t %str1len
  %string1 = OpConstantDataKHR %string1_t "test: "
  %str2len = OpSpecConstant %uint32_t 2
%string2_t = OpTypeArray %char_t %str2len
  %string2 = OpSpecConstantDataKHR %string2_t "%u"
%message_t = OpTypeStruct %string1_t %string2_t
%uintval   = OpConstant %uint32_t 6

%string1_x = OpTypeArray %char_t %str1len
%string2_x = OpTypeArray %char_t %str2len
%message_x = OpTypeStruct %string1_t %string2_t %uint32_t

%main      = OpFunction %void None %void_fn
%entry     = OpLabel
  %message = OpCompositeConstruct %message_t %string1 %string2
             OpAbortKHR %message_x %message
             OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Type of Message operand does not logically match "
                        "the type of the Message Type operand"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
