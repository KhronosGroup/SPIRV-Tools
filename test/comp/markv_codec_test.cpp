// Copyright (c) 2017 Google Inc.
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

// Tests for unique type declaration rules validator.

#include <string>

#include "gmock/gmock.h"
#include "spirv-tools/markv.h"
#include "test_fixture.h"
#include "unit_spirv.h"

namespace {

using spvtest::ScopedContext;

void DiagnosticsMessageHandler(spv_message_level_t level, const char*,
                               const spv_position_t& position,
                               const char* message) {
  switch (level) {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
      std::cerr << "error: " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_WARNING:
      std::cout << "warning: " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_INFO:
      std::cout << "info: " << position.index << ": " << message << std::endl;
      break;
    default:
      break;
  }
}

// Compiles |code| to SPIR-V |words|.
void Compile(const std::string& code, std::vector<uint32_t>* words,
             spv_target_env env = SPV_ENV_UNIVERSAL_1_1) {
  ScopedContext ctx(env);
  SetContextMessageConsumer(ctx.context, DiagnosticsMessageHandler);

  spv_binary spirv_binary;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(
      ctx.context, code.c_str(), code.size(), &spirv_binary, nullptr));

  *words = std::vector<uint32_t>(
      spirv_binary->code, spirv_binary->code + spirv_binary->wordCount);

  spvBinaryDestroy(spirv_binary);
}

// Disassembles SPIR-V |words| to |out_text|.
void Disassemble(const std::vector<uint32_t>& words,
                 std::string* out_text,
                 spv_target_env env = SPV_ENV_UNIVERSAL_1_1) {
  ScopedContext ctx(env);
  SetContextMessageConsumer(ctx.context, DiagnosticsMessageHandler);

  spv_text text = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvBinaryToText(ctx.context, words.data(),
                                         words.size(), 0, &text, nullptr));
  assert(text);

  *out_text = std::string(text->str, text->length);
  spvTextDestroy(text);
}

// Encodes SPIR-V |words| to |markv_binary|. |comments| context snippets of
// disassembly and bit sequences for debugging.
void Encode(const std::vector<uint32_t>& words,
            spv_markv_binary* markv_binary,
            std::string* comments,
            spv_target_env env = SPV_ENV_UNIVERSAL_1_1) {
  ScopedContext ctx(env);
  SetContextMessageConsumer(ctx.context, DiagnosticsMessageHandler);

  spv_markv_encoder_options_t options;
  spv_text spv_text_comments;
  ASSERT_EQ(SPV_SUCCESS, spvSpirvToMarkv(ctx.context, words.data(),
                                         words.size(), &options, markv_binary,
                                         &spv_text_comments, nullptr));

  *comments = std::string(spv_text_comments->str, spv_text_comments->length);
  spvTextDestroy(spv_text_comments);
}

// Decodes |markv_binary| to SPIR-V |words|.
void Decode(const spv_markv_binary markv_binary,
            std::vector<uint32_t>* words,
            spv_target_env env = SPV_ENV_UNIVERSAL_1_1) {
  ScopedContext ctx(env);
  SetContextMessageConsumer(ctx.context, DiagnosticsMessageHandler);

  spv_binary spirv_binary = nullptr;
  spv_markv_decoder_options_t options;
  ASSERT_EQ(SPV_SUCCESS, spvMarkvToSpirv(ctx.context, markv_binary->data,
                                         markv_binary->length, &options,
                                         &spirv_binary, nullptr, nullptr));

  *words = std::vector<uint32_t>(
      spirv_binary->code, spirv_binary->code + spirv_binary->wordCount);

  spvBinaryDestroy(spirv_binary);
}

// Encodes/decodes |original|, assembles/dissasembles |original|, then compares
// the results of the two operations.
void TestEncodeDecode(const std::string& original_text) {
  std::vector<uint32_t> expected_binary;
  Compile(original_text, &expected_binary);
  ASSERT_FALSE(expected_binary.empty());

  std::string expected_text;
  Disassemble(expected_binary, &expected_text);
  ASSERT_FALSE(expected_text.empty());

  spv_markv_binary markv_binary = nullptr;
  std::string encoder_comments;
  Encode(expected_binary, &markv_binary, &encoder_comments);
  ASSERT_NE(nullptr, markv_binary);

  // std::cerr << encoder_comments << std::endl;
  std::cerr << "SPIR-V size: " << expected_binary.size() * 4 << std::endl;
  std::cerr << "MARK-V size: " << markv_binary->length << std::endl;

  std::vector<uint32_t> decoded_binary;
  Decode(markv_binary, &decoded_binary);
  ASSERT_FALSE(decoded_binary.empty());

  std::string decoded_text;
  Disassemble(decoded_binary, &decoded_text);
  ASSERT_FALSE(decoded_text.empty());

  EXPECT_EQ(expected_binary, decoded_binary) << encoder_comments;
  EXPECT_EQ(expected_text, decoded_text) << encoder_comments;

  spvMarkvBinaryDestroy(markv_binary);
}

TEST(Markv, U32Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%u32 = OpTypeInt 32 0
%100 = OpConstant %u32 0
%200 = OpConstant %u32 1
%300 = OpConstant %u32 4294967295
)");
}

TEST(Markv, S32Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%s32 = OpTypeInt 32 1
%100 = OpConstant %s32 0
%200 = OpConstant %s32 1
%300 = OpConstant %s32 -1
%400 = OpConstant %s32 2147483647
%500 = OpConstant %s32 -2147483648
)");
}

TEST(Markv, U64Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int64
OpMemoryModel Logical GLSL450
%u64 = OpTypeInt 64 0
%100 = OpConstant %u64 0
%200 = OpConstant %u64 1
%300 = OpConstant %u64 18446744073709551615
)");
}

TEST(Markv, S64Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int64
OpMemoryModel Logical GLSL450
%s64 = OpTypeInt 64 1
%100 = OpConstant %s64 0
%200 = OpConstant %s64 1
%300 = OpConstant %s64 -1
%400 = OpConstant %s64 9223372036854775807
%500 = OpConstant %s64 -9223372036854775808
)");
}

TEST(Markv, U16Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int16
OpMemoryModel Logical GLSL450
%u16 = OpTypeInt 16 0
%100 = OpConstant %u16 0
%200 = OpConstant %u16 1
%300 = OpConstant %u16 65535
)");
}

TEST(Markv, S16Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Int16
OpMemoryModel Logical GLSL450
%s16 = OpTypeInt 16 1
%100 = OpConstant %s16 0
%200 = OpConstant %s16 1
%300 = OpConstant %s16 -1
%400 = OpConstant %s16 32767
%500 = OpConstant %s16 -32768
)");
}

TEST(Markv, F32Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%f32 = OpTypeFloat 32
%100 = OpConstant %f32 0
%200 = OpConstant %f32 1
%300 = OpConstant %f32 0.1
%400 = OpConstant %f32 -0.1
)");
}

TEST(Markv, F64Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Float64
OpMemoryModel Logical GLSL450
%f64 = OpTypeFloat 64
%100 = OpConstant %f64 0
%200 = OpConstant %f64 1
%300 = OpConstant %f64 0.1
%400 = OpConstant %f64 -0.1
)");
}

TEST(Markv, F16Literal) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpCapability Float16
OpMemoryModel Logical GLSL450
%f16 = OpTypeFloat 16
%100 = OpConstant %f16 0
%200 = OpConstant %f16 1
%300 = OpConstant %f16 0.1
%400 = OpConstant %f16 -0.1
)");
}

TEST(Markv, StringLiteral) {
  TestEncodeDecode(R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_16bit_storage"
OpExtension "xxx"
OpExtension "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OpExtension ""
OpMemoryModel Logical GLSL450
)");
}

TEST(Markv, WithFunction) {
  TestEncodeDecode(R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Physical32 OpenCL
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%100 = OpConstant %u32 1
%200 = OpConstant %u32 2
%main = OpFunction %void None %void_func
%entry_main = OpLabel
%300 = OpIAdd %u32 %100 %200
OpReturn
OpFunctionEnd
)");
}

}  // namespace
