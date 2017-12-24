// Copyright (c) 2017 Pierre Moreau
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

#include "source/opt/build_module.h"
#include "source/opt/decoration_manager.h"
#include "source/opt/ir_context.h"
#include "source/spirv_constant.h"
#include "unit_spirv.h"

namespace {

using spvtools::ir::IRContext;
using spvtools::ir::Instruction;
using spvtools::opt::analysis::DecorationManager;

class DecorationManagerTest : public ::testing::Test {
 public:
  DecorationManagerTest()
      : tools_(SPV_ENV_UNIVERSAL_1_2),
        context_(),
        consumer_([this](spv_message_level_t level, const char*,
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
        }),
        disassemble_options_(spvtools::SpirvTools::kDefaultDisassembleOption),
        error_message_() {
    tools_.SetMessageConsumer(consumer_);
  }

  virtual void TearDown() override { error_message_.clear(); }

  DecorationManager* GetDecorationManager(const std::string& text) {
    context_ = spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_2, consumer_, text);
    if (context_.get())
      return context_->get_decoration_mgr();
    else
      return nullptr;
  }

  // Disassembles |binary| and outputs the result in |text|. If |text| is a
  // null pointer, SPV_ERROR_INVALID_POINTER is returned.
  spv_result_t Disassemble(const std::vector<uint32_t>& binary,
                           std::string* text) {
    if (!text) return SPV_ERROR_INVALID_POINTER;
    return tools_.Disassemble(binary, text, disassemble_options_)
               ? SPV_SUCCESS
               : SPV_ERROR_INVALID_BINARY;
  }

  // Returns the accumulated error messages for the test.
  std::string GetErrorMessage() const { return error_message_; }

  std::string ToText(const std::vector<Instruction*>& inst) {
    std::vector<uint32_t> binary = {SpvMagicNumber, 0x10200, 0u, 2u, 0u};
    for (const Instruction* i : inst)
      i->ToBinaryWithoutAttachedDebugInsts(&binary);
    std::string text;
    Disassemble(binary, &text);
    return text;
  }

  std::string ModuleToText() {
    std::vector<uint32_t> binary;
    context_->module()->ToBinary(&binary, false);
    std::string text;
    Disassemble(binary, &text);
    return text;
  }

  spvtools::MessageConsumer GetConsumer() { return consumer_; }

 private:
  spvtools::SpirvTools
      tools_;  // An instance for calling SPIRV-Tools functionalities.
  std::unique_ptr<IRContext> context_;
  spvtools::MessageConsumer consumer_;
  uint32_t disassemble_options_;
  std::string error_message_;
};

TEST_F(DecorationManagerTest, ComparingDecorationsWithDiffOpcodes) {
  spvtools::ir::IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpDecorate %1 Constant
  Instruction inst1(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpDecorateId %1 %2
  Instruction inst2(&ir_context, SpvOpDecorateId, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}}, {SPV_OPERAND_TYPE_ID, {2u}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest, ComparingDecorationsWithDiffDeco) {
  spvtools::ir::IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpDecorate %1 Constant
  Instruction inst1(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpDecorate %1 Restrict
  Instruction inst2(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationRestrict}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest, ComparingSameDecorationsOnDiffTargetAllowed) {
  spvtools::ir::IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpDecorate %1 Constant
  Instruction inst1(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpDecorate %2 Constant
  Instruction inst2(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {2u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest, ComparingSameDecorationsOnDiffTargetDisallowed) {
  spvtools::ir::IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpDecorate %1 Constant
  Instruction inst1(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpDecorate %2 Constant
  Instruction inst2(&ir_context, SpvOpDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {2u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, false));
}

TEST_F(DecorationManagerTest, ComparingMemberDecorationsOnSameTypeDiffMember) {
  spvtools::ir::IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpMemberDecorate %1 0 Constant
  Instruction inst1(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpMemberDecorate %1 1 Constant
  Instruction inst2(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {1u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest,
       ComparingSameMemberDecorationsOnDiffTargetAllowed) {
  spvtools::ir::IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpMemberDecorate %1 0 Constant
  Instruction inst1(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpMemberDecorate %2 0 Constant
  Instruction inst2(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {2u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->AreDecorationsTheSame(&inst1, &inst2, true));
}

TEST_F(DecorationManagerTest,
       ComparingSameMemberDecorationsOnDiffTargetDisallowed) {
  spvtools::ir::IRContext ir_context(SPV_ENV_UNIVERSAL_1_2, GetConsumer());
  // OpMemberDecorate %1 0 Constant
  Instruction inst1(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {1u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  // OpMemberDecorate %2 0 Constant
  Instruction inst2(&ir_context, SpvOpMemberDecorate, 0u, 0u,
                    {{SPV_OPERAND_TYPE_ID, {2u}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0u}},
                     {SPV_OPERAND_TYPE_DECORATION, {SpvDecorationConstant}}});
  DecorationManager* decoManager = ir_context.get_decoration_mgr();
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->AreDecorationsTheSame(&inst1, &inst2, false));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsWithoutGroupsTrue) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %2 Restrict
OpDecorate %1 Constant
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsWithoutGroupsFalse) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %2 Restrict
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsWithGroupsTrue) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %1 Constant
OpDecorate %3 Restrict
%3 = OpDecorationGroup
OpGroupDecorate %3 %2
OpDecorate %4 Invariant
%4 = OpDecorationGroup
OpGroupDecorate %4 %1 %2
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsWithGroupsFalse) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %1 Constant
OpDecorate %4 Invariant
%4 = OpDecorationGroup
OpGroupDecorate %4 %1 %2
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDuplicateDecorations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %2 Constant
OpDecorate %2 Constant
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDifferentVariations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Location 0
OpDecorate %2 Location 1
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest,
       HaveTheSameDecorationsDuplicateMemberDecorations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpMemberDecorate %1 0 Location 0
OpMemberDecorate %2 0 Location 0
OpMemberDecorate %2 0 Location 0
%u32    = OpTypeInt 32 0
%1      = OpTypeStruct %u32 %u32
%2      = OpTypeStruct %u32 %u32
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest,
       HaveTheSameDecorationsDifferentMemberSameDecoration) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpMemberDecorate %1 0 Location 0
OpMemberDecorate %2 1 Location 0
%u32    = OpTypeInt 32 0
%1      = OpTypeStruct %u32 %u32
%2      = OpTypeStruct %u32 %u32
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDifferentMemberVariations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpMemberDecorate %1 0 Location 0
OpMemberDecorate %2 0 Location 1
%u32    = OpTypeInt 32 0
%1      = OpTypeStruct %u32 %u32
%2      = OpTypeStruct %u32 %u32
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDuplicateIdDecorations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorateId %1 AlignmentId %2
OpDecorateId %3 AlignmentId %2
OpDecorateId %3 AlignmentId %2
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%3      = OpVariable %u32 Uniform
%2      = OpSpecConstant %u32 0
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_TRUE(decoManager->HaveTheSameDecorations(1u, 3u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsDifferentIdVariations) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorateId %1 AlignmentId %2
OpDecorateId %3 AlignmentId %4
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%3      = OpVariable %u32 Uniform
%2      = OpSpecConstant %u32 0
%4      = OpSpecConstant %u32 0
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsLeftSymmetry) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %1 Constant
OpDecorate %2 Constant
OpDecorate %2 Restrict
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

TEST_F(DecorationManagerTest, HaveTheSameDecorationsRightSymmetry) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %1 Constant
OpDecorate %1 Restrict
OpDecorate %2 Constant
OpDecorate %2 Constant
%u32    = OpTypeInt 32 0
%1      = OpVariable %u32 Uniform
%2      = OpVariable %u32 Uniform
)";
  DecorationManager* decoManager = GetDecorationManager(spirv);
  EXPECT_THAT(GetErrorMessage(), "");
  EXPECT_FALSE(decoManager->HaveTheSameDecorations(1u, 2u));
}

}  // namespace
