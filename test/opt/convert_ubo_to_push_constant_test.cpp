// Copyright (c) 2026 Diligent Graphics LLC
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

#include <string>
#include <vector>

#include "source/opt/convert_ubo_to_push_constant_pass.h"
#include "spirv-tools/optimizer.hpp"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

// Numeric IDs make it possible to check that the exact types and variables are
// preserved, including distinct struct types with identical layouts.
const char kShader[] = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %100 "main"
OpExecutionMode %100 LocalSize 1 1 1
OpName %12 "Block"
OpName %20 "buffer"
OpMemberName %12 0 "color"
OpDecorate %12 Block
OpMemberDecorate %12 0 Offset 0
OpMemberDecorate %12 1 Offset 16
OpMemberDecorate %12 2 Offset 48
OpMemberDecorate %12 2 MatrixStride 16
OpMemberDecorate %12 2 ColMajor
OpDecorate %10 ArrayStride 16
OpDecorate %20 DescriptorSet 2
OpDecorate %20 Binding 3
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeInt 32 0
%5 = OpConstant %4 0
%6 = OpConstant %4 1
%7 = OpTypeVector %3 4
%8 = OpConstant %4 2
%10 = OpTypeArray %7 %8
%11 = OpTypeMatrix %7 4
%12 = OpTypeStruct %7 %10 %11
%13 = OpTypePointer Uniform %12
%14 = OpTypePointer Uniform %7
%15 = OpTypePointer Uniform %3
%20 = OpVariable %13 Uniform
%100 = OpFunction %1 None %2
%101 = OpLabel
%30 = OpAccessChain %14 %20 %5
%31 = OpLoad %7 %30
%32 = OpAccessChain %14 %20 %6 %6
%33 = OpInBoundsAccessChain %15 %30 %5
%34 = OpCopyObject %15 %33
%35 = OpLoad %3 %34
OpReturn
OpFunctionEnd
)";

void Replace(std::string* text, const std::string& from,
             const std::string& to) {
  const auto position = text->find(from);
  ASSERT_NE(position, std::string::npos) << from;
  text->replace(position, from.size(), to);
}

std::string SmallTypeShader(uint32_t width) {
  const std::string bits = std::to_string(width);
  return "OpCapability Shader\nOpCapability UniformAndStorageBuffer" + bits +
         "BitAccess\nOpExtension \"SPV_KHR_" + bits + "bit_storage\"\n" + R"(
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %100 "main"
OpExecutionMode %100 LocalSize 1 1 1
OpName %20 "buffer"
OpDecorate %12 Block
OpMemberDecorate %12 0 Offset 0
OpDecorate %20 DescriptorSet 2
OpDecorate %20 Binding 3
%1 = OpTypeVoid
%2 = OpTypeFunction %1
)" + "%3 = OpTypeInt " +
         bits + R"( 0
%12 = OpTypeStruct %3
%13 = OpTypePointer Uniform %12
%20 = OpVariable %13 Uniform
%100 = OpFunction %1 None %2
%101 = OpLabel
OpReturn
OpFunctionEnd
)";
}

// Use a storage-buffer destination so the copy remains live through DCE and
// can also carry Vulkan memory-model operands independently of the source.
std::string CopyMemoryShader(const char* memory_operands, bool derived_source) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 Binding 3",
          "OpDecorate %20 Binding 3\nOpDecorate %21 DescriptorSet 0\n"
          "OpDecorate %21 Binding 0");
  Replace(&text, "%20 = OpVariable",
          "%16 = OpTypePointer StorageBuffer %12\n"
          "%17 = OpTypePointer StorageBuffer %7\n"
          "%21 = OpVariable %16 StorageBuffer\n%20 = OpVariable");
  const std::string copy =
      derived_source ? "%37 = OpAccessChain %17 %21 %5\nOpCopyMemory %37 %30"
                     : "OpCopyMemory %21 %20";
  Replace(&text, "%31 = OpLoad",
          copy + " " + memory_operands + "\n%31 = OpLoad");
  return text;
}

std::vector<std::string> CopyMemoryInstructions(IRContext* context) {
  std::vector<std::string> instructions;
  context->module()->ForEachInst([&](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpCopyMemory) {
      instructions.push_back(inst->PrettyPrint());
    }
  });
  return instructions;
}

class ConvertUBOToPushConstantTest : public PassTest<::testing::Test> {
 protected:
  void SetUp() override {
    SetTargetEnv(SPV_ENV_VULKAN_1_1);
    SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    SetMessageConsumer([this](spv_message_level_t, const char*,
                              const spv_position_t&, const char* message) {
      diagnostics_ += message;
      diagnostics_ += '\n';
    });
  }

  template <typename... Args>
  void Convert(const std::string& text, Args&&... args) {
    SpirvTools tools(env_);
    tools.SetMessageConsumer(consumer());
    std::vector<uint32_t> input;
    ASSERT_TRUE(tools.Assemble(text, &input, assemble_options_));
    ASSERT_TRUE(tools.Validate(input.data(), input.size(), ValidatorOptions()))
        << diagnostics_;
    auto result = SinglePassRunToBinary<ConvertUBOToPushConstantPass>(
        text, false, std::forward<Args>(args)...);
    ASSERT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result))
        << diagnostics_;
    ASSERT_TRUE(tools.Validate(std::get<0>(result))) << diagnostics_;
  }

  template <typename... Args>
  void FailConversion(const std::string& text, const std::string& diagnostic,
                      Args&&... args) {
    SpirvTools tools(env_);
    tools.SetMessageConsumer(consumer());
    std::vector<uint32_t> input, after;
    ASSERT_TRUE(tools.Assemble(text, &input, assemble_options_));
    ASSERT_TRUE(tools.Validate(input.data(), input.size(), ValidatorOptions()))
        << diagnostics_;
    auto result = SinglePassRunToBinary<ConvertUBOToPushConstantPass>(
        text, false, std::forward<Args>(args)...);
    EXPECT_EQ(Pass::Status::Failure, std::get<1>(result));
    EXPECT_NE(diagnostics_.find(diagnostic), std::string::npos) << diagnostics_;
    // Unsupported inputs must be diagnosed before changing the module.
    context()->module()->ToBinary(&after, false);
    EXPECT_EQ(input, after);
  }

  void CheckPointer(uint32_t id, spv::StorageClass storage, uint32_t pointee) {
    const auto* value = context()->get_def_use_mgr()->GetDef(id);
    ASSERT_NE(value, nullptr);
    const auto* type = context()->get_def_use_mgr()->GetDef(value->type_id());
    ASSERT_NE(type, nullptr);
    EXPECT_EQ(type->opcode(), spv::Op::OpTypePointer);
    EXPECT_EQ(type->GetSingleWordInOperand(0), static_cast<uint32_t>(storage));
    EXPECT_EQ(type->GetSingleWordInOperand(1), pointee);
  }

  spv_target_env env_ = SPV_ENV_VULKAN_1_1;
  static constexpr uint32_t assemble_options_ =
      SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS;
  std::string diagnostics_;
};

TEST_F(ConvertUBOToPushConstantTest, VariableNameAndNestedPointers) {
  ASSERT_NO_FATAL_FAILURE(Convert(kShader, "buffer"));
  CheckPointer(20, spv::StorageClass::PushConstant, 12);
  CheckPointer(30, spv::StorageClass::PushConstant, 7);
  CheckPointer(32, spv::StorageClass::PushConstant, 7);
  CheckPointer(33, spv::StorageClass::PushConstant, 3);
  CheckPointer(34, spv::StorageClass::PushConstant, 3);
  auto* decorations = context()->get_decoration_mgr();
  EXPECT_FALSE(decorations->HasDecoration(20, spv::Decoration::Binding));
  EXPECT_FALSE(decorations->HasDecoration(20, spv::Decoration::DescriptorSet));
  // Compare every remaining layout and naming instruction verbatim.
  auto before = BuildModule(env_, consumer(), kShader, assemble_options_);
  std::vector<std::string> expected, actual;
  for (auto& inst : before->annotations()) {
    if (inst.GetSingleWordInOperand(0) != 20)
      expected.push_back(inst.PrettyPrint());
  }
  for (auto& inst : context()->annotations())
    actual.push_back(inst.PrettyPrint());
  EXPECT_EQ(expected, actual);
  expected.clear();
  actual.clear();
  for (auto& inst : before->module()->debugs2())
    expected.push_back(inst.PrettyPrint());
  for (auto& inst : context()->module()->debugs2())
    actual.push_back(inst.PrettyPrint());
  EXPECT_EQ(expected, actual);
}

TEST_F(ConvertUBOToPushConstantTest, BlockName) {
  ASSERT_NO_FATAL_FAILURE(Convert(kShader, "Block"));
  CheckPointer(20, spv::StorageClass::PushConstant, 12);
}

TEST_F(ConvertUBOToPushConstantTest, VariableAndBlockHaveSameName) {
  std::string text = kShader;
  Replace(&text, "\"Block\"", "\"buffer\"");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  CheckPointer(20, spv::StorageClass::PushConstant, 12);
}

TEST_F(ConvertUBOToPushConstantTest, IgnoresNonBufferNameCollision) {
  std::string text = kShader;
  Replace(&text, "OpName %12", "OpName %3 \"buffer\"\nOpName %12");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
}

TEST_F(ConvertUBOToPushConstantTest, BindingWithoutDebugNames) {
  std::string text = kShader;
  Replace(&text, "OpName %12 \"Block\"\n", "");
  Replace(&text, "OpName %20 \"buffer\"\n", "");
  ASSERT_NO_FATAL_FAILURE(Convert(text, 2u, 3u));
  CheckPointer(20, spv::StorageClass::PushConstant, 12);
}

TEST_F(ConvertUBOToPushConstantTest, SharedTypesAndGroupedDecorations) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 DescriptorSet 2\nOpDecorate %20 Binding 3",
          "%40 = OpDecorationGroup\nOpDecorate %40 DescriptorSet 2\n"
          "OpDecorate %40 Binding 3\nOpDecorate %40 RelaxedPrecision\n"
          "OpGroupDecorate %40 %20 %21");
  Replace(&text, "%20 = OpVariable %13 Uniform",
          "%20 = OpVariable %13 Uniform\n%21 = OpVariable %13 Uniform");
  Replace(&text, "%31 = OpLoad",
          "%36 = OpAccessChain %14 %21 %5\n%31 = OpLoad");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  CheckPointer(20, spv::StorageClass::PushConstant, 12);
  CheckPointer(21, spv::StorageClass::Uniform, 12);
  CheckPointer(36, spv::StorageClass::Uniform, 7);
  auto* decorations = context()->get_decoration_mgr();
  EXPECT_FALSE(decorations->HasDecoration(20, spv::Decoration::Binding));
  EXPECT_FALSE(decorations->HasDecoration(20, spv::Decoration::DescriptorSet));
  EXPECT_TRUE(
      decorations->HasDecoration(20, spv::Decoration::RelaxedPrecision));
  EXPECT_TRUE(decorations->HasDecoration(21, spv::Decoration::Binding));
  EXPECT_TRUE(decorations->HasDecoration(21, spv::Decoration::DescriptorSet));
  EXPECT_TRUE(
      decorations->HasDecoration(21, spv::Decoration::RelaxedPrecision));
}

TEST_F(ConvertUBOToPushConstantTest, BindingInDecorationGroup) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 DescriptorSet 2\nOpDecorate %20 Binding 3",
          "%40 = OpDecorationGroup\nOpDecorate %40 DescriptorSet 2\n"
          "OpDecorate %40 Binding 3\nOpGroupDecorate %40 %20");
  ASSERT_NO_FATAL_FAILURE(Convert(text, 2u, 3u));
}

TEST_F(ConvertUBOToPushConstantTest, RemovesNonWritableDecoration) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 Binding 3",
          "OpDecorate %20 Binding 3\nOpDecorate %20 NonWritable\n"
          "OpDecorate %21 DescriptorSet 0\nOpDecorate %21 Binding 0\n"
          "OpDecorate %21 NonWritable");
  Replace(&text, "%20 = OpVariable %13 Uniform",
          "%20 = OpVariable %13 Uniform\n%21 = OpVariable %13 Uniform");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  auto* decorations = context()->get_decoration_mgr();
  EXPECT_FALSE(decorations->HasDecoration(20, spv::Decoration::NonWritable));
  EXPECT_TRUE(decorations->HasDecoration(21, spv::Decoration::NonWritable));
  CheckPointer(21, spv::StorageClass::Uniform, 12);
}

TEST_F(ConvertUBOToPushConstantTest, RemovesGroupedNonWritableDecoration) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 DescriptorSet 2\nOpDecorate %20 Binding 3",
          "%40 = OpDecorationGroup\nOpDecorate %40 DescriptorSet 2\n"
          "OpDecorate %40 Binding 3\nOpDecorate %40 NonWritable\n"
          "OpDecorate %40 RelaxedPrecision\nOpGroupDecorate %40 %20 %21");
  Replace(&text, "%20 = OpVariable %13 Uniform",
          "%20 = OpVariable %13 Uniform\n%21 = OpVariable %13 Uniform");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  auto* decorations = context()->get_decoration_mgr();
  for (const auto decoration :
       {spv::Decoration::NonWritable, spv::Decoration::DescriptorSet,
        spv::Decoration::Binding}) {
    EXPECT_FALSE(decorations->HasDecoration(20, decoration));
    EXPECT_TRUE(decorations->HasDecoration(21, decoration));
  }
  EXPECT_TRUE(
      decorations->HasDecoration(20, spv::Decoration::RelaxedPrecision));
  EXPECT_TRUE(
      decorations->HasDecoration(21, spv::Decoration::RelaxedPrecision));
  CheckPointer(21, spv::StorageClass::Uniform, 12);
}

TEST_F(ConvertUBOToPushConstantTest, RejectsNonReadableDecoration) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 Binding 3",
          "OpDecorate %20 Binding 3\nOpDecorate %20 NonReadable");
  text.erase(text.find("%30 = OpAccessChain"),
             text.find("OpReturn") - text.find("%30 = OpAccessChain"));
  FailConversion(text, "NonReadable uniform buffers cannot be converted",
                 "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, RejectsGroupedNonReadableDecoration) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 Binding 3",
          "OpDecorate %20 Binding 3\n"
          "%40 = OpDecorationGroup\nOpDecorate %40 NonReadable\n"
          "OpGroupDecorate %40 %20 %21\n"
          "OpDecorate %21 DescriptorSet 0\nOpDecorate %21 Binding 0");
  Replace(&text, "%20 = OpVariable %13 Uniform",
          "%20 = OpVariable %13 Uniform\n%21 = OpVariable %13 Uniform");
  text.erase(text.find("%30 = OpAccessChain"),
             text.find("OpReturn") - text.find("%30 = OpAccessChain"));
  FailConversion(text, "NonReadable uniform buffers cannot be converted",
                 "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, DistinctButIdenticalStructTypes) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %12 Block",
          "OpDecorate %12 Block\nOpDecorate %22 Block\n"
          "OpMemberDecorate %22 0 Offset 0\nOpMemberDecorate %22 1 Offset 16\n"
          "OpMemberDecorate %22 2 Offset 48\nOpMemberDecorate %22 2 "
          "MatrixStride 16\n"
          "OpMemberDecorate %22 2 ColMajor\nOpDecorate %21 DescriptorSet "
          "0\nOpDecorate %21 Binding 0");
  Replace(
      &text, "%13 = OpTypePointer",
      "%22 = OpTypeStruct %7 %10 %11\n"
      "%23 = OpTypePointer Uniform %22\n%24 = OpTypePointer PushConstant %22\n"
      "%21 = OpVariable %23 Uniform\n%13 = OpTypePointer");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  CheckPointer(20, spv::StorageClass::PushConstant, 12);
  CheckPointer(21, spv::StorageClass::Uniform, 22);
}

TEST_F(ConvertUBOToPushConstantTest, ExistingPointerTypeBeforeVariable) {
  std::string text = kShader;
  Replace(&text, "%20 = OpVariable",
          "%16 = OpTypePointer PushConstant %12\n"
          "%17 = OpTypeFunction %1 %16\n%20 = OpVariable");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  EXPECT_EQ(context()->get_def_use_mgr()->GetDef(20)->type_id(), 16u);
  EXPECT_EQ(context()->get_def_use_mgr()->GetDef(17)->GetSingleWordInOperand(1),
            16u);
}

TEST_F(ConvertUBOToPushConstantTest, ExistingPointerTypeAfterVariable) {
  std::string text = kShader;
  Replace(&text, "%20 = OpVariable %13 Uniform",
          "%20 = OpVariable %13 Uniform\n"
          "%16 = OpTypePointer PushConstant %12");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  EXPECT_EQ(context()->get_def_use_mgr()->GetDef(20)->type_id(), 16u);
}

TEST_F(ConvertUBOToPushConstantTest, EntryPointInterfaceInSpirv14) {
  env_ = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
  SetTargetEnv(env_);
  std::string text = kShader;
  Replace(&text, "GLCompute %100 \"main\"", "GLCompute %100 \"main\" %20");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  EXPECT_EQ(
      context()->module()->entry_points().begin()->GetSingleWordInOperand(3),
      20u);
}

TEST_F(ConvertUBOToPushConstantTest, UniversalEnvironment) {
  env_ = SPV_ENV_UNIVERSAL_1_3;
  SetTargetEnv(env_);
  ASSERT_NO_FATAL_FAILURE(Convert(kShader, "buffer"));
  CheckPointer(20, spv::StorageClass::PushConstant, 12);
}

TEST_F(ConvertUBOToPushConstantTest, RejectsOpenGLEnvironment) {
  env_ = SPV_ENV_OPENGL_4_5;
  SetTargetEnv(env_);
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 DescriptorSet 2",
          "OpDecorate %20 DescriptorSet 0");
  FailConversion(text, "push constants are not supported in OpenGL", "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, RejectsInitializer) {
  // Vulkan disallows initializers on Uniform variables, but universal SPIR-V
  // permits them. PushConstant variables must never have initializers.
  env_ = SPV_ENV_UNIVERSAL_1_3;
  SetTargetEnv(env_);
  std::string text = kShader;
  Replace(&text, "%20 = OpVariable %13 Uniform",
          "%40 = OpConstantNull %12\n%20 = OpVariable %13 Uniform %40");
  FailConversion(text, "uniform buffers with initializers cannot be converted",
                 "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, CopyMemorySource) {
  std::string text = kShader;
  Replace(&text, "%20 = OpVariable",
          "%16 = OpTypePointer Function %12\n%20 = OpVariable");
  Replace(&text, "%101 = OpLabel",
          "%101 = OpLabel\n%36 = OpVariable %16 Function\n"
          "OpCopyMemory %36 %20");
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  CheckPointer(36, spv::StorageClass::Function, 12);
}

TEST_F(ConvertUBOToPushConstantTest, CopyMemoryMemoryOperands) {
  for (bool derived_source : {false, true}) {
    SCOPED_TRACE(derived_source);
    for (const char* operands :
         {"", "None", "Aligned 16", "Aligned 32", "Volatile", "Nontemporal",
          "Volatile|Aligned|Nontemporal 16"}) {
      SCOPED_TRACE(operands);
      const std::string text = CopyMemoryShader(operands, derived_source);
      auto before = BuildModule(env_, consumer(), text, assemble_options_);
      ASSERT_NE(before, nullptr);
      const auto expected = CopyMemoryInstructions(before.get());
      ASSERT_EQ(expected.size(), 1u);
      ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
      EXPECT_EQ(CopyMemoryInstructions(context()), expected);
      CheckPointer(20, spv::StorageClass::PushConstant, 12);
      CheckPointer(21, spv::StorageClass::StorageBuffer, 12);
      if (derived_source) {
        CheckPointer(30, spv::StorageClass::PushConstant, 7);
        CheckPointer(37, spv::StorageClass::StorageBuffer, 7);
      }
    }
  }
}

TEST_F(ConvertUBOToPushConstantTest, CopyMemorySeparateMemoryOperands) {
  env_ = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
  SetTargetEnv(env_);
  for (const char* operands :
       {"None None", "Aligned 16 None", "Aligned 32 None", "None Aligned 16",
        "Aligned 16 Aligned 16", "Volatile Nontemporal", "Nontemporal Volatile",
        "Volatile|Aligned 16 Aligned|Nontemporal 16"}) {
    SCOPED_TRACE(operands);
    std::string text = CopyMemoryShader(operands, true);
    Replace(&text, "GLCompute %100 \"main\"",
            "GLCompute %100 \"main\" %20 %21");
    auto before = BuildModule(env_, consumer(), text, assemble_options_);
    ASSERT_NE(before, nullptr);
    const auto expected = CopyMemoryInstructions(before.get());
    ASSERT_EQ(expected.size(), 1u);
    ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
    EXPECT_EQ(CopyMemoryInstructions(context()), expected);
    CheckPointer(30, spv::StorageClass::PushConstant, 7);
    CheckPointer(37, spv::StorageClass::StorageBuffer, 7);
  }
}

TEST_F(ConvertUBOToPushConstantTest, CopyMemoryRejectsNonPrivatePointer) {
  env_ = SPV_ENV_VULKAN_1_1_SPIRV_1_4;
  SetTargetEnv(env_);
  for (const char* operands :
       {"NonPrivatePointer", "Aligned|NonPrivatePointer 16",
        "None NonPrivatePointer", "Aligned 16 NonPrivatePointer",
        "None Aligned|MakePointerVisible|NonPrivatePointer 16 %6",
        // The current validator also rejects NonPrivatePointer on a separate
        // target mask when the source has PushConstant storage.
        "NonPrivatePointer None",
        "Aligned|MakePointerAvailable|NonPrivatePointer 16 %6 None"}) {
    SCOPED_TRACE(operands);
    std::string text = CopyMemoryShader(operands, true);
    Replace(&text, "GLCompute %100 \"main\"",
            "GLCompute %100 \"main\" %20 %21");
    Replace(&text, "OpMemoryModel Logical GLSL450",
            "OpCapability VulkanMemoryModel\n"
            "OpCapability VulkanMemoryModelDeviceScope\n"
            "OpExtension \"SPV_KHR_vulkan_memory_model\"\n"
            "OpMemoryModel Logical Vulkan");
    ASSERT_NO_FATAL_FAILURE(FailConversion(
        text,
        "NonPrivatePointer memory access is not supported for push constants",
        "buffer"));
  }
}

TEST_F(ConvertUBOToPushConstantTest,
       CopyMemoryAfterInliningAndDeadCodeElimination) {
  // Match Diligent's preprocessing: memory operands on a live copy must not
  // prevent conversion after inlining and dead-code elimination.
  const std::string text = CopyMemoryShader("Aligned 16", true);
  SpirvTools tools(env_);
  tools.SetMessageConsumer(consumer());
  std::vector<uint32_t> input, output;
  ASSERT_TRUE(tools.Assemble(text, &input, assemble_options_));
  ASSERT_TRUE(tools.Validate(input));
  auto before = BuildModule(env_, consumer(), input.data(), input.size());
  ASSERT_NE(before, nullptr);
  const auto expected = CopyMemoryInstructions(before.get());
  ASSERT_EQ(expected.size(), 1u);
  Optimizer optimizer(env_);
  optimizer.SetMessageConsumer(consumer());
  optimizer.RegisterPass(CreateInlineExhaustivePass());
  optimizer.RegisterPass(CreateAggressiveDCEPass());
  optimizer.RegisterPass(CreateEliminateDeadFunctionsPass());
  optimizer.RegisterPass(CreateConvertUBOToPushConstantPass("buffer"));
  ASSERT_TRUE(optimizer.Run(input.data(), input.size(), &output))
      << diagnostics_;
  ASSERT_TRUE(tools.Validate(output)) << diagnostics_;
  auto after = BuildModule(env_, consumer(), output.data(), output.size());
  ASSERT_NE(after, nullptr);
  EXPECT_EQ(CopyMemoryInstructions(after.get()), expected);
}

TEST_F(ConvertUBOToPushConstantTest, UnusedBuffer) {
  std::string text = kShader;
  text.erase(text.find("%30 = OpAccessChain"),
             text.find("OpReturn") - text.find("%30 = OpAccessChain"));
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
}

TEST_F(ConvertUBOToPushConstantTest, MissingName) {
  FailConversion(kShader, "no matching uniform buffer", "missing");
}

TEST_F(ConvertUBOToPushConstantTest, EmptyName) {
  FailConversion(kShader, "name must not be empty", "");
}

TEST_F(ConvertUBOToPushConstantTest, MissingBinding) {
  FailConversion(kShader, "no matching uniform buffer", 2u, 4u);
}

TEST_F(ConvertUBOToPushConstantTest, AmbiguousBlockName) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 Binding 3",
          "OpDecorate %20 Binding 3\n"
          "OpDecorate %21 DescriptorSet 0\nOpDecorate %21 Binding 0");
  Replace(&text, "%20 = OpVariable %13 Uniform",
          "%20 = OpVariable %13 Uniform\n%21 = OpVariable %13 Uniform");
  FailConversion(text, "more than one uniform buffer", "Block");
}

TEST_F(ConvertUBOToPushConstantTest, AmbiguousBinding) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %20 Binding 3",
          "OpDecorate %20 Binding 3\n"
          "OpDecorate %21 DescriptorSet 2\nOpDecorate %21 Binding 3");
  Replace(&text, "%20 = OpVariable %13 Uniform",
          "%20 = OpVariable %13 Uniform\n%21 = OpVariable %13 Uniform");
  FailConversion(text, "more than one uniform buffer", 2u, 3u);
}

TEST_F(ConvertUBOToPushConstantTest, RejectsBufferBlock) {
  std::string text = kShader;
  Replace(&text, "OpDecorate %12 Block", "OpDecorate %12 BufferBlock");
  FailConversion(text, "no matching uniform buffer", "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, RejectsDescriptorArray) {
  std::string text = kShader;
  Replace(&text, "%13 = OpTypePointer Uniform %12",
          "%16 = OpTypeArray %12 %8\n%13 = OpTypePointer Uniform %16");
  text.erase(text.find("%30 = OpAccessChain"),
             text.find("OpReturn") - text.find("%30 = OpAccessChain"));
  FailConversion(text, "no matching uniform buffer", "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, ExistingPushConstantVariable) {
  std::string text = kShader;
  Replace(&text, "%20 = OpVariable",
          "%16 = OpTypePointer PushConstant %12\n"
          "%21 = OpVariable %16 PushConstant\n%20 = OpVariable");
  FailConversion(text, "already contains a push constant", "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, PointerFunctionArgument) {
  // HLSL frontends may emit pointer arguments before legalization. The pass
  // must diagnose these even when the function returns no pointer (or void).
  spvValidatorOptionsSetBeforeHlslLegalization(ValidatorOptions(), true);
  std::string text = kShader;
  Replace(&text, "%20 = OpVariable",
          "%16 = OpTypeFunction %1 %13\n%20 = OpVariable");
  Replace(&text, "%30 = OpAccessChain",
          "%36 = OpFunctionCall %1 %200 %20\n%30 = OpAccessChain");
  text +=
      "%200 = OpFunction %1 None %16\n%201 = OpFunctionParameter %13\n"
      "%202 = OpLabel\nOpReturn\nOpFunctionEnd\n";
  FailConversion(text, "pointer arguments must be inlined", "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, RejectsRuntimeArray) {
  std::string text = kShader;
  Replace(&text, "%10 = OpTypeArray %7 %8", "%10 = OpTypeRuntimeArray %7");
  Replace(&text, "%12 = OpTypeStruct %7 %10 %11",
          "%12 = OpTypeStruct %7 %11 %10");
  Replace(&text, "OpMemberDecorate %12 2 Offset 48",
          "OpMemberDecorate %12 2 Offset 80");
  Replace(&text, "OpMemberDecorate %12 2 MatrixStride",
          "OpMemberDecorate %12 1 MatrixStride");
  Replace(&text, "OpMemberDecorate %12 2 ColMajor",
          "OpMemberDecorate %12 1 ColMajor");
  text.erase(text.find("%30 = OpAccessChain"),
             text.find("OpReturn") - text.find("%30 = OpAccessChain"));
  FailConversion(text, "cannot contain a runtime array", "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, RejectsUnsupportedMemoryAccess) {
  std::string text = kShader;
  Replace(&text, "OpMemoryModel Logical GLSL450",
          "OpCapability VulkanMemoryModel\n"
          "OpExtension \"SPV_KHR_vulkan_memory_model\"\nOpMemoryModel Logical "
          "Vulkan");
  Replace(&text, "%31 = OpLoad %7 %30",
          "%31 = OpLoad %7 %30 NonPrivatePointer");
  FailConversion(text, "NonPrivatePointer memory access", "buffer");
}

TEST_F(ConvertUBOToPushConstantTest,
       RejectsMissing16BitPushConstantCapability) {
  FailConversion(SmallTypeShader(16),
                 "16-bit members require StoragePushConstant16", "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, RejectsMissing8BitPushConstantCapability) {
  FailConversion(SmallTypeShader(8),
                 "8-bit members require StoragePushConstant8", "buffer");
}

TEST_F(ConvertUBOToPushConstantTest, SmallTypesWithPushConstantCapabilities) {
  for (uint32_t width : {8u, 16u}) {
    std::string text = SmallTypeShader(width);
    Replace(&text, "OpCapability Shader",
            "OpCapability Shader\nOpCapability StoragePushConstant" +
                std::to_string(width));
    ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
    CheckPointer(20, spv::StorageClass::PushConstant, 12);
  }
}

TEST_F(ConvertUBOToPushConstantTest, SmallTypesWithArithmeticCapabilities) {
  for (uint32_t width : {8u, 16u}) {
    std::string text = SmallTypeShader(width);
    Replace(&text, "OpCapability Shader",
            "OpCapability Shader\nOpCapability Int" + std::to_string(width));
    ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  }
}

TEST_F(ConvertUBOToPushConstantTest, LongVectorComponentCapabilities) {
  for (uint32_t width : {8u, 16u}) {
    SCOPED_TRACE(width);
    std::string text = SmallTypeShader(width);
    Replace(&text, "OpCapability Shader",
            "OpCapability Shader\nOpCapability LongVectorEXT");
    Replace(&text, "OpMemoryModel",
            "OpExtension \"SPV_EXT_long_vector\"\nOpMemoryModel");
    Replace(&text, "%12 = OpTypeStruct %3",
            "%4 = OpTypeInt 32 0\n%5 = OpConstant %4 8\n"
            "%7 = OpTypeVectorIdEXT %3 %5\n%12 = OpTypeStruct %7");
    ASSERT_NO_FATAL_FAILURE(FailConversion(
        text,
        std::to_string(width) + "-bit members require StoragePushConstant" +
            std::to_string(width),
        "buffer"));
    Replace(&text, "OpCapability Shader",
            "OpCapability Shader\nOpCapability StoragePushConstant" +
                std::to_string(width));
    ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
    CheckPointer(20, spv::StorageClass::PushConstant, 12);
  }
}

TEST_F(ConvertUBOToPushConstantTest, Float8DoesNotRequireIntegerCapabilities) {
  for (const char* encoding : {"Float8E4M3EXT", "Float8E5M2EXT"}) {
    SCOPED_TRACE(encoding);
    std::string text = SmallTypeShader(8);
    Replace(&text, "UniformAndStorageBuffer8BitAccess", "Float8EXT");
    Replace(&text, "SPV_KHR_8bit_storage", "SPV_EXT_float8");
    Replace(&text, "OpTypeInt 8 0", std::string("OpTypeFloat 8 ") + encoding);
    ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
    CheckPointer(20, spv::StorageClass::PushConstant, 12);
  }
}

TEST_F(ConvertUBOToPushConstantTest, LoadedPhysicalPointerKeepsStorageClass) {
  const std::string text = R"(
OpCapability Shader
OpCapability PhysicalStorageBufferAddresses
OpExtension "SPV_KHR_physical_storage_buffer"
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint GLCompute %100 "main"
OpExecutionMode %100 LocalSize 1 1 1
OpName %20 "buffer"
OpDecorate %12 Block
OpMemberDecorate %12 0 Offset 0
OpDecorate %20 DescriptorSet 2
OpDecorate %20 Binding 3
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeInt 32 0
%5 = OpConstant %4 0
%6 = OpTypePointer PhysicalStorageBuffer %3
%12 = OpTypeStruct %6
%13 = OpTypePointer Uniform %12
%14 = OpTypePointer Uniform %6
%20 = OpVariable %13 Uniform
%100 = OpFunction %1 None %2
%101 = OpLabel
%30 = OpAccessChain %14 %20 %5
%31 = OpLoad %6 %30
%32 = OpLoad %3 %31 Aligned 4
OpReturn
OpFunctionEnd
)";
  ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
  CheckPointer(30, spv::StorageClass::PushConstant, 6);
  CheckPointer(31, spv::StorageClass::PhysicalStorageBuffer, 3);
}

TEST_F(ConvertUBOToPushConstantTest, PreservesDebugGlobalVariable) {
  std::string text = R"(
OpCapability Shader
%50 = OpExtInstImport "OpenCL.DebugInfo.100"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %100 "main"
OpExecutionMode %100 LocalSize 1 1 1
%51 = OpString "test.hlsl"
%52 = OpString "float"
%53 = OpString "Block"
%54 = OpString "color"
OpName %20 "buffer"
OpDecorate %12 Block
OpMemberDecorate %12 0 Offset 0
OpDecorate %20 DescriptorSet 2
OpDecorate %20 Binding 3
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpTypeFloat 32
%4 = OpTypeInt 32 0
%5 = OpConstant %4 0
%6 = OpConstant %4 32
%7 = OpTypeVector %3 4
%8 = OpConstant %4 128
%12 = OpTypeStruct %7
%13 = OpTypePointer Uniform %12
%14 = OpTypePointer Uniform %7
%20 = OpVariable %13 Uniform
%55 = OpExtInst %1 %50 DebugSource %51
%56 = OpExtInst %1 %50 DebugCompilationUnit 1 4 %55 HLSL
%57 = OpExtInst %1 %50 DebugTypeBasic %52 %6 Float
%58 = OpExtInst %1 %50 DebugTypeVector %57 4
%59 = OpExtInst %1 %50 DebugTypeComposite %53 Structure %55 1 1 %56 %53 %8 FlagIsPublic %60
%60 = OpExtInst %1 %50 DebugTypeMember %54 %58 %55 1 1 %59 %5 %8 FlagIsPublic
%61 = OpExtInst %1 %50 DebugGlobalVariable %53 %59 %55 1 1 %56 %53 %20 FlagIsDefinition
%100 = OpFunction %1 None %2
%101 = OpLabel
%30 = OpAccessChain %14 %20 %5
%31 = OpLoad %7 %30
OpReturn
OpFunctionEnd
)";
  for (bool nonsemantic : {false, true}) {
    SCOPED_TRACE(nonsemantic);
    if (nonsemantic) {
      Replace(&text, "OpCapability Shader",
              "OpCapability Shader\nOpExtension \"SPV_KHR_non_semantic_info\"");
      Replace(&text, "OpenCL.DebugInfo.100",
              "NonSemantic.Shader.DebugInfo.100");
      Replace(&text, "%8 = OpConstant %4 128",
              "%8 = OpConstant %4 128\n%9 = OpConstant %4 1\n"
              "%10 = OpConstant %4 4\n%11 = OpConstant %4 5\n"
              "%15 = OpConstant %4 3\n%16 = OpConstant %4 8");
      Replace(&text, "DebugCompilationUnit 1 4 %55 HLSL",
              "DebugCompilationUnit %9 %10 %55 %11");
      Replace(&text, "DebugTypeBasic %52 %6 Float",
              "DebugTypeBasic %52 %6 %15 %5");
      Replace(&text, "DebugTypeVector %57 4", "DebugTypeVector %57 %10");
      Replace(
          &text,
          "DebugTypeComposite %53 Structure %55 1 1 %56 %53 %8 FlagIsPublic",
          "DebugTypeComposite %53 %9 %55 %9 %9 %56 %53 %8 %15");
      Replace(&text, "DebugTypeMember %54 %58 %55 1 1 %59 %5 %8 FlagIsPublic",
              "DebugTypeMember %54 %58 %55 %9 %9 %5 %8 %15");
      Replace(
          &text,
          "DebugGlobalVariable %53 %59 %55 1 1 %56 %53 %20 FlagIsDefinition",
          "DebugGlobalVariable %53 %59 %55 %9 %9 %56 %53 %20 %16");
      // Nonsemantic members have no parent operand, so define them before the
      // composite type instead of relying on legacy debug forward references.
      const auto member_begin = text.find("%60 =");
      const auto member_end = text.find('\n', member_begin) + 1;
      const auto member = text.substr(member_begin, member_end - member_begin);
      text.erase(member_begin, member_end - member_begin);
      text.insert(text.find("%59 ="), member);
    }
    auto before = BuildModule(env_, consumer(), text, assemble_options_);
    ASSERT_NE(before, nullptr);
    const auto expected = before->get_def_use_mgr()->GetDef(61)->PrettyPrint();
    ASSERT_NO_FATAL_FAILURE(Convert(text, "buffer"));
    CheckPointer(20, spv::StorageClass::PushConstant, 12);
    CheckPointer(30, spv::StorageClass::PushConstant, 7);
    EXPECT_EQ(context()->get_def_use_mgr()->GetDef(61)->PrettyPrint(),
              expected);
  }
}

TEST_F(ConvertUBOToPushConstantTest, PublicFactoriesAndCommandLineOptions) {
  SpirvTools tools(env_);
  std::vector<uint32_t> input, output;
  ASSERT_TRUE(tools.Assemble(kShader, &input));
  for (const bool by_binding : {false, true}) {
    Optimizer optimizer(env_);
    optimizer.RegisterPass(by_binding
                               ? CreateConvertUBOToPushConstantPass(2, 3)
                               : CreateConvertUBOToPushConstantPass("buffer"));
    ASSERT_TRUE(optimizer.Run(input.data(), input.size(), &output));
    EXPECT_TRUE(tools.Validate(output));
  }
  for (const char* flag : {"--convert-ubo-to-push-constant=buffer",
                           "--convert-ubo-to-push-constant-by-binding=2:3"}) {
    Optimizer optimizer(env_);
    ASSERT_TRUE(optimizer.RegisterPassFromFlag(flag));
    ASSERT_TRUE(optimizer.Run(input.data(), input.size(), &output));
    EXPECT_TRUE(tools.Validate(output));
  }
}

TEST_F(ConvertUBOToPushConstantTest, InvalidCommandLineOptions) {
  for (const char* flag :
       {"--convert-ubo-to-push-constant", "--convert-ubo-to-push-constant=",
        "--convert-ubo-to-push-constant-by-binding",
        "--convert-ubo-to-push-constant-by-binding=",
        "--convert-ubo-to-push-constant-by-binding=2",
        "--convert-ubo-to-push-constant-by-binding=:3",
        "--convert-ubo-to-push-constant-by-binding=2:",
        "--convert-ubo-to-push-constant-by-binding=-2:3",
        "--convert-ubo-to-push-constant-by-binding=2:-3",
        "--convert-ubo-to-push-constant-by-binding=2:3junk",
        "--convert-ubo-to-push-constant-by-binding=2:3:4",
        "--convert-ubo-to-push-constant-by-binding=4294967296:3",
        "--convert-ubo-to-push-constant-by-binding=2:4294967296"}) {
    SCOPED_TRACE(flag);
    Optimizer optimizer(env_);
    optimizer.SetMessageConsumer(consumer());
    EXPECT_FALSE(optimizer.RegisterPassFromFlag(flag));
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
