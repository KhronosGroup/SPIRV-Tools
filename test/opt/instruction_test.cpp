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

#include "opt/instruction.h"
#include "opt/ir_context.h"

#include "gmock/gmock.h"

#include "pass_fixture.h"
#include "pass_utils.h"
#include "spirv-tools/libspirv.h"
#include "unit_spirv.h"

namespace {

using namespace spvtools;
using ir::Instruction;
using ir::IRContext;
using ir::Operand;
using spvtest::MakeInstruction;
using ::testing::Eq;
using DescriptorTypeTest = PassTest<::testing::Test>;

TEST(InstructionTest, CreateTrivial) {
  Instruction empty;
  EXPECT_EQ(SpvOpNop, empty.opcode());
  EXPECT_EQ(0u, empty.type_id());
  EXPECT_EQ(0u, empty.result_id());
  EXPECT_EQ(0u, empty.NumOperands());
  EXPECT_EQ(0u, empty.NumOperandWords());
  EXPECT_EQ(0u, empty.NumInOperandWords());
  EXPECT_EQ(empty.cend(), empty.cbegin());
  EXPECT_EQ(empty.end(), empty.begin());
}

TEST(InstructionTest, CreateWithOpcodeAndNoOperands) {
  IRContext context(nullptr);
  Instruction inst(&context, SpvOpReturn);
  EXPECT_EQ(SpvOpReturn, inst.opcode());
  EXPECT_EQ(0u, inst.type_id());
  EXPECT_EQ(0u, inst.result_id());
  EXPECT_EQ(0u, inst.NumOperands());
  EXPECT_EQ(0u, inst.NumOperandWords());
  EXPECT_EQ(0u, inst.NumInOperandWords());
  EXPECT_EQ(inst.cend(), inst.cbegin());
  EXPECT_EQ(inst.end(), inst.begin());
}

// The words for an OpTypeInt for 32-bit signed integer resulting in Id 44.
uint32_t kSampleInstructionWords[] = {(4 << 16) | uint32_t(SpvOpTypeInt), 44,
                                      32, 1};
// The operands that would be parsed from kSampleInstructionWords
spv_parsed_operand_t kSampleParsedOperands[] = {
    {1, 1, SPV_OPERAND_TYPE_RESULT_ID, SPV_NUMBER_NONE, 0},
    {2, 1, SPV_OPERAND_TYPE_LITERAL_INTEGER, SPV_NUMBER_UNSIGNED_INT, 32},
    {3, 1, SPV_OPERAND_TYPE_LITERAL_INTEGER, SPV_NUMBER_UNSIGNED_INT, 1},
};

// A valid parse of kSampleParsedOperands.
spv_parsed_instruction_t kSampleParsedInstruction = {kSampleInstructionWords,
                                                     uint16_t(4),
                                                     uint16_t(SpvOpTypeInt),
                                                     SPV_EXT_INST_TYPE_NONE,
                                                     0,   // type id
                                                     44,  // result id
                                                     kSampleParsedOperands,
                                                     3};

// The words for an OpAccessChain instruction.
uint32_t kSampleAccessChainInstructionWords[] = {
    (7 << 16) | uint32_t(SpvOpAccessChain), 100, 101, 102, 103, 104, 105};

// The operands that would be parsed from kSampleAccessChainInstructionWords.
spv_parsed_operand_t kSampleAccessChainOperands[] = {
    {1, 1, SPV_OPERAND_TYPE_RESULT_ID, SPV_NUMBER_NONE, 0},
    {2, 1, SPV_OPERAND_TYPE_TYPE_ID, SPV_NUMBER_NONE, 0},
    {3, 1, SPV_OPERAND_TYPE_ID, SPV_NUMBER_NONE, 0},
    {4, 1, SPV_OPERAND_TYPE_ID, SPV_NUMBER_NONE, 0},
    {5, 1, SPV_OPERAND_TYPE_ID, SPV_NUMBER_NONE, 0},
    {6, 1, SPV_OPERAND_TYPE_ID, SPV_NUMBER_NONE, 0},
};

// A valid parse of kSampleAccessChainInstructionWords
spv_parsed_instruction_t kSampleAccessChainInstruction = {
    kSampleAccessChainInstructionWords,
    uint16_t(7),
    uint16_t(SpvOpAccessChain),
    SPV_EXT_INST_TYPE_NONE,
    100,  // type id
    101,  // result id
    kSampleAccessChainOperands,
    6};

// The words for an OpControlBarrier instruction.
uint32_t kSampleControlBarrierInstructionWords[] = {
    (4 << 16) | uint32_t(SpvOpControlBarrier), 100, 101, 102};

// The operands that would be parsed from kSampleControlBarrierInstructionWords.
spv_parsed_operand_t kSampleControlBarrierOperands[] = {
    {1, 1, SPV_OPERAND_TYPE_SCOPE_ID, SPV_NUMBER_NONE, 0},  // Execution
    {2, 1, SPV_OPERAND_TYPE_SCOPE_ID, SPV_NUMBER_NONE, 0},  // Memory
    {3, 1, SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID, SPV_NUMBER_NONE,
     0},  // Semantics
};

// A valid parse of kSampleControlBarrierInstructionWords
spv_parsed_instruction_t kSampleControlBarrierInstruction = {
    kSampleControlBarrierInstructionWords,
    uint16_t(4),
    uint16_t(SpvOpControlBarrier),
    SPV_EXT_INST_TYPE_NONE,
    0,  // type id
    0,  // result id
    kSampleControlBarrierOperands,
    3};

TEST(InstructionTest, CreateWithOpcodeAndOperands) {
  IRContext context(nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  EXPECT_EQ(SpvOpTypeInt, inst.opcode());
  EXPECT_EQ(0u, inst.type_id());
  EXPECT_EQ(44u, inst.result_id());
  EXPECT_EQ(3u, inst.NumOperands());
  EXPECT_EQ(3u, inst.NumOperandWords());
  EXPECT_EQ(2u, inst.NumInOperandWords());
}

TEST(InstructionTest, GetOperand) {
  IRContext context(nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  EXPECT_THAT(inst.GetOperand(0).words, Eq(std::vector<uint32_t>{44}));
  EXPECT_THAT(inst.GetOperand(1).words, Eq(std::vector<uint32_t>{32}));
  EXPECT_THAT(inst.GetOperand(2).words, Eq(std::vector<uint32_t>{1}));
}

TEST(InstructionTest, GetInOperand) {
  IRContext context(nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  EXPECT_THAT(inst.GetInOperand(0).words, Eq(std::vector<uint32_t>{32}));
  EXPECT_THAT(inst.GetInOperand(1).words, Eq(std::vector<uint32_t>{1}));
}

TEST(InstructionTest, OperandConstIterators) {
  IRContext context(nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  // Spot check iteration across operands.
  auto cbegin = inst.cbegin();
  auto cend = inst.cend();
  EXPECT_NE(cend, inst.cbegin());

  auto citer = inst.cbegin();
  for (int i = 0; i < 3; ++i, ++citer) {
    const auto& operand = *citer;
    EXPECT_THAT(operand.type, Eq(kSampleParsedOperands[i].type));
    EXPECT_THAT(operand.words,
                Eq(std::vector<uint32_t>{kSampleInstructionWords[i + 1]}));
    EXPECT_NE(cend, citer);
  }
  EXPECT_EQ(cend, citer);

  // Check that cbegin and cend have not changed.
  EXPECT_EQ(cbegin, inst.cbegin());
  EXPECT_EQ(cend, inst.cend());

  // Check arithmetic.
  const Operand& operand2 = *(inst.cbegin() + 2);
  EXPECT_EQ(SPV_OPERAND_TYPE_LITERAL_INTEGER, operand2.type);
}

TEST(InstructionTest, OperandIterators) {
  IRContext context(nullptr);
  Instruction inst(&context, kSampleParsedInstruction);
  // Spot check iteration across operands, with mutable iterators.
  auto begin = inst.begin();
  auto end = inst.end();
  EXPECT_NE(end, inst.begin());

  auto iter = inst.begin();
  for (int i = 0; i < 3; ++i, ++iter) {
    const auto& operand = *iter;
    EXPECT_THAT(operand.type, Eq(kSampleParsedOperands[i].type));
    EXPECT_THAT(operand.words,
                Eq(std::vector<uint32_t>{kSampleInstructionWords[i + 1]}));
    EXPECT_NE(end, iter);
  }
  EXPECT_EQ(end, iter);

  // Check that begin and end have not changed.
  EXPECT_EQ(begin, inst.begin());
  EXPECT_EQ(end, inst.end());

  // Check arithmetic.
  Operand& operand2 = *(inst.begin() + 2);
  EXPECT_EQ(SPV_OPERAND_TYPE_LITERAL_INTEGER, operand2.type);

  // Check mutation through an iterator.
  operand2.type = SPV_OPERAND_TYPE_TYPE_ID;
  EXPECT_EQ(SPV_OPERAND_TYPE_TYPE_ID, (*(inst.cbegin() + 2)).type);
}

TEST(InstructionTest, ForInIdStandardIdTypes) {
  IRContext context(nullptr);
  Instruction inst(&context, kSampleAccessChainInstruction);

  std::vector<uint32_t> ids;
  inst.ForEachInId([&ids](const uint32_t* idptr) { ids.push_back(*idptr); });
  EXPECT_THAT(ids, Eq(std::vector<uint32_t>{102, 103, 104, 105}));

  ids.clear();
  inst.ForEachInId([&ids](uint32_t* idptr) { ids.push_back(*idptr); });
  EXPECT_THAT(ids, Eq(std::vector<uint32_t>{102, 103, 104, 105}));
}

TEST(InstructionTest, ForInIdNonstandardIdTypes) {
  IRContext context(nullptr);
  Instruction inst(&context, kSampleControlBarrierInstruction);

  std::vector<uint32_t> ids;
  inst.ForEachInId([&ids](const uint32_t* idptr) { ids.push_back(*idptr); });
  EXPECT_THAT(ids, Eq(std::vector<uint32_t>{100, 101, 102}));

  ids.clear();
  inst.ForEachInId([&ids](uint32_t* idptr) { ids.push_back(*idptr); });
  EXPECT_THAT(ids, Eq(std::vector<uint32_t>{100, 101, 102}));
}

TEST(InstructionTest, UniqueIds) {
  IRContext context(nullptr);
  Instruction inst1(&context);
  Instruction inst2(&context);
  EXPECT_NE(inst1.unique_id(), inst2.unique_id());
}

TEST(InstructionTest, CloneUniqueIdDifferent) {
  IRContext context(nullptr);
  Instruction inst(&context);
  std::unique_ptr<Instruction> clone(inst.Clone(&context));
  EXPECT_EQ(inst.context(), clone->context());
  EXPECT_NE(inst.unique_id(), clone->unique_id());
}

TEST(InstructionTest, CloneDifferentContext) {
  IRContext c1(nullptr);
  IRContext c2(nullptr);
  Instruction inst(&c1);
  std::unique_ptr<Instruction> clone(inst.Clone(&c2));
  EXPECT_EQ(&c1, inst.context());
  EXPECT_EQ(&c2, clone->context());
  EXPECT_NE(&c1, &c2);
}

TEST(InstructionTest, CloneDifferentContextDifferentUniqueId) {
  IRContext c1(nullptr);
  IRContext c2(nullptr);
  Instruction inst(&c1);
  Instruction other(&c2);
  std::unique_ptr<Instruction> clone(inst.Clone(&c2));
  EXPECT_EQ(&c2, clone->context());
  EXPECT_NE(other.unique_id(), clone->unique_id());
}

TEST(InstructionTest, EqualsEqualsOperator) {
  IRContext context(nullptr);
  Instruction i1(&context);
  Instruction i2(&context);
  std::unique_ptr<Instruction> clone(i1.Clone(&context));
  EXPECT_TRUE(i1 == i1);
  EXPECT_FALSE(i1 == i2);
  EXPECT_FALSE(i1 == *clone);
  EXPECT_FALSE(i2 == *clone);
}

TEST(InstructionTest, LessThanOperator) {
  IRContext context(nullptr);
  Instruction i1(&context);
  Instruction i2(&context);
  std::unique_ptr<Instruction> clone(i1.Clone(&context));
  EXPECT_TRUE(i1 < i2);
  EXPECT_TRUE(i1 < *clone);
  EXPECT_TRUE(i2 < *clone);
}

TEST_F(DescriptorTypeTest, StorageImage) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeImage %6 2D 0 0 0 2 R32f
          %8 = OpTypePointer UniformConstant %7
          %3 = OpVariable %8 UniformConstant
          %2 = OpFunction %4 None %5
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(8);
  EXPECT_TRUE(type->IsVulkanStorageImage());
  EXPECT_FALSE(type->IsVulkanSampledImage());
  EXPECT_FALSE(type->IsVulkanStorageTexelBuffer());
  EXPECT_FALSE(type->IsVulkanStorageBuffer());
  EXPECT_FALSE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_FALSE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, SampledImage) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeImage %6 2D 0 0 0 1 Unknown
          %8 = OpTypePointer UniformConstant %7
          %3 = OpVariable %8 UniformConstant
          %2 = OpFunction %4 None %5
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(8);
  EXPECT_FALSE(type->IsVulkanStorageImage());
  EXPECT_TRUE(type->IsVulkanSampledImage());
  EXPECT_FALSE(type->IsVulkanStorageTexelBuffer());
  EXPECT_FALSE(type->IsVulkanStorageBuffer());
  EXPECT_FALSE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_TRUE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, StorageTexelBuffer) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeImage %6 Buffer 0 0 0 2 R32f
          %8 = OpTypePointer UniformConstant %7
          %3 = OpVariable %8 UniformConstant
          %2 = OpFunction %4 None %5
          %9 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(8);
  EXPECT_FALSE(type->IsVulkanStorageImage());
  EXPECT_FALSE(type->IsVulkanSampledImage());
  EXPECT_TRUE(type->IsVulkanStorageTexelBuffer());
  EXPECT_FALSE(type->IsVulkanStorageBuffer());
  EXPECT_FALSE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_FALSE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, StorageBuffer) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
               OpDecorate %9 BufferBlock
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypeRuntimeArray %7
          %9 = OpTypeStruct %8
         %10 = OpTypePointer Uniform %9
          %3 = OpVariable %10 Uniform
          %2 = OpFunction %4 None %5
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(10);
  EXPECT_FALSE(type->IsVulkanStorageImage());
  EXPECT_FALSE(type->IsVulkanSampledImage());
  EXPECT_FALSE(type->IsVulkanStorageTexelBuffer());
  EXPECT_TRUE(type->IsVulkanStorageBuffer());
  EXPECT_FALSE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_FALSE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, UniformBuffer) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
               OpDecorate %9 Block
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypeRuntimeArray %7
          %9 = OpTypeStruct %8
         %10 = OpTypePointer Uniform %9
          %3 = OpVariable %10 Uniform
          %2 = OpFunction %4 None %5
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* type = context->get_def_use_mgr()->GetDef(10);
  EXPECT_FALSE(type->IsVulkanStorageImage());
  EXPECT_FALSE(type->IsVulkanSampledImage());
  EXPECT_FALSE(type->IsVulkanStorageTexelBuffer());
  EXPECT_FALSE(type->IsVulkanStorageBuffer());
  EXPECT_TRUE(type->IsVulkanUniformBuffer());

  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_TRUE(variable->IsReadOnlyVariable());
}

TEST_F(DescriptorTypeTest, NonWritableIsReadOnly) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "myStorageImage"
               OpDecorate %3 DescriptorSet 0
               OpDecorate %3 Binding 0
               OpDecorate %9 BufferBlock
               OpDecorate %3 NonWritable
          %4 = OpTypeVoid
          %5 = OpTypeFunction %4
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 4
          %8 = OpTypeRuntimeArray %7
          %9 = OpTypeStruct %8
         %10 = OpTypePointer Uniform %9
          %3 = OpVariable %10 Uniform
          %2 = OpFunction %4 None %5
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  Instruction* variable = context->get_def_use_mgr()->GetDef(3);
  EXPECT_TRUE(variable->IsReadOnlyVariable());
}
}  // anonymous namespace
