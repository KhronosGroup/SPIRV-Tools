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

#include "opt/value_number_table.h"

#include "assembly_builder.h"
#include "gmock/gmock.h"
#include "opt/build_module.h"
#include "pass_fixture.h"

namespace {

using namespace spvtools;

using ::testing::HasSubstr;
using ::testing::MatchesRegex;

using ValueTableTest = PassTest<::testing::Test>;

TEST_F(ValueTableTest, SameInstructionSameValue) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %8 = OpVariable %6 Function
          %9 = OpLoad %5 %8
         %10 = OpFAdd %5 %9 %9
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst), vtable.GetValueNumber(inst));
}

TEST_F(ValueTableTest, DifferentInstructionSameValue) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %8 = OpVariable %6 Function
          %9 = OpLoad %5 %8
         %10 = OpFAdd %5 %9 %9
         %11 = OpFAdd %5 %9 %9
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst1 = context->get_def_use_mgr()->GetDef(10);
  ir::Instruction* inst2 = context->get_def_use_mgr()->GetDef(11);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentValue) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %8 = OpVariable %6 Function
          %9 = OpLoad %5 %8
         %10 = OpFAdd %5 %9 %9
         %11 = OpFAdd %5 %9 %10
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst1 = context->get_def_use_mgr()->GetDef(10);
  ir::Instruction* inst2 = context->get_def_use_mgr()->GetDef(11);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, SameLoad) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %8 = OpVariable %6 Function
          %9 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst = context->get_def_use_mgr()->GetDef(9);
  EXPECT_EQ(vtable.GetValueNumber(inst), vtable.GetValueNumber(inst));
}

// Two different loads, even from the same memory, must given different value
// numbers if the memory is not read-only.
TEST_F(ValueTableTest, DifferentFunctionLoad) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Function %5
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %8 = OpVariable %6 Function
          %9 = OpLoad %5 %8
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  ir::Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentUniformLoad) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Uniform %5
          %8 = OpVariable %6 Uniform
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %5 %8
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  ir::Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentInputLoad) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer Input %5
          %8 = OpVariable %6 Input
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %5 %8
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  ir::Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentUniformConstantLoad) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer UniformConstant %5
          %8 = OpVariable %6 UniformConstant
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %5 %8
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  ir::Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, DifferentPushConstantLoad) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypePointer PushConstant %5
          %8 = OpVariable %6 PushConstant
          %2 = OpFunction %3 None %4
          %7 = OpLabel
          %9 = OpLoad %5 %8
          %10 = OpLoad %5 %8
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst1 = context->get_def_use_mgr()->GetDef(9);
  ir::Instruction* inst2 = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

TEST_F(ValueTableTest, SameCall) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypeFunction %5
          %7 = OpTypePointer Function %5
          %8 = OpVariable %7 Private
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpFunctionCall %5 %11
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %5 None %6
         %12 = OpLabel
         %13 = OpLoad %5 %8
               OpReturnValue %13
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst = context->get_def_use_mgr()->GetDef(10);
  EXPECT_EQ(vtable.GetValueNumber(inst), vtable.GetValueNumber(inst));
}

// Function calls should be given a new value number, even if they are the same.
TEST_F(ValueTableTest, DifferentCall) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeFloat 32
          %6 = OpTypeFunction %5
          %7 = OpTypePointer Function %5
          %8 = OpVariable %7 Private
          %2 = OpFunction %3 None %4
          %9 = OpLabel
         %10 = OpFunctionCall %5 %11
         %12 = OpFunctionCall %5 %11
               OpReturn
               OpFunctionEnd
         %11 = OpFunction %5 None %6
         %13 = OpLabel
         %14 = OpLoad %5 %8
               OpReturnValue %14
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst1 = context->get_def_use_mgr()->GetDef(10);
  ir::Instruction* inst2 = context->get_def_use_mgr()->GetDef(12);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}

// It is possible to have two instruction that compute the same numerical value,
// but with different types.  They should have different value numbers.
TEST_F(ValueTableTest, DifferentTypes) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 0
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %5
          %2 = OpFunction %3 None %4
          %8 = OpLabel
          %9 = OpVariable %7 Function
         %10 = OpLoad %5 %9
         %11 = OpIAdd %5 %10 %10
         %12 = OpIAdd %6 %10 %10
               OpReturn
               OpFunctionEnd
  )";
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  opt::ValueNumberTable vtable(context.get());
  ir::Instruction* inst1 = context->get_def_use_mgr()->GetDef(11);
  ir::Instruction* inst2 = context->get_def_use_mgr()->GetDef(12);
  EXPECT_NE(vtable.GetValueNumber(inst1), vtable.GetValueNumber(inst2));
}
}  // anonymous namespace
