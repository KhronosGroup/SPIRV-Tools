// Copyright (c) 2018 Google LLC.
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

#include <gmock/gmock.h>

#ifdef SPIRV_EFFCEE
#include "effcee/effcee.h"
#endif

#include "../pass_fixture.h"
#include "opt/ir_builder.h"
#include "opt/loop_descriptor.h"
#include "opt/loop_peeling.h"

namespace {

using namespace spvtools;

using PeelingTest = PassTest<::testing::Test>;

bool Validate(const std::vector<uint32_t>& bin) {
  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_2;
  spv_context spvContext = spvContextCreate(target_env);
  spv_diagnostic diagnostic = nullptr;
  spv_const_binary_t binary = {bin.data(), bin.size()};
  spv_result_t error = spvValidate(spvContext, &binary, &diagnostic);
  if (error != 0) spvDiagnosticPrint(diagnostic);
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(spvContext);
  return error == 0;
}

void Match(const std::string& checks, ir::IRContext* context) {
  std::vector<uint32_t> bin;
  context->module()->ToBinary(&bin, true);
  EXPECT_TRUE(Validate(bin));
#ifdef SPIRV_EFFCEE
  std::string assembly;
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_2);
  EXPECT_TRUE(
      tools.Disassemble(bin, &assembly, SPV_BINARY_TO_TEXT_OPTION_NO_HEADER))
      << "Disassembling failed for shader:\n"
      << assembly << std::endl;
  auto match_result = effcee::Match(assembly, checks);
  EXPECT_EQ(effcee::Result::Status::Ok, match_result.status())
      << match_result.message() << "\nChecking result:\n"
      << assembly;
#endif  // ! SPIRV_EFFCEE
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

first test:
#version 330 core
void main() {
  for(int i = 0; i < 10; ++i) {
    if (i < 4)
      break;
  }
}

second test (with a common sub-expression elimination):
#version 330 core
void main() {
  for(int i = 0; i + 1 < 10; ++i) {
  }
}
*/
TEST_F(PeelingTest, CannotPeel) {
  {
    SCOPED_TRACE("loop with break");

    const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
      %int_4 = OpConstant %int 4
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %28 = OpPhi %int %int_0 %5 %27 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %bool %28 %int_10
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
         %21 = OpSLessThan %bool %28 %int_4
               OpSelectionMerge %23 None
               OpBranchConditional %21 %22 %23
         %22 = OpLabel
               OpBranch %12
         %23 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %27 = OpIAdd %int %28 %int_1
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

    std::unique_ptr<ir::IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ir::Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    ir::Function& f = *module->begin();
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    opt::InstructionBuilder builder(context.get(), &*f.begin());

    opt::LoopPeeling peel(context.get(), &*ld.begin());
    EXPECT_FALSE(peel.CanPeelLoop());
  }

  {
    SCOPED_TRACE("Ambiguous iterator update");

    const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %23 = OpPhi %int %int_0 %5 %17 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %17 = OpIAdd %int %23 %int_1
         %20 = OpSLessThan %bool %17 %int_10
               OpBranchConditional %20 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

    std::unique_ptr<ir::IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ir::Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    ir::Function& f = *module->begin();
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    opt::InstructionBuilder builder(context.get(), &*f.begin());

    opt::LoopPeeling peel(context.get(), &*ld.begin());
    EXPECT_FALSE(peel.CanPeelLoop());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
void main() {
  int i = 0;
  for (; i < 10; i++) {}
}
*/
TEST_F(PeelingTest, SimplePeeling) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %22 = OpPhi %int %int_0 %5 %21 %13
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %18 = OpSLessThan %bool %22 %int_10
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %21 = OpIAdd %int %22 %int_1
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  // Peel before.
  {
    SCOPED_TRACE("Peel before");

    std::unique_ptr<ir::IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ir::Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    ir::Function& f = *module->begin();
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    opt::InstructionBuilder builder(context.get(), &*f.begin());
    ir::Instruction* two_cst = builder.Add32BitSignedIntegerConstant(2);

    opt::LoopPeeling peel(context.get(), &*ld.begin());
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelBefore(two_cst);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[BEFORE_LOOP:%\w+]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[i:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[AFTER_LOOP:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[AFTER_LOOP]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[i]]
CHECK-NEXT: [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} [[i]] [[COND_BLOCK]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }

  // Peel after.
  {
    SCOPED_TRACE("Peel after");

    std::unique_ptr<ir::IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ir::Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    ir::Function& f = *module->begin();
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    opt::InstructionBuilder builder(context.get(), &*f.begin());
    ir::Instruction* two_cst = builder.Add32BitSignedIntegerConstant(2);
    ir::Instruction* loop_count = builder.Add32BitSignedIntegerConstant(10);

    opt::LoopPeeling peel(context.get(), &*ld.begin());
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelAfter(two_cst, loop_count);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[BEFORE_LOOP:%\w+]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[i:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[AFTER_LOOP:%\w+]] [[BE]] None
CHECK:      [[COND_BLOCK:%\w+]] = OpLabel
CHECK-NEXT: OpSLessThan
CHECK-NEXT: [[TMP:%\w+]] = OpIAdd {{%\w+}} [[DUMMY_IT]] {{%\w+}}
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[TMP]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] {{%\w+}} [[AFTER_LOOP]]
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[i]]
CHECK-NEXT: [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: OpBranch [[BEFORE_LOOP]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} [[i]] [[COND_BLOCK]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
void main() {
  int i = 0;
  do {
    i++;
  } while (i < 10);
}
*/
TEST_F(PeelingTest, DoWhilePeeling) {
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 330
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %21 = OpPhi %int %int_0 %5 %16 %13
               OpLoopMerge %12 %13 None
               OpBranch %11
         %11 = OpLabel
         %16 = OpIAdd %int %21 %int_1
               OpBranch %13
         %13 = OpLabel
         %20 = OpSLessThan %bool %16 %int_10
               OpBranchConditional %20 %10 %12
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  // Peel before.
  {
    SCOPED_TRACE("Peel before");

    std::unique_ptr<ir::IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ir::Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    ir::Function& f = *module->begin();
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    opt::InstructionBuilder builder(context.get(), &*f.begin());
    ir::Instruction* two_cst = builder.Add32BitSignedIntegerConstant(2);

    opt::LoopPeeling peel(context.get(), &*ld.begin());
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelBefore(two_cst);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[BEFORE_LOOP:%\w+]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[i:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[AFTER_LOOP:%\w+]] [[BE]] None
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[i]]
CHECK:      [[BE]] = OpLabel
CHECK:      [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[DUMMY_IT_1]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] [[BEFORE_LOOP]] [[AFTER_LOOP]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} [[I_1]] [[BE]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }

  // Peel after.
  {
    SCOPED_TRACE("Peel after");

    std::unique_ptr<ir::IRContext> context =
        BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text,
                    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ir::Module* module = context->module();
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << text << std::endl;
    ir::Function& f = *module->begin();
    ir::LoopDescriptor& ld = *context->GetLoopDescriptor(&f);

    EXPECT_EQ(ld.NumLoops(), 1u);

    opt::InstructionBuilder builder(context.get(), &*f.begin());
    ir::Instruction* two_cst = builder.Add32BitSignedIntegerConstant(2);
    ir::Instruction* loop_count = builder.Add32BitSignedIntegerConstant(10);

    opt::LoopPeeling peel(context.get(), &*ld.begin());
    EXPECT_TRUE(peel.CanPeelLoop());
    peel.PeelAfter(two_cst, loop_count);

    const std::string check = R"(
CHECK:      OpFunction
CHECK-NEXT: [[ENTRY:%\w+]] = OpLabel
CHECK:      [[BEFORE_LOOP:%\w+]] = OpLabel
CHECK-NEXT: [[DUMMY_IT:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[DUMMY_IT_1:%\w+]] [[BE:%\w+]]
CHECK-NEXT: [[i:%\w+]] = OpPhi {{%\w+}} {{%\w+}} [[ENTRY]] [[I_1:%\w+]] [[BE]]
CHECK-NEXT: OpLoopMerge [[AFTER_LOOP:%\w+]] [[BE]] None
CHECK:      [[I_1]] = OpIAdd {{%\w+}} [[i]]
CHECK:      [[BE]] = OpLabel
CHECK:      [[DUMMY_IT_1]] = OpIAdd {{%\w+}} [[DUMMY_IT]]
CHECK-NEXT: [[EXIT_VAL:%\w+]] = OpIAdd {{%\w+}} [[DUMMY_IT_1]]
CHECK-NEXT: [[EXIT_COND:%\w+]] = OpSLessThan {{%\w+}} [[EXIT_VAL]]
CHECK-NEXT: OpBranchConditional [[EXIT_COND]] [[BEFORE_LOOP]] [[AFTER_LOOP]]

CHECK:      [[AFTER_LOOP]] = OpLabel
CHECK-NEXT: OpPhi {{%\w+}} [[I_1]] [[BE]]
CHECK-NEXT: OpLoopMerge
)";

    Match(check, context.get());
  }
}

}  // namespace
