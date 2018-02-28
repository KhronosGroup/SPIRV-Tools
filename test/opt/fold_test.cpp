// Copyright (c) 2016 Google Inc.
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

#include <memory>
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opt/fold.h>

#ifdef SPIRV_EFFCEE
#include "effcee/effcee.h"
#endif

#include "opt/build_module.h"
#include "opt/def_use_manager.h"
#include "opt/ir_context.h"
#include "opt/module.h"
#include "pass_utils.h"
#include "spirv-tools/libspirv.hpp"

namespace {

using ::testing::Contains;

using namespace spvtools;
using spvtools::opt::analysis::DefUseManager;

std::string Disassemble(const std::string& original, ir::IRContext* context,
                        uint32_t disassemble_options = 0) {
  std::vector<uint32_t> optimized_bin;
  context->module()->ToBinary(&optimized_bin, true);
  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_2;
  SpirvTools tools(target_env);
  std::string optimized_asm;
  EXPECT_TRUE(
      tools.Disassemble(optimized_bin, &optimized_asm, disassemble_options))
      << "Disassembling failed for shader:\n"
      << original << std::endl;
  return optimized_asm;
}

#ifdef SPIRV_EFFCEE
void Match(const std::string& original, ir::IRContext* context,
           uint32_t disassemble_options = 0) {
  std::string disassembly = Disassemble(original, context, disassemble_options);
  auto match_result = effcee::Match(disassembly, original);
  EXPECT_EQ(effcee::Result::Status::Ok, match_result.status())
      << match_result.message() << "\nChecking result:\n"
      << disassembly;
}
#endif

template <class ResultType>
struct InstructionFoldingCase {
  InstructionFoldingCase(const std::string& tb, uint32_t id, ResultType result)
      : test_body(tb), id_to_fold(id), expected_result(result) {}

  std::string test_body;
  uint32_t id_to_fold;
  ResultType expected_result;
};

using IntegerInstructionFoldingTest =
    ::testing::TestWithParam<InstructionFoldingCase<uint32_t>>;

TEST_P(IntegerInstructionFoldingTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  bool succeeded = opt::FoldInstruction(inst);

  // Make sure the instruction folded as expected.
  EXPECT_TRUE(succeeded);
  if (inst != nullptr) {
    EXPECT_EQ(inst->opcode(), SpvOpCopyObject);
    inst = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
    EXPECT_EQ(inst->opcode(), SpvOpConstant);
    opt::analysis::ConstantManager* const_mrg = context->get_constant_mgr();
    const opt::analysis::IntConstant* result =
        const_mrg->GetConstantFromInst(inst)->AsIntConstant();
    EXPECT_NE(result, nullptr);
    if (result != nullptr) {
      EXPECT_EQ(result->GetU32BitValue(), tc.expected_result);
    }
  }
}

// Returns a common SPIR-V header for all of the test that follow.
#define INT_0_ID 100
#define TRUE_ID 101
#define VEC2_0_ID 102
#define INT_7_ID 103
#define FLOAT_0_ID 104
#define DOUBLE_0_ID 105
#define VEC4_0_ID 106
#define DVEC4_0_ID 106
const std::string& Header() {
  static const std::string header = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
%void = OpTypeVoid
%void_func = OpTypeFunction %void
%bool = OpTypeBool
%float16 = OpTypeFloat 16
%float = OpTypeFloat 32
%double = OpTypeFloat 64
%101 = OpConstantTrue %bool ; Need a def with an numerical id to define id maps.
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%short = OpTypeInt 16 1
%int = OpTypeInt 32 1
%long = OpTypeInt 64 1
%uint = OpTypeInt 32 0
%v2int = OpTypeVector %int 2
%v4int = OpTypeVector %int 4
%v4float = OpTypeVector %float 4
%v4double = OpTypeVector %double 4
%v2float = OpTypeVector %float 2
%struct_v2int_int_int = OpTypeStruct %v2int %int %int
%_ptr_int = OpTypePointer Function %int
%_ptr_uint = OpTypePointer Function %uint
%_ptr_bool = OpTypePointer Function %bool
%_ptr_float = OpTypePointer Function %float
%_ptr_double = OpTypePointer Function %double
%_ptr_long = OpTypePointer Function %long
%_ptr_v2int = OpTypePointer Function %v2int
%_ptr_v4float = OpTypePointer Function %v4float
%_ptr_v4double = OpTypePointer Function %v4double
%_ptr_struct_v2int_int_int = OpTypePointer Function %struct_v2int_int_int
%_ptr_v2float = OpTypePointer Function %v2float
%short_0 = OpConstant %short 0
%short_3 = OpConstant %short 3
%100 = OpConstant %int 0 ; Need a def with an numerical id to define id maps.
%103 = OpConstant %int 7 ; Need a def with an numerical id to define id maps.
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%int_3 = OpConstant %int 3
%int_4 = OpConstant %int 4
%int_min = OpConstant %int -2147483648
%int_max = OpConstant %int 2147483647
%long_0 = OpConstant %long 0
%long_2 = OpConstant %long 2
%long_3 = OpConstant %long 3
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_32 = OpConstant %uint 32
%uint_max = OpConstant %uint 4294967295
%v2int_undef = OpUndef %v2int
%v2int_2_2 = OpConstantComposite %v2int %int_2 %int_2
%v2int_2_3 = OpConstantComposite %v2int %int_2 %int_3
%v2int_3_2 = OpConstantComposite %v2int %int_3 %int_2
%v2int_4_4 = OpConstantComposite %v2int %int_4 %int_4
%struct_v2int_int_int_null = OpConstantNull %struct_v2int_int_int
%v2int_null = OpConstantNull %v2int
%102 = OpConstantComposite %v2int %103 %103
%v4int_0_0_0_0 = OpConstantComposite %v4int %int_0 %int_0 %int_0 %int_0
%struct_undef_0_0 = OpConstantComposite %struct_v2int_int_int %v2int_undef %int_0 %int_0
%float16_0 = OpConstant %float16 0
%float16_1 = OpConstant %float16 1
%float16_2 = OpConstant %float16 2
%float_n1 = OpConstant %float -1
%104 = OpConstant %float 0 ; Need a def with an numerical id to define id maps.
%float_0 = OpConstant %float 0
%float_half = OpConstant %float 0.5
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%float_3 = OpConstant %float 3
%float_4 = OpConstant %float 4
%float_0p5 = OpConstant %float 0.5
%v2float_2_2 = OpConstantComposite %v2float %float_2 %float_2
%v2float_2_3 = OpConstantComposite %v2float %float_2 %float_3
%v2float_3_2 = OpConstantComposite %v2float %float_3 %float_2
%v2float_4_4 = OpConstantComposite %v2float %float_4 %float_4
%v2float_2_0p5 = OpConstantComposite %v2float %float_2 %float_0p5
%v2float_null = OpConstantNull %v2float
%double_n1 = OpConstant %double -1
%105 = OpConstant %double 0 ; Need a def with an numerical id to define id maps.
%double_0 = OpConstant %double 0
%double_1 = OpConstant %double 1
%double_2 = OpConstant %double 2
%double_3 = OpConstant %double 3
%float_nan = OpConstant %float -0x1.8p+128
%double_nan = OpConstant %double -0x1.8p+1024
%106 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%v4float_0_0_0_0 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%v4float_0_0_0_1 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_1
%v4float_1_1_1_1 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%107 = OpConstantComposite %v4double %double_0 %double_0 %double_0 %double_0
%v4double_0_0_0_0 = OpConstantComposite %v4double %double_0 %double_0 %double_0 %double_0
%v4double_0_0_0_1 = OpConstantComposite %v4double %double_0 %double_0 %double_0 %double_1
%v4double_1_1_1_1 = OpConstantComposite %v4double %double_1 %double_1 %double_1 %double_1
)";

  return header;
}

// clang-format off
INSTANTIATE_TEST_CASE_P(TestCase, IntegerInstructionFoldingTest,
                        ::testing::Values(
  // Test case 0: fold 0*n
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
    "%main_lab = OpLabel\n" +
           "%n = OpVariable %_ptr_int Function\n" +
        "%load = OpLoad %int %n\n" +
           "%2 = OpIMul %int %int_0 %load\n" +
                "OpReturn\n" +
                "OpFunctionEnd",
    2, 0),
  // Test case 1: fold n*0
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_int Function\n" +
        "%load = OpLoad %int %n\n" +
        "%2 = OpIMul %int %load %int_0\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 2: fold 0/n (signed)
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_int Function\n" +
        "%load = OpLoad %int %n\n" +
        "%2 = OpSDiv %int %int_0 %load\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
        2, 0),
  // Test case 3: fold n/0 (signed)
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_int Function\n" +
        "%load = OpLoad %int %n\n" +
        "%2 = OpSDiv %int %load %int_0\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 4: fold 0/n (unsigned)
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_uint Function\n" +
        "%load = OpLoad %uint %n\n" +
        "%2 = OpUDiv %uint %uint_0 %load\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 5: fold n/0 (unsigned)
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_int Function\n" +
        "%load = OpLoad %int %n\n" +
        "%2 = OpSDiv %int %load %int_0\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 6: fold 0 remainder n
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_int Function\n" +
        "%load = OpLoad %int %n\n" +
        "%2 = OpSRem %int %int_0 %load\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 7: fold n remainder 0
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_int Function\n" +
        "%load = OpLoad %int %n\n" +
        "%2 = OpSRem %int %load %int_0\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 8: fold 0%n (signed)
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_int Function\n" +
        "%load = OpLoad %int %n\n" +
        "%2 = OpSMod %int %int_0 %load\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 9: fold n%0 (signed)
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_int Function\n" +
        "%load = OpLoad %int %n\n" +
        "%2 = OpSMod %int %load %int_0\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 10: fold 0%n (unsigned)
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_uint Function\n" +
        "%load = OpLoad %uint %n\n" +
        "%2 = OpUMod %uint %uint_0 %load\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 11: fold n%0 (unsigned)
  InstructionFoldingCase<uint32_t>(
    Header() + "%main = OpFunction %void None %void_func\n" +
        "%main_lab = OpLabel\n" +
        "%n = OpVariable %_ptr_uint Function\n" +
        "%load = OpLoad %uint %n\n" +
        "%2 = OpUMod %uint %load %uint_0\n" +
        "OpReturn\n" +
        "OpFunctionEnd",
    2, 0),
  // Test case 12: fold n << 32
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpShiftLeftLogical %uint %load %uint_32\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, 0),
  // Test case 13: fold n >> 32
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpShiftRightLogical %uint %load %uint_32\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, 0),
  // Test case 14: fold n | 0xFFFFFFFF
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
  "%main_lab = OpLabel\n" +
  "%n = OpVariable %_ptr_uint Function\n" +
  "%load = OpLoad %uint %n\n" +
  "%2 = OpBitwiseOr %uint %load %uint_max\n" +
  "OpReturn\n" +
  "OpFunctionEnd",
  2, 0xFFFFFFFF),
  // Test case 15: fold 0xFFFFFFFF | n
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpBitwiseOr %uint %uint_max %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, 0xFFFFFFFF),
  // Test case 16: fold n & 0
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpBitwiseAnd %uint %load %uint_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, 0)
));
// clang-format on

using IntVectorInstructionFoldingTest =
    ::testing::TestWithParam<InstructionFoldingCase<std::vector<uint32_t>>>;

TEST_P(IntVectorInstructionFoldingTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  bool succeeded = opt::FoldInstruction(inst);

  // Make sure the instruction folded as expected.
  EXPECT_TRUE(succeeded);
  if (inst != nullptr) {
    EXPECT_EQ(inst->opcode(), SpvOpCopyObject);
    inst = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
    std::vector<SpvOp> opcodes = {SpvOpConstantComposite};
    EXPECT_THAT(opcodes, Contains(inst->opcode()));
    opt::analysis::ConstantManager* const_mrg = context->get_constant_mgr();
    const opt::analysis::Constant* result =
        const_mrg->GetConstantFromInst(inst);
    EXPECT_NE(result, nullptr);
    if (result != nullptr) {
      const std::vector<const opt::analysis::Constant*>& componenets =
          result->AsVectorConstant()->GetComponents();
      EXPECT_EQ(componenets.size(), tc.expected_result.size());
      for (size_t i = 0; i < componenets.size(); i++) {
        EXPECT_EQ(tc.expected_result[i], componenets[i]->GetU32());
      }
    }
  }
}

// clang-format off
INSTANTIATE_TEST_CASE_P(TestCase, IntVectorInstructionFoldingTest,
::testing::Values(
    // Test case 0: fold 0*n
    InstructionFoldingCase<std::vector<uint32_t>>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%load = OpLoad %int %n\n" +
            "%2 = OpVectorShuffle %v2int %v2int_2_2 %v2int_2_3 0 3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, {2,3}),
    InstructionFoldingCase<std::vector<uint32_t>>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %int %n\n" +
          "%2 = OpVectorShuffle %v2int %v2int_null %v2int_2_3 0 3\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, {0,3})
));
// clang-format on

using BooleanInstructionFoldingTest =
    ::testing::TestWithParam<InstructionFoldingCase<bool>>;

TEST_P(BooleanInstructionFoldingTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  bool succeeded = opt::FoldInstruction(inst);

  // Make sure the instruction folded as expected.
  EXPECT_TRUE(succeeded);
  if (inst != nullptr) {
    EXPECT_EQ(inst->opcode(), SpvOpCopyObject);
    inst = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
    std::vector<SpvOp> bool_opcodes = {SpvOpConstantTrue, SpvOpConstantFalse};
    EXPECT_THAT(bool_opcodes, Contains(inst->opcode()));
    opt::analysis::ConstantManager* const_mrg = context->get_constant_mgr();
    const opt::analysis::BoolConstant* result =
        const_mrg->GetConstantFromInst(inst)->AsBoolConstant();
    EXPECT_NE(result, nullptr);
    if (result != nullptr) {
      EXPECT_EQ(result->value(), tc.expected_result);
    }
  }
}

// clang-format off
INSTANTIATE_TEST_CASE_P(TestCase, BooleanInstructionFoldingTest,
                        ::testing::Values(
  // Test case 0: fold true || n
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_bool Function\n" +
          "%load = OpLoad %bool %n\n" +
          "%2 = OpLogicalOr %bool %true %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 1: fold n || true
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_bool Function\n" +
          "%load = OpLoad %bool %n\n" +
          "%2 = OpLogicalOr %bool %load %true\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 2: fold false && n
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_bool Function\n" +
          "%load = OpLoad %bool %n\n" +
          "%2 = OpLogicalAnd %bool %false %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 3: fold n && false
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_bool Function\n" +
          "%load = OpLoad %bool %n\n" +
          "%2 = OpLogicalAnd %bool %load %false\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 4: fold n < 0 (unsigned)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpULessThan %bool %load %uint_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 5: fold UINT_MAX < n (unsigned)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpULessThan %bool %uint_max %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 6: fold INT_MAX < n (signed)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %int %n\n" +
          "%2 = OpSLessThan %bool %int_max %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 7: fold n < INT_MIN (signed)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %int %n\n" +
          "%2 = OpSLessThan %bool %load %int_min\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 8: fold 0 > n (unsigned)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpUGreaterThan %bool %uint_0 %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 9: fold n > UINT_MAX (unsigned)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpUGreaterThan %bool %load %uint_max\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 10: fold n > INT_MAX (signed)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %int %n\n" +
          "%2 = OpSGreaterThan %bool %load %int_max\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 11: fold INT_MIN > n (signed)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpSGreaterThan %bool %int_min %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 12: fold 0 <= n (unsigned)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpULessThanEqual %bool %uint_0 %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 13: fold n <= UINT_MAX (unsigned)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpULessThanEqual %bool %load %uint_max\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 14: fold INT_MIN <= n (signed)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %int %n\n" +
          "%2 = OpSLessThanEqual %bool %int_min %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 15: fold n <= INT_MAX (signed)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %int %n\n" +
          "%2 = OpSLessThanEqual %bool %load %int_max\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 16: fold n >= 0 (unsigned)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpUGreaterThanEqual %bool %load %uint_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 17: fold UINT_MAX >= n (unsigned)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_uint Function\n" +
          "%load = OpLoad %uint %n\n" +
          "%2 = OpUGreaterThanEqual %bool %uint_max %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 18: fold n >= INT_MIN (signed)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %int %n\n" +
          "%2 = OpSGreaterThanEqual %bool %load %int_min\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 19: fold INT_MAX >= n (signed)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %int %n\n" +
          "%2 = OpSGreaterThanEqual %bool %int_max %load\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true)
));
// clang-format on

using FloatInstructionFoldingTest =
    ::testing::TestWithParam<InstructionFoldingCase<float>>;

TEST_P(FloatInstructionFoldingTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  bool succeeded = opt::FoldInstruction(inst);

  // Make sure the instruction folded as expected.
  EXPECT_TRUE(succeeded);
  if (inst != nullptr) {
    EXPECT_EQ(inst->opcode(), SpvOpCopyObject);
    inst = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
    EXPECT_EQ(inst->opcode(), SpvOpConstant);
    opt::analysis::ConstantManager* const_mrg = context->get_constant_mgr();
    const opt::analysis::FloatConstant* result =
        const_mrg->GetConstantFromInst(inst)->AsFloatConstant();
    EXPECT_NE(result, nullptr);
    if (result != nullptr) {
      EXPECT_EQ(result->GetFloatValue(), tc.expected_result);
    }
  }
}

// Not testing NaNs because there are no expectations concerning NaNs according
// to the "Precision and Operation of SPIR-V Instructions" section of the Vulkan
// specification.

// clang-format off
INSTANTIATE_TEST_CASE_P(FloatConstantFoldingTest, FloatInstructionFoldingTest,
::testing::Values(
    // Test case 0: Fold 2.0 - 1.0
    InstructionFoldingCase<float>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpFSub %float %float_2 %float_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 1.0),
    // Test case 1: Fold 2.0 + 1.0
    InstructionFoldingCase<float>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpFAdd %float %float_2 %float_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3.0),
    // Test case 2: Fold 3.0 * 2.0
    InstructionFoldingCase<float>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpFMul %float %float_3 %float_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 6.0),
    // Test case 3: Fold 1.0 / 2.0
    InstructionFoldingCase<float>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpFDiv %float %float_1 %float_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0.5),
    // Test case 4: Fold 1.0 / 0.0
    InstructionFoldingCase<float>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpFDiv %float %float_1 %float_0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, std::numeric_limits<float>::infinity()),
    // Test case 4: Fold -1.0 / 0.0
    InstructionFoldingCase<float>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpFDiv %float %float_n1 %float_0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, -std::numeric_limits<float>::infinity())
));
// clang-format on

using DoubleInstructionFoldingTest =
    ::testing::TestWithParam<InstructionFoldingCase<double>>;

TEST_P(DoubleInstructionFoldingTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  bool succeeded = opt::FoldInstruction(inst);

  // Make sure the instruction folded as expected.
  EXPECT_TRUE(succeeded);
  if (inst != nullptr) {
    EXPECT_EQ(inst->opcode(), SpvOpCopyObject);
    inst = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
    EXPECT_EQ(inst->opcode(), SpvOpConstant);
    opt::analysis::ConstantManager* const_mrg = context->get_constant_mgr();
    const opt::analysis::FloatConstant* result =
        const_mrg->GetConstantFromInst(inst)->AsFloatConstant();
    EXPECT_NE(result, nullptr);
    if (result != nullptr) {
      EXPECT_EQ(result->GetDoubleValue(), tc.expected_result);
    }
  }
}

// clang-format off
INSTANTIATE_TEST_CASE_P(DoubleConstantFoldingTest, DoubleInstructionFoldingTest,
::testing::Values(
    // Test case 0: Fold 2.0 - 1.0
    InstructionFoldingCase<double>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpFSub %double %double_2 %double_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 1.0),
        // Test case 1: Fold 2.0 + 1.0
        InstructionFoldingCase<double>(
            Header() + "%main = OpFunction %void None %void_func\n" +
                "%main_lab = OpLabel\n" +
                "%2 = OpFAdd %double %double_2 %double_1\n" +
                "OpReturn\n" +
                "OpFunctionEnd",
            2, 3.0),
        // Test case 2: Fold 3.0 * 2.0
        InstructionFoldingCase<double>(
            Header() + "%main = OpFunction %void None %void_func\n" +
                "%main_lab = OpLabel\n" +
                "%2 = OpFMul %double %double_3 %double_2\n" +
                "OpReturn\n" +
                "OpFunctionEnd",
            2, 6.0),
        // Test case 3: Fold 1.0 / 2.0
        InstructionFoldingCase<double>(
            Header() + "%main = OpFunction %void None %void_func\n" +
                "%main_lab = OpLabel\n" +
                "%2 = OpFDiv %double %double_1 %double_2\n" +
                "OpReturn\n" +
                "OpFunctionEnd",
            2, 0.5),
        // Test case 4: Fold 1.0 / 0.0
        InstructionFoldingCase<double>(
            Header() + "%main = OpFunction %void None %void_func\n" +
                "%main_lab = OpLabel\n" +
                "%2 = OpFDiv %double %double_1 %double_0\n" +
                "OpReturn\n" +
                "OpFunctionEnd",
            2, std::numeric_limits<double>::infinity()),
        // Test case 4: Fold -1.0 / 0.0
        InstructionFoldingCase<double>(
            Header() + "%main = OpFunction %void None %void_func\n" +
                "%main_lab = OpLabel\n" +
                "%2 = OpFDiv %double %double_n1 %double_0\n" +
                "OpReturn\n" +
                "OpFunctionEnd",
            2, -std::numeric_limits<double>::infinity())
));
// clang-format on

// clang-format off
INSTANTIATE_TEST_CASE_P(DoubleOrderedCompareConstantFoldingTest, BooleanInstructionFoldingTest,
                        ::testing::Values(
  // Test case 0: fold 1.0 == 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdEqual %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 1: fold 1.0 != 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdNotEqual %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 2: fold 1.0 < 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThan %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 3: fold 1.0 > 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThan %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 4: fold 1.0 <= 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThanEqual %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 5: fold 1.0 >= 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThanEqual %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 6: fold 1.0 == 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdEqual %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 7: fold 1.0 != 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdNotEqual %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 8: fold 1.0 < 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThan %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 9: fold 1.0 > 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThan %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 10: fold 1.0 <= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThanEqual %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 11: fold 1.0 >= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThanEqual %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 12: fold 2.0 < 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThan %bool %double_2 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 13: fold 2.0 > 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThan %bool %double_2 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 14: fold 2.0 <= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThanEqual %bool %double_2 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 15: fold 2.0 >= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThanEqual %bool %double_2 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true)
));

INSTANTIATE_TEST_CASE_P(DoubleUnorderedCompareConstantFoldingTest, BooleanInstructionFoldingTest,
                        ::testing::Values(
  // Test case 0: fold 1.0 == 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordEqual %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 1: fold 1.0 != 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordNotEqual %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 2: fold 1.0 < 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThan %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 3: fold 1.0 > 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThan %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 4: fold 1.0 <= 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThanEqual %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 5: fold 1.0 >= 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThanEqual %bool %double_1 %double_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 6: fold 1.0 == 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordEqual %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 7: fold 1.0 != 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordNotEqual %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 8: fold 1.0 < 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThan %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 9: fold 1.0 > 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThan %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 10: fold 1.0 <= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThanEqual %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 11: fold 1.0 >= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThanEqual %bool %double_1 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 12: fold 2.0 < 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThan %bool %double_2 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 13: fold 2.0 > 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThan %bool %double_2 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 14: fold 2.0 <= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThanEqual %bool %double_2 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 15: fold 2.0 >= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThanEqual %bool %double_2 %double_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true)
));

INSTANTIATE_TEST_CASE_P(FloatOrderedCompareConstantFoldingTest, BooleanInstructionFoldingTest,
                        ::testing::Values(
  // Test case 0: fold 1.0 == 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdEqual %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 1: fold 1.0 != 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdNotEqual %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 2: fold 1.0 < 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThan %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 3: fold 1.0 > 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThan %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 4: fold 1.0 <= 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThanEqual %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 5: fold 1.0 >= 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThanEqual %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 6: fold 1.0 == 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdEqual %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 7: fold 1.0 != 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdNotEqual %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 8: fold 1.0 < 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThan %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 9: fold 1.0 > 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThan %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 10: fold 1.0 <= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThanEqual %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 11: fold 1.0 >= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThanEqual %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 12: fold 2.0 < 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThan %bool %float_2 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 13: fold 2.0 > 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThan %bool %float_2 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 14: fold 2.0 <= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdLessThanEqual %bool %float_2 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 15: fold 2.0 >= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdGreaterThanEqual %bool %float_2 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true)
));

INSTANTIATE_TEST_CASE_P(FloatUnorderedCompareConstantFoldingTest, BooleanInstructionFoldingTest,
                        ::testing::Values(
  // Test case 0: fold 1.0 == 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordEqual %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 1: fold 1.0 != 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordNotEqual %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 2: fold 1.0 < 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThan %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 3: fold 1.0 > 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThan %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 4: fold 1.0 <= 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThanEqual %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 5: fold 1.0 >= 2.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThanEqual %bool %float_1 %float_2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 6: fold 1.0 == 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordEqual %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 7: fold 1.0 != 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordNotEqual %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 8: fold 1.0 < 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThan %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 9: fold 1.0 > 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThan %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 10: fold 1.0 <= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThanEqual %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 11: fold 1.0 >= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThanEqual %bool %float_1 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 12: fold 2.0 < 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThan %bool %float_2 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 13: fold 2.0 > 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThan %bool %float_2 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 14: fold 2.0 <= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordLessThanEqual %bool %float_2 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 15: fold 2.0 >= 1.0
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordGreaterThanEqual %bool %float_2 %float_1\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true)
));

INSTANTIATE_TEST_CASE_P(DoubleNaNCompareConstantFoldingTest, BooleanInstructionFoldingTest,
                        ::testing::Values(
  // Test case 0: fold NaN == 0 (ord)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdEqual %bool %double_nan %double_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 1: fold NaN == NaN (unord)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordEqual %bool %double_nan %double_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 2: fold NaN != NaN (ord)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdNotEqual %bool %double_nan %double_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 3: fold NaN != NaN (unord)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordNotEqual %bool %double_nan %double_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true)
));

INSTANTIATE_TEST_CASE_P(FloatNaNCompareConstantFoldingTest, BooleanInstructionFoldingTest,
                        ::testing::Values(
  // Test case 0: fold NaN == 0 (ord)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdEqual %bool %float_nan %float_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 1: fold NaN == NaN (unord)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordEqual %bool %float_nan %float_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true),
  // Test case 2: fold NaN != NaN (ord)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFOrdNotEqual %bool %float_nan %float_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, false),
  // Test case 3: fold NaN != NaN (unord)
  InstructionFoldingCase<bool>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%2 = OpFUnordNotEqual %bool %float_nan %float_0\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, true)
));
// clang-format on

template <class ResultType>
struct InstructionFoldingCaseWithMap {
  InstructionFoldingCaseWithMap(const std::string& tb, uint32_t id,
                                ResultType result,
                                std::function<uint32_t(uint32_t)> map)
      : test_body(tb), id_to_fold(id), expected_result(result), id_map(map) {}

  std::string test_body;
  uint32_t id_to_fold;
  ResultType expected_result;
  std::function<uint32_t(uint32_t)> id_map;
};

using IntegerInstructionFoldingTestWithMap =
    ::testing::TestWithParam<InstructionFoldingCaseWithMap<uint32_t>>;

TEST_P(IntegerInstructionFoldingTestWithMap, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  inst = opt::FoldInstructionToConstant(inst, tc.id_map);

  // Make sure the instruction folded as expected.
  EXPECT_NE(inst, nullptr);
  if (inst != nullptr) {
    EXPECT_EQ(inst->opcode(), SpvOpConstant);
    opt::analysis::ConstantManager* const_mrg = context->get_constant_mgr();
    const opt::analysis::IntConstant* result =
        const_mrg->GetConstantFromInst(inst)->AsIntConstant();
    EXPECT_NE(result, nullptr);
    if (result != nullptr) {
      EXPECT_EQ(result->GetU32BitValue(), tc.expected_result);
    }
  }
}
// clang-format off
INSTANTIATE_TEST_CASE_P(TestCase, IntegerInstructionFoldingTestWithMap,
  ::testing::Values(
      // Test case 0: fold %3 = 0; %3 * n
      InstructionFoldingCaseWithMap<uint32_t>(
          Header() + "%main = OpFunction %void None %void_func\n" +
              "%main_lab = OpLabel\n" +
              "%n = OpVariable %_ptr_int Function\n" +
              "%load = OpLoad %int %n\n" +
              "%3 = OpCopyObject %int %int_0\n"
              "%2 = OpIMul %int %3 %load\n" +
              "OpReturn\n" +
              "OpFunctionEnd",
          2, 0, [](uint32_t id) {return (id == 3 ? INT_0_ID : id);})
  ));
// clang-format on

using BooleanInstructionFoldingTestWithMap =
    ::testing::TestWithParam<InstructionFoldingCaseWithMap<bool>>;

TEST_P(BooleanInstructionFoldingTestWithMap, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  inst = opt::FoldInstructionToConstant(inst, tc.id_map);

  // Make sure the instruction folded as expected.
  EXPECT_NE(inst, nullptr);
  if (inst != nullptr) {
    std::vector<SpvOp> bool_opcodes = {SpvOpConstantTrue, SpvOpConstantFalse};
    EXPECT_THAT(bool_opcodes, Contains(inst->opcode()));
    opt::analysis::ConstantManager* const_mrg = context->get_constant_mgr();
    const opt::analysis::BoolConstant* result =
        const_mrg->GetConstantFromInst(inst)->AsBoolConstant();
    EXPECT_NE(result, nullptr);
    if (result != nullptr) {
      EXPECT_EQ(result->value(), tc.expected_result);
    }
  }
}

// clang-format off
INSTANTIATE_TEST_CASE_P(TestCase, BooleanInstructionFoldingTestWithMap,
  ::testing::Values(
      // Test case 0: fold %3 = true; %3 || n
      InstructionFoldingCaseWithMap<bool>(
          Header() + "%main = OpFunction %void None %void_func\n" +
              "%main_lab = OpLabel\n" +
              "%n = OpVariable %_ptr_bool Function\n" +
              "%load = OpLoad %bool %n\n" +
              "%3 = OpCopyObject %bool %true\n" +
              "%2 = OpLogicalOr %bool %3 %load\n" +
              "OpReturn\n" +
              "OpFunctionEnd",
          2, true, [](uint32_t id) {return (id == 3 ? TRUE_ID : id);})
  ));
// clang-format on

using GeneralInstructionFoldingTest =
    ::testing::TestWithParam<InstructionFoldingCase<uint32_t>>;

TEST_P(GeneralInstructionFoldingTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  std::unique_ptr<ir::Instruction> original_inst(inst->Clone(context.get()));
  bool succeeded = opt::FoldInstruction(inst);

  // Make sure the instruction folded as expected.
  EXPECT_EQ(inst->result_id(), original_inst->result_id());
  EXPECT_EQ(inst->type_id(), original_inst->type_id());
  EXPECT_TRUE((!succeeded) == (tc.expected_result == 0));
  if (succeeded) {
    EXPECT_EQ(inst->opcode(), SpvOpCopyObject);
    EXPECT_EQ(inst->GetSingleWordInOperand(0), tc.expected_result);
  } else {
    EXPECT_EQ(inst->NumInOperands(), original_inst->NumInOperands());
    for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
      EXPECT_EQ(inst->GetOperand(i), original_inst->GetOperand(i));
    }
  }
}

// clang-format off
INSTANTIATE_TEST_CASE_P(IntegerArithmeticTestCases, GeneralInstructionFoldingTest,
                        ::testing::Values(
    // Test case 0: Don't fold n * m
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%m = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%load_m = OpLoad %int %m\n" +
            "%2 = OpIMul %int %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 1: Don't fold n / m (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpUDiv %uint %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 2: Don't fold n / m (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%m = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%load_m = OpLoad %int %m\n" +
            "%2 = OpSDiv %int %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 3: Don't fold n remainder m
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%m = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%load_m = OpLoad %int %m\n" +
            "%2 = OpSRem %int %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 4: Don't fold n % m (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%m = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%load_m = OpLoad %int %m\n" +
            "%2 = OpSMod %int %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 5: Don't fold n % m (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpUMod %int %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 6: Don't fold n << m
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpShiftRightLogical %int %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 7: Don't fold n >> m
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpShiftLeftLogical %int %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 8: Don't fold n | m
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpBitwiseOr %int %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 9: Don't fold n & m
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpBitwiseAnd %int %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 10: Don't fold n < m (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpULessThan %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 11: Don't fold n > m (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpUGreaterThan %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 12: Don't fold n <= m (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpULessThanEqual %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 13: Don't fold n >= m (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%m = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%load_m = OpLoad %uint %m\n" +
            "%2 = OpUGreaterThanEqual %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 14: Don't fold n < m (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%m = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%load_m = OpLoad %int %m\n" +
            "%2 = OpULessThan %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 15: Don't fold n > m (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%m = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%load_m = OpLoad %int %m\n" +
            "%2 = OpUGreaterThan %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 16: Don't fold n <= m (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%m = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%load_m = OpLoad %int %m\n" +
            "%2 = OpULessThanEqual %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 17: Don't fold n >= m (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%m = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%load_m = OpLoad %int %m\n" +
            "%2 = OpUGreaterThanEqual %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 18: Don't fold n || m
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_bool Function\n" +
            "%m = OpVariable %_ptr_bool Function\n" +
            "%load_n = OpLoad %bool %n\n" +
            "%load_m = OpLoad %bool %m\n" +
            "%2 = OpLogicalOr %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 19: Don't fold n && m
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_bool Function\n" +
            "%m = OpVariable %_ptr_bool Function\n" +
            "%load_n = OpLoad %bool %n\n" +
            "%load_m = OpLoad %bool %m\n" +
            "%2 = OpLogicalAnd %bool %load_n %load_m\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 20: Don't fold n * 3
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%2 = OpIMul %int %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 21: Don't fold n / 3 (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpUDiv %uint %load_n %uint_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 22: Don't fold n / 3 (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%2 = OpSDiv %int %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 23: Don't fold n remainder 3
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%2 = OpSRem %int %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 24: Don't fold n % 3 (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%2 = OpSMod %int %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 25: Don't fold n % 3 (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpUMod %int %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 26: Don't fold n << 3
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpShiftRightLogical %int %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 27: Don't fold n >> 3
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpShiftLeftLogical %int %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 28: Don't fold n | 3
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpBitwiseOr %int %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 29: Don't fold n & 3
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpBitwiseAnd %uint %load_n %uint_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 30: Don't fold n < 3 (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpULessThan %bool %load_n %uint_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 31: Don't fold n > 3 (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpUGreaterThan %bool %load_n %uint_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 32: Don't fold n <= 3 (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpULessThanEqual %bool %load_n %uint_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 33: Don't fold n >= 3 (unsigned)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_uint Function\n" +
            "%load_n = OpLoad %uint %n\n" +
            "%2 = OpUGreaterThanEqual %bool %load_n %uint_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 34: Don't fold n < 3 (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%2 = OpULessThan %bool %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 35: Don't fold n > 3 (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%2 = OpUGreaterThan %bool %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 36: Don't fold n <= 3 (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%2 = OpULessThanEqual %bool %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 37: Don't fold n >= 3 (signed)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%load_n = OpLoad %int %n\n" +
            "%2 = OpUGreaterThanEqual %bool %load_n %int_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 38: Don't fold 0 + 3 (long), bad length
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpIAdd %long %long_0 %long_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 39: Don't fold 0 + 3 (short), bad length
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpIAdd %short %short_0 %short_3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 40: fold 1*n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%3 = OpLoad %int %n\n" +
            "%2 = OpIMul %int %int_1 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 41: fold n*1
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%3 = OpLoad %int %n\n" +
            "%2 = OpIMul %int %3 %int_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3)
));

INSTANTIATE_TEST_CASE_P(CompositeExtractFoldingTest, GeneralInstructionFoldingTest,
::testing::Values(
    // Test case 0: fold Insert feeding extract
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%2 = OpLoad %int %n\n" +
            "%3 = OpCompositeInsert %v4int %2 %v4int_0_0_0_0 0\n" +
            "%4 = OpCompositeInsert %v4int %int_1 %3 1\n" +
            "%5 = OpCompositeInsert %v4int %int_1 %4 2\n" +
            "%6 = OpCompositeInsert %v4int %int_1 %5 3\n" +
            "%7 = OpCompositeExtract %int %6 0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        7, 2),
    // Test case 1: fold Composite construct feeding extract (position 0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%2 = OpLoad %int %n\n" +
            "%3 = OpCompositeConstruct %v4int %2 %int_0 %int_0 %int_0\n" +
            "%4 = OpCompositeExtract %int %3 0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        4, 2),
    // Test case 2: fold Composite construct feeding extract (position 3)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%2 = OpLoad %int %n\n" +
            "%3 = OpCompositeConstruct %v4int %2 %int_0 %int_0 %100\n" +
            "%4 = OpCompositeExtract %int %3 3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        4, INT_0_ID),
    // Test case 3: fold Composite construct with vectors feeding extract (scalar element)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%2 = OpLoad %int %n\n" +
            "%3 = OpCompositeConstruct %v2int %2 %int_0\n" +
            "%4 = OpCompositeConstruct %v4int %3 %int_0 %100\n" +
            "%5 = OpCompositeExtract %int %4 3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        5, INT_0_ID),
    // Test case 4: fold Composite construct with vectors feeding extract (start of vector element)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%2 = OpLoad %int %n\n" +
            "%3 = OpCompositeConstruct %v2int %2 %int_0\n" +
            "%4 = OpCompositeConstruct %v4int %3 %int_0 %100\n" +
            "%5 = OpCompositeExtract %int %4 0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        5, 2),
    // Test case 5: fold Composite construct with vectors feeding extract (middle of vector element)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%2 = OpLoad %int %n\n" +
            "%3 = OpCompositeConstruct %v2int %int_0 %2\n" +
            "%4 = OpCompositeConstruct %v4int %3 %int_0 %100\n" +
            "%5 = OpCompositeExtract %int %4 1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        5, 2),
    // Test case 6: fold Composite construct with multiple indices.
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_int Function\n" +
            "%2 = OpLoad %int %n\n" +
            "%3 = OpCompositeConstruct %v2int %int_0 %2\n" +
            "%4 = OpCompositeConstruct %struct_v2int_int_int %3 %int_0 %100\n" +
            "%5 = OpCompositeExtract %int %4 0 1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        5, 2),
    // Test case 7: fold constant extract.
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpCompositeExtract %int %102 1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, INT_7_ID),
    // Test case 8: constant struct has OpUndef
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpCompositeExtract %int %struct_undef_0_0 0 1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 9: Extracting a member of element inserted via Insert
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_struct_v2int_int_int Function\n" +
            "%2 = OpLoad %struct_v2int_int_int %n\n" +
            "%3 = OpCompositeInsert %struct_v2int_int_int %102 %2 0\n" +
            "%4 = OpCompositeExtract %int %3 0 1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        4, 103),
    // Test case 10: Extracting a element that is partially changed by Insert. (Don't fold)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_struct_v2int_int_int Function\n" +
            "%2 = OpLoad %struct_v2int_int_int %n\n" +
            "%3 = OpCompositeInsert %struct_v2int_int_int %int_0 %2 0 1\n" +
            "%4 = OpCompositeExtract %v2int %3 0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        4, 0),
    // Test case 11: Extracting from result of vector shuffle (first input)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v2int Function\n" +
            "%2 = OpLoad %v2int %n\n" +
            "%3 = OpVectorShuffle %v2int %102 %2 3 0\n" +
            "%4 = OpCompositeExtract %int %3 1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        4, INT_7_ID),
    // Test case 12: Extracting from result of vector shuffle (second input)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v2int Function\n" +
            "%2 = OpLoad %v2int %n\n" +
            "%3 = OpVectorShuffle %v2int %2 %102 2 0\n" +
            "%4 = OpCompositeExtract %int %3 0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        4, INT_7_ID)
));

INSTANTIATE_TEST_CASE_P(CompositeConstructFoldingTest, GeneralInstructionFoldingTest,
::testing::Values(
    // Test case 0: fold Extracts feeding construct
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpCopyObject %v4int %v4int_0_0_0_0\n" +
            "%3 = OpCompositeExtract %int %2 0\n" +
            "%4 = OpCompositeExtract %int %2 1\n" +
            "%5 = OpCompositeExtract %int %2 2\n" +
            "%6 = OpCompositeExtract %int %2 3\n" +
            "%7 = OpCompositeConstruct %v4int %3 %4 %5 %6\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        7, 2),
    // Test case 1: Don't fold Extracts feeding construct (Different source)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpCopyObject %v4int %v4int_0_0_0_0\n" +
            "%3 = OpCompositeExtract %int %2 0\n" +
            "%4 = OpCompositeExtract %int %2 1\n" +
            "%5 = OpCompositeExtract %int %2 2\n" +
            "%6 = OpCompositeExtract %int %v4int_0_0_0_0 3\n" +
            "%7 = OpCompositeConstruct %v4int %3 %4 %5 %6\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        7, 0),
    // Test case 2: Don't fold Extracts feeding construct (bad indices)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpCopyObject %v4int %v4int_0_0_0_0\n" +
            "%3 = OpCompositeExtract %int %2 0\n" +
            "%4 = OpCompositeExtract %int %2 0\n" +
            "%5 = OpCompositeExtract %int %2 2\n" +
            "%6 = OpCompositeExtract %int %2 3\n" +
            "%7 = OpCompositeConstruct %v4int %3 %4 %5 %6\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        7, 0),
    // Test case 3: Don't fold Extracts feeding construct (different type)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpCopyObject %struct_v2int_int_int %struct_v2int_int_int_null\n" +
            "%3 = OpCompositeExtract %v2int %2 0\n" +
            "%4 = OpCompositeExtract %int %2 1\n" +
            "%5 = OpCompositeExtract %int %2 2\n" +
            "%7 = OpCompositeConstruct %v4int %3 %4 %5\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        7, 0),
    // Test case 4: Fold construct with constants to constant.
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%2 = OpCompositeConstruct %v2int %103 %103\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, VEC2_0_ID)
));

INSTANTIATE_TEST_CASE_P(PhiFoldingTest, GeneralInstructionFoldingTest,
::testing::Values(
  // Test case 0: Fold phi with the same values for all edges.
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "            OpBranchConditional %true %l1 %l2\n" +
          "%l1 = OpLabel\n" +
          "      OpBranch %merge_lab\n" +
          "%l2 = OpLabel\n" +
          "      OpBranch %merge_lab\n" +
          "%merge_lab = OpLabel\n" +
          "%2 = OpPhi %int %100 %l1 %100 %l2\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, INT_0_ID),
  // Test case 1: Fold phi in pass through loop.
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "            OpBranch %l1\n" +
          "%l1 = OpLabel\n" +
          "%2 = OpPhi %int %100 %main_lab %2 %l1\n" +
          "      OpBranchConditional %true %l1 %merge_lab\n" +
          "%merge_lab = OpLabel\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, INT_0_ID),
  // Test case 2: Don't Fold phi because of different values.
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "            OpBranch %l1\n" +
          "%l1 = OpLabel\n" +
          "%2 = OpPhi %int %int_0 %main_lab %int_3 %l1\n" +
          "      OpBranchConditional %true %l1 %merge_lab\n" +
          "%merge_lab = OpLabel\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, 0)
));

INSTANTIATE_TEST_CASE_P(SelectFoldingTest, GeneralInstructionFoldingTest,
::testing::Values(
  // Test case 0: Fold select with the same values for both sides
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_bool Function\n" +
          "%load = OpLoad %bool %n\n" +
          "%2 = OpSelect %int %load %100 %100\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, INT_0_ID),
  // Test case 1: Fold select true to left side
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %bool %n\n" +
          "%2 = OpSelect %int %true %100 %n\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, INT_0_ID),
  // Test case 2: Fold select false to right side
  InstructionFoldingCase<uint32_t>(
      Header() + "%main = OpFunction %void None %void_func\n" +
          "%main_lab = OpLabel\n" +
          "%n = OpVariable %_ptr_int Function\n" +
          "%load = OpLoad %bool %n\n" +
          "%2 = OpSelect %int %false %n %100\n" +
          "OpReturn\n" +
          "OpFunctionEnd",
      2, INT_0_ID)
));

INSTANTIATE_TEST_CASE_P(FloatRedundantFoldingTest, GeneralInstructionFoldingTest,
                        ::testing::Values(
    // Test case 0: Don't fold n + 1.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFAdd %float %3 %float_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 1: Don't fold n - 1.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFSub %float %3 %float_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 2: Don't fold n * 2.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFMul %float %3 %float_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 3: Fold n + 0.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFAdd %float %3 %float_0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 4: Fold 0.0 + n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFAdd %float %float_0 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 5: Fold n - 0.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFSub %float %3 %float_0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 6: Fold n * 1.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFMul %float %3 %float_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 7: Fold 1.0 * n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFMul %float %float_1 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 8: Fold n / 1.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFDiv %float %3 %float_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 9: Fold n * 0.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFMul %float %3 %104\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, FLOAT_0_ID),
    // Test case 10: Fold 0.0 * n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFMul %float %104 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, FLOAT_0_ID),
    // Test case 11: Fold 0.0 / n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFDiv %float %104 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, FLOAT_0_ID),
    // Test case 12: Don't fold mix(a, b, 2.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%a = OpVariable %_ptr_float Function\n" +
            "%b = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %a\n" +
            "%4 = OpLoad %float %b\n" +
            "%2 = OpExtInst %float %1 FMix %3 %4 %float_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 13: Fold mix(a, b, 0.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%a = OpVariable %_ptr_float Function\n" +
            "%b = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %a\n" +
            "%4 = OpLoad %float %b\n" +
            "%2 = OpExtInst %float %1 FMix %3 %4 %float_0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 14: Fold mix(a, b, 1.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%a = OpVariable %_ptr_float Function\n" +
            "%b = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %a\n" +
            "%4 = OpLoad %float %b\n" +
            "%2 = OpExtInst %float %1 FMix %3 %4 %float_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 4),
    // Test case 15: Fold vector fadd with null
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%a = OpVariable %_ptr_v2float Function\n" +
            "%2 = OpLoad %v2float %a\n" +
            "%3 = OpFAdd %v2float %2 %v2float_null\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        3, 2),
    // Test case 16: Fold vector fadd with null
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%a = OpVariable %_ptr_v2float Function\n" +
            "%2 = OpLoad %v2float %a\n" +
            "%3 = OpFAdd %v2float %v2float_null %2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        3, 2),
    // Test case 15: Fold vector fsub with null
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%a = OpVariable %_ptr_v2float Function\n" +
            "%2 = OpLoad %v2float %a\n" +
            "%3 = OpFSub %v2float %2 %v2float_null\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        3, 2)
));

INSTANTIATE_TEST_CASE_P(DoubleRedundantFoldingTest, GeneralInstructionFoldingTest,
                        ::testing::Values(
    // Test case 0: Don't fold n + 1.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFAdd %double %3 %double_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 1: Don't fold n - 1.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFSub %double %3 %double_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 2: Don't fold n * 2.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFMul %double %3 %double_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 3: Fold n + 0.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFAdd %double %3 %double_0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 4: Fold 0.0 + n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFAdd %double %double_0 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 5: Fold n - 0.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFSub %double %3 %double_0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 6: Fold n * 1.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFMul %double %3 %double_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 7: Fold 1.0 * n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFMul %double %double_1 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 8: Fold n / 1.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFDiv %double %3 %double_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 9: Fold n * 0.0
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFMul %double %3 %105\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, DOUBLE_0_ID),
    // Test case 10: Fold 0.0 * n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFMul %double %105 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, DOUBLE_0_ID),
    // Test case 11: Fold 0.0 / n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFDiv %double %105 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, DOUBLE_0_ID),
    // Test case 12: Don't fold mix(a, b, 2.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%a = OpVariable %_ptr_double Function\n" +
            "%b = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %a\n" +
            "%4 = OpLoad %double %b\n" +
            "%2 = OpExtInst %double %1 FMix %3 %4 %double_2\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 13: Fold mix(a, b, 0.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%a = OpVariable %_ptr_double Function\n" +
            "%b = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %a\n" +
            "%4 = OpLoad %double %b\n" +
            "%2 = OpExtInst %double %1 FMix %3 %4 %double_0\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
    // Test case 14: Fold mix(a, b, 1.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%a = OpVariable %_ptr_double Function\n" +
            "%b = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %a\n" +
            "%4 = OpLoad %double %b\n" +
            "%2 = OpExtInst %double %1 FMix %3 %4 %double_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 4)
));

INSTANTIATE_TEST_CASE_P(FloatVectorRedundantFoldingTest, GeneralInstructionFoldingTest,
                        ::testing::Values(
    // Test case 0: Don't fold a * vec4(0.0, 0.0, 0.0, 1.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4float Function\n" +
            "%3 = OpLoad %v4float %n\n" +
            "%2 = OpFMul %v4float %3 %v4float_0_0_0_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 1: Fold a * vec4(0.0, 0.0, 0.0, 0.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4float Function\n" +
            "%3 = OpLoad %v4float %n\n" +
            "%2 = OpFMul %v4float %3 %106\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, VEC4_0_ID),
    // Test case 2: Fold a * vec4(1.0, 1.0, 1.0, 1.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4float Function\n" +
            "%3 = OpLoad %v4float %n\n" +
            "%2 = OpFMul %v4float %3 %v4float_1_1_1_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3)
));

INSTANTIATE_TEST_CASE_P(DoubleVectorRedundantFoldingTest, GeneralInstructionFoldingTest,
                        ::testing::Values(
    // Test case 0: Don't fold a * vec4(0.0, 0.0, 0.0, 1.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4double Function\n" +
            "%3 = OpLoad %v4double %n\n" +
            "%2 = OpFMul %v4double %3 %v4double_0_0_0_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 1: Fold a * vec4(0.0, 0.0, 0.0, 0.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4double Function\n" +
            "%3 = OpLoad %v4double %n\n" +
            "%2 = OpFMul %v4double %3 %106\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, DVEC4_0_ID),
    // Test case 2: Fold a * vec4(1.0, 1.0, 1.0, 1.0)
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4double Function\n" +
            "%3 = OpLoad %v4double %n\n" +
            "%2 = OpFMul %v4double %3 %v4double_1_1_1_1\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3)
));
// clang-format on

using ToNegateFoldingTest =
    ::testing::TestWithParam<InstructionFoldingCase<uint32_t>>;

TEST_P(ToNegateFoldingTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  std::unique_ptr<ir::Instruction> original_inst(inst->Clone(context.get()));
  bool succeeded = opt::FoldInstruction(inst);

  // Make sure the instruction folded as expected.
  EXPECT_EQ(inst->result_id(), original_inst->result_id());
  EXPECT_EQ(inst->type_id(), original_inst->type_id());
  EXPECT_TRUE((!succeeded) == (tc.expected_result == 0));
  if (succeeded) {
    EXPECT_EQ(inst->opcode(), SpvOpFNegate);
    EXPECT_EQ(inst->GetSingleWordInOperand(0), tc.expected_result);
  } else {
    EXPECT_EQ(inst->NumInOperands(), original_inst->NumInOperands());
    for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
      EXPECT_EQ(inst->GetOperand(i), original_inst->GetOperand(i));
    }
  }
}

// clang-format off
INSTANTIATE_TEST_CASE_P(FloatRedundantSubFoldingTest, ToNegateFoldingTest,
                        ::testing::Values(
    // Test case 0: Don't fold 1.0 - n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFSub %float %float_1 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 1: Fold 0.0 - n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_float Function\n" +
            "%3 = OpLoad %float %n\n" +
            "%2 = OpFSub %float %float_0 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
	// Test case 2: Don't fold (0,0,0,1) - n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4float Function\n" +
            "%3 = OpLoad %v4float %n\n" +
            "%2 = OpFSub %v4float %v4float_0_0_0_1 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
	// Test case 3: Fold (0,0,0,0) - n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4float Function\n" +
            "%3 = OpLoad %v4float %n\n" +
            "%2 = OpFSub %v4float %v4float_0_0_0_0 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3)
));

INSTANTIATE_TEST_CASE_P(DoubleRedundantSubFoldingTest, ToNegateFoldingTest,
                        ::testing::Values(
    // Test case 0: Don't fold 1.0 - n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFSub %double %double_1 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
    // Test case 1: Fold 0.0 - n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_double Function\n" +
            "%3 = OpLoad %double %n\n" +
            "%2 = OpFSub %double %double_0 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3),
	// Test case 2: Don't fold (0,0,0,1) - n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4double Function\n" +
            "%3 = OpLoad %v4double %n\n" +
            "%2 = OpFSub %v4double %v4double_0_0_0_1 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 0),
	// Test case 3: Fold (0,0,0,0) - n
    InstructionFoldingCase<uint32_t>(
        Header() + "%main = OpFunction %void None %void_func\n" +
            "%main_lab = OpLabel\n" +
            "%n = OpVariable %_ptr_v4double Function\n" +
            "%3 = OpLoad %v4double %n\n" +
            "%2 = OpFSub %v4double %v4double_0_0_0_0 %3\n" +
            "OpReturn\n" +
            "OpFunctionEnd",
        2, 3)
));

#ifdef SPIRV_EFFCEE
using MatchingInstructionFoldingTest =
    ::testing::TestWithParam<InstructionFoldingCase<bool>>;

TEST_P(MatchingInstructionFoldingTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.test_body,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Fold the instruction to test.
  opt::analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();
  ir::Instruction* inst = def_use_mgr->GetDef(tc.id_to_fold);
  std::unique_ptr<ir::Instruction> original_inst(inst->Clone(context.get()));
  bool succeeded = opt::FoldInstruction(inst);
  EXPECT_EQ(succeeded, tc.expected_result);
  if (succeeded) {
    Match(tc.test_body, context.get());
  }
}

INSTANTIATE_TEST_CASE_P(MergeNegateTest, MatchingInstructionFoldingTest,
::testing::Values(
  // Test case 0: fold consecutive fnegate
  // -(-x) = x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float:%\\w+]]\n" +
      "; CHECK: %4 = OpCopyObject [[float]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFNegate %float %2\n" +
      "%4 = OpFNegate %float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 1: fold fnegate(fmul with const).
  // -(x * 2.0) = x * -2.0
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_n2:%\\w+]] = OpConstant [[float]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFMul [[float]] [[ld]] [[float_n2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFMul %float %2 %float_2\n" +
      "%4 = OpFNegate %float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 2: fold fnegate(fmul with const).
  // -(2.0 * x) = x * 2.0
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_n2:%\\w+]] = OpConstant [[float]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFMul [[float]] [[ld]] [[float_n2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFMul %float %float_2 %2\n" +
      "%4 = OpFNegate %float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 3: fold fnegate(fdiv with const).
  // -(x / 2.0) = x * -0.5
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_n0p5:%\\w+]] = OpConstant [[float]] -0.5\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFMul [[float]] [[ld]] [[float_n0p5]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %2 %float_2\n" +
      "%4 = OpFNegate %float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 4: fold fnegate(fdiv with const).
  // -(2.0 / x) = -2.0 / x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_n2:%\\w+]] = OpConstant [[float]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFDiv [[float]] [[float_n2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %float_2 %2\n" +
      "%4 = OpFNegate %float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 5: fold fnegate(fadd with const).
  // -(2.0 + x) = -2.0 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_n2:%\\w+]] = OpConstant [[float]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_n2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %float_2 %2\n" +
      "%4 = OpFNegate %float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 6: fold fnegate(fadd with const).
  // -(x + 2.0) = -2.0 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_n2:%\\w+]] = OpConstant [[float]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_n2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %2 %float_2\n" +
      "%4 = OpFNegate %float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 7: fold fnegate(fsub with const).
  // -(2.0 - x) = x - 2.0
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[ld]] [[float_2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %float_2 %2\n" +
      "%4 = OpFNegate %float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 8: fold fnegate(fsub with const).
  // -(x - 2.0) = 2.0 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %2 %float_2\n" +
      "%4 = OpFNegate %float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 9: fold consecutive snegate
  // -(-x) = x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[int:%\\w+]]\n" +
      "; CHECK: %4 = OpCopyObject [[int]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpSNegate %int %2\n" +
      "%4 = OpSNegate %int %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 10: fold consecutive vector negate
  // -(-x) = x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[v2float:%\\w+]]\n" +
      "; CHECK: %4 = OpCopyObject [[v2float]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_v2float Function\n" +
      "%2 = OpLoad %v2float %var\n" +
      "%3 = OpFNegate %v2float %2\n" +
      "%4 = OpFNegate %v2float %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 11: fold snegate(iadd with const).
  // -(2 + x) = -2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: OpConstant [[int]] -2147483648\n" +
      "; CHECK: [[int_n2:%\\w+]] = OpConstant [[int]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[int]]\n" +
      "; CHECK: %4 = OpISub [[int]] [[int_n2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpIAdd %int %int_2 %2\n" +
      "%4 = OpSNegate %int %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 12: fold snegate(iadd with const).
  // -(x + 2) = -2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: OpConstant [[int]] -2147483648\n" +
      "; CHECK: [[int_n2:%\\w+]] = OpConstant [[int]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[int]]\n" +
      "; CHECK: %4 = OpISub [[int]] [[int_n2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpIAdd %int %2 %int_2\n" +
      "%4 = OpSNegate %int %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 13: fold snegate(isub with const).
  // -(2 - x) = x - 2
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: [[int_2:%\\w+]] = OpConstant [[int]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[int]]\n" +
      "; CHECK: %4 = OpISub [[int]] [[ld]] [[int_2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpISub %int %int_2 %2\n" +
      "%4 = OpSNegate %int %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 14: fold snegate(isub with const).
  // -(x - 2) = 2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: [[int_2:%\\w+]] = OpConstant [[int]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[int]]\n" +
      "; CHECK: %4 = OpISub [[int]] [[int_2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpISub %int %2 %int_2\n" +
      "%4 = OpSNegate %int %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 15: fold snegate(iadd with const).
  // -(x + 2) = -2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[long:%\\w+]] = OpTypeInt 64 1\n" +
      "; CHECK: [[long_n2:%\\w+]] = OpConstant [[long]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[long]]\n" +
      "; CHECK: %4 = OpISub [[long]] [[long_n2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_long Function\n" +
      "%2 = OpLoad %long %var\n" +
      "%3 = OpIAdd %long %2 %long_2\n" +
      "%4 = OpSNegate %long %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 16: fold snegate(isub with const).
  // -(2 - x) = x - 2
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[long:%\\w+]] = OpTypeInt 64 1\n" +
      "; CHECK: [[long_2:%\\w+]] = OpConstant [[long]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[long]]\n" +
      "; CHECK: %4 = OpISub [[long]] [[ld]] [[long_2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_long Function\n" +
      "%2 = OpLoad %long %var\n" +
      "%3 = OpISub %long %long_2 %2\n" +
      "%4 = OpSNegate %long %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true),
  // Test case 17: fold snegate(isub with const).
  // -(x - 2) = 2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[long:%\\w+]] = OpTypeInt 64 1\n" +
      "; CHECK: [[long_2:%\\w+]] = OpConstant [[long]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[long]]\n" +
      "; CHECK: %4 = OpISub [[long]] [[long_2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_long Function\n" +
      "%2 = OpLoad %long %var\n" +
      "%3 = OpISub %long %2 %long_2\n" +
      "%4 = OpSNegate %long %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd",
    4, true)
));

INSTANTIATE_TEST_CASE_P(ReciprocalFDivTest, MatchingInstructionFoldingTest,
::testing::Values(
  // Test case 0: scalar reicprocal
  // x / 0.5 = x * 2.0
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %3 = OpFMul [[float]] [[ld]] [[float_2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %2 %float_0p5\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    3, true),
  // Test case 1: Unfoldable
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_0:%\\w+]] = OpConstant [[float]] 0\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %3 = OpFDiv [[float]] [[ld]] [[float_0]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %2 %104\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    3, false),
  // Test case 2: Vector reciprocal
  // x / {2.0, 0.5} = x * {0.5, 2.0}
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[v2float:%\\w+]] = OpTypeVector [[float]] 2\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[float_0p5:%\\w+]] = OpConstant [[float]] 0.5\n" +
      "; CHECK: [[v2float_0p5_2:%\\w+]] = OpConstantComposite [[v2float]] [[float_0p5]] [[float_2]]\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[v2float]]\n" +
      "; CHECK: %3 = OpFMul [[v2float]] [[ld]] [[v2float_0p5_2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_v2float Function\n" +
      "%2 = OpLoad %v2float %var\n" +
      "%3 = OpFDiv %v2float %2 %v2float_2_0p5\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    3, true),
  // Test case 3: double reciprocal
  // x / 2.0 = x * 0.5
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[double:%\\w+]] = OpTypeFloat 64\n" +
      "; CHECK: [[double_0p5:%\\w+]] = OpConstant [[double]] 0.5\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[double]]\n" +
      "; CHECK: %3 = OpFMul [[double]] [[ld]] [[double_0p5]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_double Function\n" +
      "%2 = OpLoad %double %var\n" +
      "%3 = OpFDiv %double %2 %double_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    3, true),
  // Test case 4: don't fold x / 0.
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_v2float Function\n" +
      "%2 = OpLoad %v2float %var\n" +
      "%3 = OpFDiv %v2float %2 %v2float_null\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    3, false)
));

INSTANTIATE_TEST_CASE_P(MergeMulTest, MatchingInstructionFoldingTest,
::testing::Values(
  // Test case 0: fold consecutive fmuls
  // (x * 3.0) * 2.0 = x * 6.0
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_6:%\\w+]] = OpConstant [[float]] 6\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFMul [[float]] [[ld]] [[float_6]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFMul %float %2 %float_3\n" +
      "%4 = OpFMul %float %3 %float_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 1: fold consecutive fmuls
  // 2.0 * (x * 3.0) = x * 6.0
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_6:%\\w+]] = OpConstant [[float]] 6\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFMul [[float]] [[ld]] [[float_6]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFMul %float %2 %float_3\n" +
      "%4 = OpFMul %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 2: fold consecutive fmuls
  // (3.0 * x) * 2.0 = x * 6.0
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_6:%\\w+]] = OpConstant [[float]] 6\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFMul [[float]] [[ld]] [[float_6]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFMul %float %float_3 %2\n" +
      "%4 = OpFMul %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 3: fold vector fmul
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[v2float:%\\w+]] = OpTypeVector [[float]] 2\n" +
      "; CHECK: [[float_6:%\\w+]] = OpConstant [[float]] 6\n" +
      "; CHECK: [[v2float_6_6:%\\w+]] = OpConstantComposite [[v2float]] [[float_6]] [[float_6]]\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[v2float]]\n" +
      "; CHECK: %4 = OpFMul [[v2float]] [[ld]] [[v2float_6_6]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_v2float Function\n" +
      "%2 = OpLoad %v2float %var\n" +
      "%3 = OpFMul %v2float %2 %v2float_2_3\n" +
      "%4 = OpFMul %v2float %3 %v2float_3_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 4: fold double fmuls
  // (x * 3.0) * 2.0 = x * 6.0
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[double:%\\w+]] = OpTypeFloat 64\n" +
      "; CHECK: [[double_6:%\\w+]] = OpConstant [[double]] 6\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[double]]\n" +
      "; CHECK: %4 = OpFMul [[double]] [[ld]] [[double_6]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_double Function\n" +
      "%2 = OpLoad %double %var\n" +
      "%3 = OpFMul %double %2 %double_3\n" +
      "%4 = OpFMul %double %3 %double_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 5: fold 32 bit imuls
  // (x * 3) * 2 = x * 6
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: [[int_6:%\\w+]] = OpConstant [[int]] 6\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[int]]\n" +
      "; CHECK: %4 = OpIMul [[int]] [[ld]] [[int_6]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpIMul %int %2 %int_3\n" +
      "%4 = OpIMul %int %3 %int_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 6: fold 64 bit imuls
  // (x * 3) * 2 = x * 6
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[long:%\\w+]] = OpTypeInt 64\n" +
      "; CHECK: [[long_6:%\\w+]] = OpConstant [[long]] 6\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[long]]\n" +
      "; CHECK: %4 = OpIMul [[long]] [[ld]] [[long_6]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_long Function\n" +
      "%2 = OpLoad %long %var\n" +
      "%3 = OpIMul %long %2 %long_3\n" +
      "%4 = OpIMul %long %3 %long_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 7: merge vector integer mults
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: [[v2int:%\\w+]] = OpTypeVector [[int]] 2\n" +
      "; CHECK: [[int_6:%\\w+]] = OpConstant [[int]] 6\n" +
      "; CHECK: [[v2int_6_6:%\\w+]] = OpConstantComposite [[v2int]] [[int_6]] [[int_6]]\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[v2int]]\n" +
      "; CHECK: %4 = OpIMul [[v2int]] [[ld]] [[v2int_6_6]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_v2int Function\n" +
      "%2 = OpLoad %v2int %var\n" +
      "%3 = OpIMul %v2int %2 %v2int_2_3\n" +
      "%4 = OpIMul %v2int %3 %v2int_3_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 8: merge fmul of fdiv
  // 2.0 * (2.0 / x) = 4.0 / x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_4:%\\w+]] = OpConstant [[float]] 4\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFDiv [[float]] [[float_4]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %float_2 %2\n" +
      "%4 = OpFMul %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 9: merge fmul of fdiv
  // (2.0 / x) * 2.0 = 4.0 / x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_4:%\\w+]] = OpConstant [[float]] 4\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFDiv [[float]] [[float_4]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %float_2 %2\n" +
      "%4 = OpFMul %float %3 %float_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 10: Do not merge imul of sdiv
  // 4 * (x / 2)
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpSDiv %int %2 %int_2\n" +
      "%4 = OpIMul %int %int_4 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 11: Do not merge imul of sdiv
  // (x / 2) * 4
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpSDiv %int %2 %int_2\n" +
      "%4 = OpIMul %int %3 %int_4\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 12: Do not merge imul of udiv
  // 4 * (x / 2)
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_uint Function\n" +
      "%2 = OpLoad %uint %var\n" +
      "%3 = OpUDiv %uint %2 %uint_2\n" +
      "%4 = OpIMul %uint %uint_4 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 13: Do not merge imul of udiv
  // (x / 2) * 4
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_uint Function\n" +
      "%2 = OpLoad %uint %var\n" +
      "%3 = OpUDiv %uint %2 %uint_2\n" +
      "%4 = OpIMul %uint %3 %uint_4\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 14: Don't fold
  // (x / 3) * 4
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_uint Function\n" +
      "%2 = OpLoad %uint %var\n" +
      "%3 = OpUDiv %uint %2 %uint_3\n" +
      "%4 = OpIMul %uint %3 %uint_4\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 15: merge vector fmul of fdiv
  // (x / {2,2}) * {4,4} = x * {2,2}
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[v2float:%\\w+]] = OpTypeVector [[float]] 2\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[v2float_2_2:%\\w+]] = OpConstantComposite [[v2float]] [[float_2]] [[float_2]]\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[v2float]]\n" +
      "; CHECK: %4 = OpFMul [[v2float]] [[ld]] [[v2float_2_2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_v2float Function\n" +
      "%2 = OpLoad %v2float %var\n" +
      "%3 = OpFDiv %v2float %2 %v2float_2_2\n" +
      "%4 = OpFMul %v2float %3 %v2float_4_4\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 16: merge vector imul of snegate
  // (-x) * {2,2} = x * {-2,-2}
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: [[v2int:%\\w+]] = OpTypeVector [[int]] 2\n" +
      "; CHECK: OpConstant [[int]] -2147483648\n" +
      "; CHECK: [[int_n2:%\\w+]] = OpConstant [[int]] -2\n" +
      "; CHECK: [[v2int_n2_n2:%\\w+]] = OpConstantComposite [[v2int]] [[int_n2]] [[int_n2]]\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[v2int]]\n" +
      "; CHECK: %4 = OpIMul [[v2int]] [[ld]] [[v2int_n2_n2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_v2int Function\n" +
      "%2 = OpLoad %v2int %var\n" +
      "%3 = OpSNegate %v2int %2\n" +
      "%4 = OpIMul %v2int %3 %v2int_2_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 17: merge vector imul of snegate
  // {2,2} * (-x) = x * {-2,-2}
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: [[v2int:%\\w+]] = OpTypeVector [[int]] 2\n" +
      "; CHECK: OpConstant [[int]] -2147483648\n" +
      "; CHECK: [[int_n2:%\\w+]] = OpConstant [[int]] -2\n" +
      "; CHECK: [[v2int_n2_n2:%\\w+]] = OpConstantComposite [[v2int]] [[int_n2]] [[int_n2]]\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[v2int]]\n" +
      "; CHECK: %4 = OpIMul [[v2int]] [[ld]] [[v2int_n2_n2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_v2int Function\n" +
      "%2 = OpLoad %v2int %var\n" +
      "%3 = OpSNegate %v2int %2\n" +
      "%4 = OpIMul %v2int %v2int_2_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true)
));

INSTANTIATE_TEST_CASE_P(MergeDivTest, MatchingInstructionFoldingTest,
::testing::Values(
  // Test case 0: merge consecutive fdiv
  // 4.0 / (2.0 / x) = 2.0 * x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFMul [[float]] [[float_2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %float_2 %2\n" +
      "%4 = OpFDiv %float %float_4 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 1: merge consecutive fdiv
  // 4.0 / (x / 2.0) = 8.0 / x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_8:%\\w+]] = OpConstant [[float]] 8\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFDiv [[float]] [[float_8]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %2 %float_2\n" +
      "%4 = OpFDiv %float %float_4 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 2: merge consecutive fdiv
  // (4.0 / x) / 2.0 = 2.0 / x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFDiv [[float]] [[float_2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %float_4 %2\n" +
      "%4 = OpFDiv %float %3 %float_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 3: Do not merge consecutive sdiv
  // 4 / (2 / x)
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpSDiv %int %int_2 %2\n" +
      "%4 = OpSDiv %int %int_4 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 4: Do not merge consecutive sdiv
  // 4 / (x / 2)
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpSDiv %int %2 %int_2\n" +
      "%4 = OpSDiv %int %int_4 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 5: Do not merge consecutive sdiv
  // (4 / x) / 2
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpSDiv %int %int_4 %2\n" +
      "%4 = OpSDiv %int %3 %int_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 6: Do not merge consecutive sdiv
  // (x / 4) / 2
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpSDiv %int %2 %int_4\n" +
      "%4 = OpSDiv %int %3 %int_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 7: Do not merge sdiv of imul
  // 4 / (2 * x)
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpIMul %int %int_2 %2\n" +
      "%4 = OpSDiv %int %int_4 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 8: Do not merge sdiv of imul
  // 4 / (x * 2)
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpIMul %int %2 %int_2\n" +
      "%4 = OpSDiv %int %int_4 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 9: Do not merge sdiv of imul
  // (4 * x) / 2
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpIMul %int %int_4 %2\n" +
      "%4 = OpSDiv %int %3 %int_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 10: Do not merge sdiv of imul
  // (x * 4) / 2
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpIMul %int %2 %int_4\n" +
      "%4 = OpSDiv %int %3 %int_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false),
  // Test case 11: merge sdiv of snegate
  // (-x) / 2 = x / -2
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: OpConstant [[int]] -2147483648\n" +
      "; CHECK: [[int_n2:%\\w+]] = OpConstant [[int]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[int]]\n" +
      "; CHECK: %4 = OpSDiv [[int]] [[ld]] [[int_n2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpSNegate %int %2\n" +
      "%4 = OpSDiv %int %3 %int_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 12: merge sdiv of snegate
  // 2 / (-x) = -2 / x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[int:%\\w+]] = OpTypeInt 32 1\n" +
      "; CHECK: OpConstant [[int]] -2147483648\n" +
      "; CHECK: [[int_n2:%\\w+]] = OpConstant [[int]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[int]]\n" +
      "; CHECK: %4 = OpSDiv [[int]] [[int_n2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_int Function\n" +
      "%2 = OpLoad %int %var\n" +
      "%3 = OpSNegate %int %2\n" +
      "%4 = OpSDiv %int %int_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 13: Don't merge
  // (x / {null}) / {null}
  InstructionFoldingCase<bool>(
    Header() +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_v2float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFDiv %float %2 %v2float_null\n" +
      "%4 = OpFDiv %float %3 %v2float_null\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, false)
));

INSTANTIATE_TEST_CASE_P(MergeAddTest, MatchingInstructionFoldingTest,
::testing::Values(
  // Test case 0: merge add of negate
  // (-x) + 2 = 2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFNegate %float %2\n" +
      "%4 = OpFAdd %float %3 %float_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 1: merge add of negate
  // 2 + (-x) = 2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpSNegate %float %2\n" +
      "%4 = OpIAdd %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 2: merge add of negate
  // (-x) + 2 = 2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[long:%\\w+]] = OpTypeInt 64 1\n" +
      "; CHECK: [[long_2:%\\w+]] = OpConstant [[long]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[long]]\n" +
      "; CHECK: %4 = OpISub [[long]] [[long_2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_long Function\n" +
      "%2 = OpLoad %long %var\n" +
      "%3 = OpSNegate %long %2\n" +
      "%4 = OpIAdd %long %3 %long_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 3: merge add of negate
  // 2 + (-x) = 2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[long:%\\w+]] = OpTypeInt 64 1\n" +
      "; CHECK: [[long_2:%\\w+]] = OpConstant [[long]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[long]]\n" +
      "; CHECK: %4 = OpISub [[long]] [[long_2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_long Function\n" +
      "%2 = OpLoad %long %var\n" +
      "%3 = OpSNegate %long %2\n" +
      "%4 = OpIAdd %long %long_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 4: merge add of subtract
  // (x - 1) + 2 = x + 1
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_1:%\\w+]] = OpConstant [[float]] 1\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_1]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %2 %float_1\n" +
      "%4 = OpFAdd %float %3 %float_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 5: merge add of subtract
  // (1 - x) + 2 = 3 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_3:%\\w+]] = OpConstant [[float]] 3\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_3]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %float_1 %2\n" +
      "%4 = OpFAdd %float %3 %float_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 6: merge add of subtract
  // 2 + (x - 1) = x + 1
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_1:%\\w+]] = OpConstant [[float]] 1\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_1]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %2 %float_1\n" +
      "%4 = OpFAdd %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 7: merge add of subtract
  // 2 + (1 - x) = 3 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_3:%\\w+]] = OpConstant [[float]] 3\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_3]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %float_1 %2\n" +
      "%4 = OpFAdd %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 8: merge add of add
  // (x + 1) + 2 = x + 3
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_3:%\\w+]] = OpConstant [[float]] 3\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_3]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %2 %float_1\n" +
      "%4 = OpFAdd %float %3 %float_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 9: merge add of add
  // (1 + x) + 2 = 3 + x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_3:%\\w+]] = OpConstant [[float]] 3\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_3]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %float_1 %2\n" +
      "%4 = OpFAdd %float %3 %float_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 10: merge add of add
  // 2 + (x + 1) = x + 1
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_3:%\\w+]] = OpConstant [[float]] 3\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_3]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %2 %float_1\n" +
      "%4 = OpFAdd %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 11: merge add of add
  // 2 + (1 + x) = 3 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_3:%\\w+]] = OpConstant [[float]] 3\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_3]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %float_1 %2\n" +
      "%4 = OpFAdd %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true)
));

INSTANTIATE_TEST_CASE_P(MergeSubTest, MatchingInstructionFoldingTest,
::testing::Values(
  // Test case 0: merge sub of negate
  // (-x) - 2 = -2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_n2:%\\w+]] = OpConstant [[float]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_n2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFNegate %float %2\n" +
      "%4 = OpFSub %float %3 %float_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 1: merge sub of negate
  // 2 - (-x) = x + 2
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_2:%\\w+]] = OpConstant [[float]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFNegate %float %2\n" +
      "%4 = OpFSub %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 2: merge sub of negate
  // (-x) - 2 = -2 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[long:%\\w+]] = OpTypeInt 64 1\n" +
      "; CHECK: [[long_n2:%\\w+]] = OpConstant [[long]] -2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[long]]\n" +
      "; CHECK: %4 = OpISub [[long]] [[long_n2]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_long Function\n" +
      "%2 = OpLoad %long %var\n" +
      "%3 = OpSNegate %long %2\n" +
      "%4 = OpISub %long %3 %long_2\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 3: merge sub of negate
  // 2 - (-x) = x + 2
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[long:%\\w+]] = OpTypeInt 64 1\n" +
      "; CHECK: [[long_2:%\\w+]] = OpConstant [[long]] 2\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[long]]\n" +
      "; CHECK: %4 = OpIAdd [[long]] [[ld]] [[long_2]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_long Function\n" +
      "%2 = OpLoad %long %var\n" +
      "%3 = OpSNegate %long %2\n" +
      "%4 = OpISub %long %long_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 4: merge add of subtract
  // (x + 2) - 1 = x + 1
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_1:%\\w+]] = OpConstant [[float]] 1\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_1]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %2 %float_2\n" +
      "%4 = OpFSub %float %3 %float_1\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 5: merge add of subtract
  // (2 + x) - 1 = x + 1
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_1:%\\w+]] = OpConstant [[float]] 1\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_1]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %float_2 %2\n" +
      "%4 = OpFSub %float %3 %float_1\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 6: merge add of subtract
  // 2 - (x + 1) = 1 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_1:%\\w+]] = OpConstant [[float]] 1\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_1]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %2 %float_1\n" +
      "%4 = OpFSub %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 7: merge add of subtract
  // 2 - (1 + x) = 1 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_1:%\\w+]] = OpConstant [[float]] 1\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_1]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFAdd %float %float_1 %2\n" +
      "%4 = OpFSub %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 8: merge subtract of subtract
  // (x - 2) - 1 = x - 3
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_3:%\\w+]] = OpConstant [[float]] 3\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[ld]] [[float_3]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %2 %float_2\n" +
      "%4 = OpFSub %float %3 %float_1\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 9: merge subtract of subtract
  // (2 - x) - 1 = 1 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_1:%\\w+]] = OpConstant [[float]] 1\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_1]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %float_2 %2\n" +
      "%4 = OpFSub %float %3 %float_1\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 10: merge subtract of subtract
  // 2 - (x - 1) = 3 - x
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_3:%\\w+]] = OpConstant [[float]] 3\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFSub [[float]] [[float_3]] [[ld]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %2 %float_1\n" +
      "%4 = OpFSub %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 11: merge subtract of subtract
  // 1 - (2 - x) = x + (-1)
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_n1:%\\w+]] = OpConstant [[float]] -1\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_n1]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %float_2 %2\n" +
      "%4 = OpFSub %float %float_1 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true),
  // Test case 12: merge subtract of subtract
  // 2 - (1 - x) = x + 1
  InstructionFoldingCase<bool>(
    Header() +
      "; CHECK: [[float:%\\w+]] = OpTypeFloat 32\n" +
      "; CHECK: [[float_1:%\\w+]] = OpConstant [[float]] 1\n" +
      "; CHECK: [[ld:%\\w+]] = OpLoad [[float]]\n" +
      "; CHECK: %4 = OpFAdd [[float]] [[ld]] [[float_1]]\n" +
      "%main = OpFunction %void None %void_func\n" +
      "%main_lab = OpLabel\n" +
      "%var = OpVariable %_ptr_float Function\n" +
      "%2 = OpLoad %float %var\n" +
      "%3 = OpFSub %float %float_1 %2\n" +
      "%4 = OpFSub %float %float_2 %3\n" +
      "OpReturn\n" +
      "OpFunctionEnd\n",
    4, true)
));
#endif
}  // anonymous namespace
