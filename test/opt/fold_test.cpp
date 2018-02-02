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
%true = OpConstantTrue %bool
%101 = OpConstantTrue %bool ; Need a def with an numerical id to define id maps.
%false = OpConstantFalse %bool
%short = OpTypeInt 16 1
%int = OpTypeInt 32 1
%long = OpTypeInt 64 1
%uint = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%v4int = OpTypeVector %int 4
%struct_v2int_int_int = OpTypeStruct %v2int %int %int
%_ptr_int = OpTypePointer Function %int
%_ptr_uint = OpTypePointer Function %uint
%_ptr_bool = OpTypePointer Function %bool
%short_0 = OpConstant %short 0
%short_3 = OpConstant %short 3
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%100 = OpConstant %int 0 ; Need a def with an numerical id to define id maps.
%int_3 = OpConstant %int 3
%int_min = OpConstant %int -2147483648
%int_max = OpConstant %int 2147483647
%long_0 = OpConstant %long 0
%long_3 = OpConstant %long 3
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_32 = OpConstant %uint 32
%uint_max = OpConstant %uint -1
%struct_v2int_int_int_null = OpConstantNull %struct_v2int_int_int
%v4int_0_0_0_0 = OpConstantComposite %v4int %int_0 %int_0 %int_0 %int_0
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
        5, 2)
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
        7, 0)
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
// clang-format off
}  // anonymous namespace
