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

#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "../assembly_builder.h"
#include "../pass_fixture.h"
#include "../pass_utils.h"
#include "opt/dominator_analysis_pass.h"
#include "opt/pass.h"

namespace {

using namespace spvtools;
using ::testing::UnorderedElementsAre;

using PassClassTest = PassTest<::testing::Test>;

const ir::Function* getFunction(const ir::Module& module, uint32_t id) {
  for (const ir::Function& F : module) {
    if (F.result_id() == id) {
      return &F;
    }
  }
  return nullptr;
}

const ir::BasicBlock* getBasicBlock(const ir::Function* fn, uint32_t id) {
  for (const ir::BasicBlock& BB : *fn) {
    if (BB.id() == id) {
      return &BB;
    }
  }
  return nullptr;
}

// Check that X dominates Y, and
//   if X != Y then
//      X strictly dominates Y and
//      Y does not dominate X and
//      Y does not strictly dominate X
//   if X == X then
//      X does not strictly dominate itself
void check_dominance(const opt::DominatorAnalysisBase& DomTree,
                     const ir::Function* TestFn, uint32_t X, uint32_t Y) {
  SCOPED_TRACE("Check dominance properties for Basic Block " +
               std::to_string(X) + " and " + std::to_string(Y));
  EXPECT_TRUE(
      DomTree.Dominates(getBasicBlock(TestFn, X), getBasicBlock(TestFn, Y)));
  EXPECT_TRUE(DomTree.Dominates(X, Y));
  if (X == Y) {
    EXPECT_FALSE(DomTree.StrictlyDominates(X, X));
  } else {
    EXPECT_TRUE(DomTree.StrictlyDominates(X, Y));
    EXPECT_FALSE(DomTree.Dominates(Y, X));
    EXPECT_FALSE(DomTree.StrictlyDominates(Y, X));
  }
}

// Check that X does not dominates Y and vise versa
void check_no_dominance(const opt::DominatorAnalysisBase& DomTree,
                        const ir::Function* TestFn, uint32_t X, uint32_t Y) {
  SCOPED_TRACE("Check no domination for Basic Block " + std::to_string(X) +
               " and " + std::to_string(Y));
  EXPECT_FALSE(
      DomTree.Dominates(getBasicBlock(TestFn, X), getBasicBlock(TestFn, Y)));
  EXPECT_FALSE(DomTree.Dominates(X, Y));
  EXPECT_FALSE(DomTree.StrictlyDominates(getBasicBlock(TestFn, X),
                                         getBasicBlock(TestFn, Y)));
  EXPECT_FALSE(DomTree.StrictlyDominates(X, Y));

  EXPECT_FALSE(
      DomTree.Dominates(getBasicBlock(TestFn, Y), getBasicBlock(TestFn, X)));
  EXPECT_FALSE(DomTree.Dominates(Y, X));
  EXPECT_FALSE(DomTree.StrictlyDominates(getBasicBlock(TestFn, Y),
                                         getBasicBlock(TestFn, X)));
  EXPECT_FALSE(DomTree.StrictlyDominates(Y, X));
}

TEST_F(PassClassTest, DominatorSimpleCFG) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstant %5 0
          %7 = OpConstantFalse %4
          %8 = OpConstantTrue %4
          %9 = OpConstant %5 1
          %1 = OpFunction %2 None %3
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpSwitch %6 %12 1 %13
         %12 = OpLabel
               OpBranch %14
         %13 = OpLabel
               OpBranch %14
         %14 = OpLabel
               OpBranchConditional %8 %11 %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const ir::Function* TestFn = getFunction(*module, 1);
  const ir::BasicBlock* Entry = getBasicBlock(TestFn, 10);
  EXPECT_EQ(Entry, TestFn->entry().get())
      << "The entry node is not the expected one";

  // Test normal dominator tree
  {
    opt::DominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, Entry);

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12, 13, 14, 15})
      check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 10, 11);
    check_dominance(DomTree, TestFn, 10, 12);
    check_dominance(DomTree, TestFn, 10, 13);
    check_dominance(DomTree, TestFn, 10, 14);
    check_dominance(DomTree, TestFn, 10, 15);

    check_dominance(DomTree, TestFn, 11, 12);
    check_dominance(DomTree, TestFn, 11, 13);
    check_dominance(DomTree, TestFn, 11, 14);
    check_dominance(DomTree, TestFn, 11, 15);

    check_dominance(DomTree, TestFn, 14, 15);

    check_no_dominance(DomTree, TestFn, 12, 13);
    check_no_dominance(DomTree, TestFn, 12, 14);
    check_no_dominance(DomTree, TestFn, 13, 14);

    // check with some invalid inputs
    EXPECT_FALSE(DomTree.Dominates(nullptr, Entry));
    EXPECT_FALSE(DomTree.Dominates(Entry, nullptr));
    EXPECT_FALSE(DomTree.Dominates(nullptr, nullptr));
    EXPECT_FALSE(DomTree.Dominates(10, 1));
    EXPECT_FALSE(DomTree.Dominates(1, 10));
    EXPECT_FALSE(DomTree.Dominates(1, 1));

    EXPECT_FALSE(DomTree.StrictlyDominates(nullptr, Entry));
    EXPECT_FALSE(DomTree.StrictlyDominates(Entry, nullptr));
    EXPECT_FALSE(DomTree.StrictlyDominates(nullptr, nullptr));
    EXPECT_FALSE(DomTree.StrictlyDominates(10, 1));
    EXPECT_FALSE(DomTree.StrictlyDominates(1, 10));
    EXPECT_FALSE(DomTree.StrictlyDominates(1, 1));

    EXPECT_EQ(DomTree.ImmediateDominator(Entry), nullptr);
    EXPECT_EQ(DomTree.ImmediateDominator(nullptr), nullptr);

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)),
              getBasicBlock(TestFn, 10));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)),
              getBasicBlock(TestFn, 11));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 13)),
              getBasicBlock(TestFn, 11));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 14)),
              getBasicBlock(TestFn, 11));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 15)),
              getBasicBlock(TestFn, 14));
  }

  // Test post dominator tree
  {
    opt::PostDominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, getBasicBlock(TestFn, 15));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12, 13, 14, 15})
      check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 14, 10);
    check_dominance(DomTree, TestFn, 14, 11);
    check_dominance(DomTree, TestFn, 14, 12);
    check_dominance(DomTree, TestFn, 14, 13);

    check_dominance(DomTree, TestFn, 15, 10);
    check_dominance(DomTree, TestFn, 15, 11);
    check_dominance(DomTree, TestFn, 15, 12);
    check_dominance(DomTree, TestFn, 15, 13);
    check_dominance(DomTree, TestFn, 15, 14);

    check_no_dominance(DomTree, TestFn, 13, 12);
    check_no_dominance(DomTree, TestFn, 12, 11);
    check_no_dominance(DomTree, TestFn, 13, 11);

    // check with some invalid inputs
    EXPECT_FALSE(DomTree.Dominates(nullptr, Entry));
    EXPECT_FALSE(DomTree.Dominates(Entry, nullptr));
    EXPECT_FALSE(DomTree.Dominates(nullptr, nullptr));
    EXPECT_FALSE(DomTree.Dominates(10, 1));
    EXPECT_FALSE(DomTree.Dominates(1, 10));
    EXPECT_FALSE(DomTree.Dominates(1, 1));

    EXPECT_FALSE(DomTree.StrictlyDominates(nullptr, Entry));
    EXPECT_FALSE(DomTree.StrictlyDominates(Entry, nullptr));
    EXPECT_FALSE(DomTree.StrictlyDominates(nullptr, nullptr));
    EXPECT_FALSE(DomTree.StrictlyDominates(10, 1));
    EXPECT_FALSE(DomTree.StrictlyDominates(1, 10));
    EXPECT_FALSE(DomTree.StrictlyDominates(1, 1));

    EXPECT_EQ(DomTree.ImmediateDominator(nullptr), nullptr);

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)),
              getBasicBlock(TestFn, 14));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)),
              getBasicBlock(TestFn, 14));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 13)),
              getBasicBlock(TestFn, 14));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 14)),
              getBasicBlock(TestFn, 15));

    // Exit node
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 15)), nullptr);
  }
}

TEST_F(PassClassTest, DominatorIrreducibleCFG) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstantFalse %4
          %7 = OpConstantTrue %4
          %1 = OpFunction %2 None %3
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpBranchConditional %7 %10 %11
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpBranchConditional %7 %10 %12
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const ir::Function* TestFn = getFunction(*module, 1);

  const ir::BasicBlock* Entry = getBasicBlock(TestFn, 8);
  EXPECT_EQ(Entry, TestFn->entry().get())
      << "The entry node is not the expected one";

  // Check normal dominator tree
  {
    opt::DominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, Entry);

    // (strict) dominance checks
    for (uint32_t id : {8, 9, 10, 11, 12})
      check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 8, 9);
    check_dominance(DomTree, TestFn, 8, 10);
    check_dominance(DomTree, TestFn, 8, 11);
    check_dominance(DomTree, TestFn, 8, 12);

    check_dominance(DomTree, TestFn, 9, 10);
    check_dominance(DomTree, TestFn, 9, 11);
    check_dominance(DomTree, TestFn, 9, 12);

    check_dominance(DomTree, TestFn, 11, 12);

    check_no_dominance(DomTree, TestFn, 10, 11);

    EXPECT_EQ(DomTree.ImmediateDominator(Entry), nullptr);

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 9)),
              getBasicBlock(TestFn, 8));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 10)),
              getBasicBlock(TestFn, 9));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)),
              getBasicBlock(TestFn, 9));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)),
              getBasicBlock(TestFn, 11));
  }

  // Check post dominator tree
  {
    opt::PostDominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, getBasicBlock(TestFn, 12));

    // (strict) dominance checks
    for (uint32_t id : {8, 9, 10, 11, 12})
      check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 12, 8);
    check_dominance(DomTree, TestFn, 12, 10);
    check_dominance(DomTree, TestFn, 12, 11);
    check_dominance(DomTree, TestFn, 12, 12);

    check_dominance(DomTree, TestFn, 11, 8);
    check_dominance(DomTree, TestFn, 11, 9);
    check_dominance(DomTree, TestFn, 11, 10);

    check_dominance(DomTree, TestFn, 9, 8);

    EXPECT_EQ(DomTree.ImmediateDominator(Entry), getBasicBlock(TestFn, 9));

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 9)),
              getBasicBlock(TestFn, 11));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 10)),
              getBasicBlock(TestFn, 11));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)),
              getBasicBlock(TestFn, 12));

    // Exit node.
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)), nullptr);
  }
}

TEST_F(PassClassTest, DominatorLoopToSelf) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstant %5 0
          %7 = OpConstantFalse %4
          %8 = OpConstantTrue %4
          %9 = OpConstant %5 1
          %1 = OpFunction %2 None %3
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpSwitch %6 %12 1 %11
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const ir::Function* TestFn = getFunction(*module, 1);

  const ir::BasicBlock* Entry = getBasicBlock(TestFn, 10);
  EXPECT_EQ(Entry, TestFn->entry().get())
      << "The entry node is not the expected one";

  // Check normal dominator tree
  {
    opt::DominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, Entry);

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12}) check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 10, 11);
    check_dominance(DomTree, TestFn, 10, 12);
    check_dominance(DomTree, TestFn, 11, 12);

    EXPECT_EQ(DomTree.ImmediateDominator(Entry), nullptr);

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)),
              getBasicBlock(TestFn, 10));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)),
              getBasicBlock(TestFn, 11));
  }

  // Check post dominator tree
  {
    opt::PostDominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, getBasicBlock(TestFn, 12));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12}) check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 12, 10);
    check_dominance(DomTree, TestFn, 12, 11);
    check_dominance(DomTree, TestFn, 12, 12);

    EXPECT_EQ(DomTree.ImmediateDominator(Entry), getBasicBlock(TestFn, 11));

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)),
              getBasicBlock(TestFn, 12));

    // Exit node
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)), nullptr);
  }
}

TEST_F(PassClassTest, DominatorUnreachableInLoop) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstant %5 0
          %7 = OpConstantFalse %4
          %8 = OpConstantTrue %4
          %9 = OpConstant %5 1
          %1 = OpFunction %2 None %3
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpSwitch %6 %12 1 %13
         %12 = OpLabel
               OpBranch %14
         %13 = OpLabel
               OpUnreachable
         %14 = OpLabel
               OpBranchConditional %8 %11 %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const ir::Function* TestFn = getFunction(*module, 1);

  const ir::BasicBlock* Entry = getBasicBlock(TestFn, 10);
  EXPECT_EQ(Entry, TestFn->entry().get())
      << "The entry node is not the expected one";

  // Check normal dominator tree
  {
    opt::DominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, Entry);

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12, 13, 14, 15})
      check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 10, 11);
    check_dominance(DomTree, TestFn, 10, 13);
    check_dominance(DomTree, TestFn, 10, 12);
    check_dominance(DomTree, TestFn, 10, 14);
    check_dominance(DomTree, TestFn, 10, 15);

    check_dominance(DomTree, TestFn, 11, 12);
    check_dominance(DomTree, TestFn, 11, 13);
    check_dominance(DomTree, TestFn, 11, 14);
    check_dominance(DomTree, TestFn, 11, 15);

    check_dominance(DomTree, TestFn, 12, 14);
    check_dominance(DomTree, TestFn, 12, 15);

    check_dominance(DomTree, TestFn, 14, 15);

    check_no_dominance(DomTree, TestFn, 13, 12);
    check_no_dominance(DomTree, TestFn, 13, 14);
    check_no_dominance(DomTree, TestFn, 13, 15);

    EXPECT_EQ(DomTree.ImmediateDominator(Entry), nullptr);

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)),
              getBasicBlock(TestFn, 10));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)),
              getBasicBlock(TestFn, 11));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 13)),
              getBasicBlock(TestFn, 11));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 14)),
              getBasicBlock(TestFn, 12));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 15)),
              getBasicBlock(TestFn, 14));
  }

  // Check post dominator tree
  {
    opt::PostDominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    std::set<uint32_t> exits{15, 13, 14, 11};
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    for (opt::DominatorTreeNode* node : Tree) {
      EXPECT_TRUE(exits.count(node->id()) != 0);
    }

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12, 13, 14, 15})
      check_dominance(DomTree, TestFn, id, id);

    check_no_dominance(DomTree, TestFn, 15, 10);
    check_no_dominance(DomTree, TestFn, 15, 11);
    check_no_dominance(DomTree, TestFn, 15, 12);
    check_no_dominance(DomTree, TestFn, 15, 13);
    check_no_dominance(DomTree, TestFn, 15, 14);

    check_dominance(DomTree, TestFn, 14, 12);

    check_no_dominance(DomTree, TestFn, 13, 10);
    check_no_dominance(DomTree, TestFn, 13, 11);
    check_no_dominance(DomTree, TestFn, 13, 12);
    check_no_dominance(DomTree, TestFn, 13, 14);
    check_no_dominance(DomTree, TestFn, 13, 15);

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 10)),
              getBasicBlock(TestFn, 11));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)),
              getBasicBlock(TestFn, 14));

    // Exit nodes.
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 15)), nullptr);
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 13)), nullptr);
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 14)), nullptr);
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)), nullptr);
  }
}

TEST_F(PassClassTest, DominatorInfinitLoop) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstant %5 0
          %7 = OpConstantFalse %4
          %8 = OpConstantTrue %4
          %9 = OpConstant %5 1
          %1 = OpFunction %2 None %3
         %10 = OpLabel
               OpBranch %11
         %11 = OpLabel
               OpSwitch %6 %12 1 %13
         %12 = OpLabel
               OpReturn
         %13 = OpLabel
               OpBranch %13
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const ir::Function* TestFn = getFunction(*module, 1);

  const ir::BasicBlock* Entry = getBasicBlock(TestFn, 10);
  EXPECT_EQ(Entry, TestFn->entry().get())
      << "The entry node is not the expected one";
  // Check normal dominator tree
  {
    opt::DominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, Entry);

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12, 13})
      check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 10, 11);
    check_dominance(DomTree, TestFn, 10, 12);
    check_dominance(DomTree, TestFn, 10, 13);

    check_dominance(DomTree, TestFn, 11, 12);
    check_dominance(DomTree, TestFn, 11, 13);

    check_no_dominance(DomTree, TestFn, 13, 12);

    EXPECT_EQ(DomTree.ImmediateDominator(Entry), nullptr);

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)),
              getBasicBlock(TestFn, 10));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)),
              getBasicBlock(TestFn, 11));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 13)),
              getBasicBlock(TestFn, 11));
  }

  // Check post dominator tree
  {
    opt::PostDominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, getBasicBlock(TestFn, 12));

    // (strict) dominance checks
    for (uint32_t id : {10, 11, 12}) check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 12, 11);
    check_dominance(DomTree, TestFn, 12, 10);

    // 13 should be completely out of tree as it's unreachable from exit nodes
    check_no_dominance(DomTree, TestFn, 12, 13);
    check_no_dominance(DomTree, TestFn, 11, 13);
    check_no_dominance(DomTree, TestFn, 10, 13);

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 10)),
              getBasicBlock(TestFn, 11));

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 11)),
              getBasicBlock(TestFn, 12));

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 12)), nullptr);
  }
}

TEST_F(PassClassTest, DominatorUnreachableFromEntry) {
  const std::string text = R"(
               OpCapability Addresses
               OpCapability Addresses
               OpCapability Kernel
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %1 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpTypeBool
          %5 = OpTypeInt 32 0
          %6 = OpConstantFalse %4
          %7 = OpConstantTrue %4
          %1 = OpFunction %2 None %3
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpReturn
         %10 = OpLabel
               OpBranch %9
               OpFunctionEnd
)";
  // clang-format on
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_0, nullptr, text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ir::Module* module = context->module();
  EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                             << text << std::endl;
  const ir::Function* TestFn = getFunction(*module, 1);

  const ir::BasicBlock* Entry = getBasicBlock(TestFn, 8);
  EXPECT_EQ(Entry, TestFn->entry().get())
      << "The entry node is not the expected one";

  // Check dominator tree
  {
    opt::DominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);
    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, Entry);

    // (strict) dominance checks
    for (uint32_t id : {8, 9}) check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 8, 9);

    check_no_dominance(DomTree, TestFn, 10, 8);
    check_no_dominance(DomTree, TestFn, 10, 9);

    EXPECT_EQ(DomTree.ImmediateDominator(Entry), nullptr);

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 9)),
              getBasicBlock(TestFn, 8));
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 10)), nullptr);
  }

  // Check post dominator tree
  {
    opt::PostDominatorAnalysis DomTree;
    DomTree.InitializeTree(TestFn);

    // Inspect the actual tree
    opt::DominatorTree& Tree = DomTree.GetDomTree();
    EXPECT_EQ(Tree.GetRoot()->BB, getBasicBlock(TestFn, 9));

    // (strict) dominance checks
    for (uint32_t id : {8, 9, 10}) check_dominance(DomTree, TestFn, id, id);

    check_dominance(DomTree, TestFn, 9, 8);
    check_dominance(DomTree, TestFn, 9, 10);

    EXPECT_EQ(DomTree.ImmediateDominator(Entry), getBasicBlock(TestFn, 9));

    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 9)), nullptr);
    EXPECT_EQ(DomTree.ImmediateDominator(getBasicBlock(TestFn, 10)),
              getBasicBlock(TestFn, 9));
  }
}

}  // namespace
