
// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

// Validation tests for Control Flow Graph

#include <array>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"

#include "TestFixture.h"
#include "UnitSPIRV.h"
#include "ValidateFixtures.h"
#include "source/diagnostic.h"
#include "source/validate.h"

using std::array;
using std::make_pair;
using std::pair;
using std::string;
using std::stringstream;
using std::vector;

using ::testing::HasSubstr;
using ::testing::MatchesRegex;

using libspirv::BasicBlock;
using libspirv::ValidationState_t;

using ValidateCFG = spvtest::ValidateBase<bool>;
using spvtest::ScopedContext;

namespace {

string nameOps() { return ""; }

template <typename... Args>
string nameOps(pair<string, string> head, Args... names) {
  return "OpName %" + head.first + " \"" + head.second + "\"\n" +
         nameOps(names...);
}

template <typename... Args>
string nameOps(string head, Args... names) {
  return "OpName %" + head + " \"" + head + "\"\n" + nameOps(names...);
}

/// This class allows the easy creation of complex control flow without writing
/// SPIR-V. This class is used in the test cases below.
class Block {
  string label_;
  string body_;
  SpvOp type_;
  vector<Block> successors_;

 public:
  /// Creates a Block with a given label
  ///
  /// @param[in]: label the label id of the block
  /// @param[in]: type the branch instruciton that ends the block
  explicit Block(string label, SpvOp type = SpvOpBranch)
      : label_(label), body_(), type_(type), successors_() {}

  /// Sets the instructions which will appear in the body of the block
  Block& setBody(std::string body) {
    body_ = body;
    return *this;
  }

  /// Converts the block into a SPIR-V string
  operator string() {
    stringstream out;
    out << std::setw(8) << "%" + label_ + "  = OpLabel \n";
    if (!body_.empty()) {
      out << body_;
    }

    switch (type_) {
      case SpvOpBranchConditional:
        out << "OpBranchConditional %cond ";
        for (Block& b : successors_) {
          out << "%" + b.label_ + " ";
        }
        break;
      case SpvOpSwitch: {
        out << "OpSwitch %one %" + successors_.front().label_;
        stringstream ss;
        for (size_t i = 1; i < successors_.size(); i++) {
          ss << " " << i << " %" << successors_[i].label_;
        }
        out << ss.str();
      } break;
      case SpvOpReturn:
        assert(successors_.size() == 0);
        out << "OpReturn\n";
        break;
      case SpvOpUnreachable:
        assert(successors_.size() == 0);
        out << "OpUnreachable\n";
        break;
      case SpvOpBranch:
        assert(successors_.size() == 1);
        out << "OpBranch %" + successors_.front().label_;
        break;
      default:
        assert(1 != 0 && "Unhandled");
    }
    out << "\n";

    return out.str();
  }
  friend Block& operator>>(Block& curr, vector<Block> successors);
  friend Block& operator>>(Block& lhs, Block& successor);
};

/// Assigns the successors for the Block on the lhs
Block& operator>>(Block& lhs, vector<Block> successors) {
  if (lhs.type_ == SpvOpBranchConditional) {
    assert(successors.size() == 2);
  } else if (lhs.type_ == SpvOpSwitch) {
    assert(successors.size() > 1);
  }
  lhs.successors_ = successors;
  return lhs;
}

/// Assigns the successor for the Block on the lhs
Block& operator>>(Block& lhs, Block& successor) {
  assert(lhs.type_ == SpvOpBranch);
  lhs.successors_.push_back(successor);
  return lhs;
}

string header =
    "OpCapability Shader\n"
    "OpMemoryModel Logical GLSL450\n";

string types_consts =
    "%voidt   = OpTypeVoid\n"
    "%boolt   = OpTypeBool\n"
    "%intt    = OpTypeInt 32 1\n"
    "%one     = OpConstant %intt 1\n"
    "%two     = OpConstant %intt 2\n"
    "%ptrt    = OpTypePointer Function %intt\n"
    "%funct   = OpTypeFunction %voidt\n";

TEST_F(ValidateCFG, Simple) {
  Block first("first");
  Block loop("loop", SpvOpBranchConditional);
  Block cont("cont");
  Block merge("merge", SpvOpReturn);

  loop.setBody(
      "%cond    = OpSLessThan %intt %one %two\n"
      "OpLoopMerge %merge %cont None\n");

  string str = header + nameOps("loop", "first", "cont", "merge",
                                make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += first >> loop;
  str += loop >> vector<Block>({cont, merge});
  str += cont >> loop;
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, Variable) {
  Block entry("entry");
  Block cont("cont");
  Block exit("exit", SpvOpReturn);

  entry.setBody("%var = OpVariable %ptrt Function\n");

  string str = header + nameOps(make_pair("func", "Main")) + types_consts +
               " %func    = OpFunction %voidt None %funct\n";
  str += entry >> cont;
  str += cont >> exit;
  str += exit;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, VariableNotInFirstBlockBad) {
  Block entry("entry");
  Block cont("cont");
  Block exit("exit", SpvOpReturn);

  // This operation should only be performed in the entry block
  cont.setBody("%var = OpVariable %ptrt Function\n");

  string str = header + nameOps(make_pair("func", "Main")) + types_consts +
               " %func    = OpFunction %voidt None %funct\n";

  str += entry >> cont;
  str += cont >> exit;
  str += exit;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Variables can only be defined in the first block of a function"));
}

TEST_F(ValidateCFG, BlockAppearsBeforeDominatorBad) {
  Block entry("entry");
  Block cont("cont");
  Block branch("branch", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  branch.setBody(
      " %cond    = OpSLessThan %intt %one %two\n"
      "OpSelectionMerge %merge None\n");

  string str = header + nameOps("cont", "branch", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> branch;
  str += cont >> merge;  // cont appears before its dominator
  str += branch >> vector<Block>({cont, merge});
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("Block .\\[cont\\] appears in the binary "
                           "before its dominator .\\[branch\\]"));
}

TEST_F(ValidateCFG, MergeBlockTargetedByMultipleHeaderBlocksBad) {
  Block entry("entry");
  Block loop("loop");
  Block selection("selection", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  loop.setBody(
      " %cond   = OpSLessThan %intt %one %two\n"
      " OpLoopMerge %merge %loop None\n");
  // cannot share the same merge
  selection.setBody("OpSelectionMerge %merge None\n");

  string str = header + nameOps("merge", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> selection;
  str += selection >> vector<Block>({loop, merge});
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("Block .\\[merge\\] is already a merge block "
                           "for another header"));
}

TEST_F(ValidateCFG, MergeBlockTargetedByMultipleHeaderBlocksSelectionBad) {
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block selection("selection", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  selection.setBody(
      " %cond   = OpSLessThan %intt %one %two\n"
      " OpSelectionMerge %merge None\n");
  // cannot share the same merge
  loop.setBody(" OpLoopMerge %merge %loop None\n");

  string str = header + nameOps("merge", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> selection;
  str += selection >> vector<Block>({merge, loop});
  str += loop >> vector<Block>({loop, merge});
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("Block .\\[merge\\] is already a merge block "
                           "for another header"));
}

TEST_F(ValidateCFG, BranchTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad");
  Block end("end", SpvOpReturn);
  string str = header + nameOps("entry", "bad", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> entry;  // Cannot target entry block
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("First block .\\[entry\\] of funciton .\\[Main\\] "
                           "is targeted by block .\\[bad\\]"));
}

TEST_F(ValidateCFG, BranchConditionalTrueTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpBranchConditional);
  Block exit("exit", SpvOpReturn);

  bad.setBody(
      " %cond    = OpSLessThan %intt %one %two\n"
      " OpLoopMerge %entry %exit None\n");

  string str = header + nameOps("entry", "bad", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> vector<Block>({entry, exit});  // cannot target entry block
  str += exit;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("First block .\\[entry\\] of funciton .\\[Main\\] "
                           "is targeted by block .\\[bad\\]"));
}

TEST_F(ValidateCFG, BranchConditionalFalseTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpBranchConditional);
  Block t("t");
  Block merge("merge");
  Block end("end", SpvOpReturn);

  bad.setBody(
      "%cond    = OpSLessThan %intt %one %two\n"
      "OpLoopMerge %merge %cont None\n");

  string str = header + nameOps("entry", "bad", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> vector<Block>({t, entry});
  str += merge >> end;
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("First block .\\[entry\\] of funciton .\\[Main\\] "
                           "is targeted by block .\\[bad\\]"));
}

TEST_F(ValidateCFG, SwitchTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpSwitch);
  Block block1("block1");
  Block block2("block2");
  Block block3("block3");
  Block def("def");  // default block
  Block merge("merge");
  Block end("end", SpvOpReturn);

  bad.setBody(
      "%cond    = OpSLessThan %intt %one %two\n"
      "OpSelectionMerge %merge None\n");

  string str = header + nameOps("entry", "bad", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> vector<Block>({def, block1, block2, block3, entry});
  str += def >> merge;
  str += block1 >> merge;
  str += block2 >> merge;
  str += block3 >> merge;
  str += merge >> end;
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("First block .\\[entry\\] of funciton .\\[Main\\] "
                           "is targeted by block .\\[bad\\]"));
}

TEST_F(ValidateCFG, BranchToBlockInOtherFunctionBad) {
  Block entry("entry");
  Block middle("middle", SpvOpBranchConditional);
  Block end("end", SpvOpReturn);

  middle.setBody(
      "%cond    = OpSLessThan %intt %one %two\n"
      "OpSelectionMerge %end None\n");

  Block entry2("entry2");
  Block middle2("middle2");
  Block end2("end2", SpvOpReturn);

  string str = header + nameOps("middle2", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> middle;
  str += middle >> vector<Block>({end, middle2});
  str += end;
  str += "OpFunctionEnd\n";

  str += "%func2    = OpFunction %voidt None %funct\n";
  str += entry2 >> middle2;
  str += middle2 >> end2;
  str += end2;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      MatchesRegex("Block\\(s\\) \\{.\\[middle2\\] .\\} are referenced but not "
                   "defined in function .\\[Main\\]"));
}

TEST_F(ValidateCFG, HeaderDoesntDominatesMergeBad) {
  Block entry("entry");
  Block merge("merge");
  Block head("head", SpvOpBranchConditional);
  Block f("f");
  Block exit("exit", SpvOpReturn);

  entry.setBody("%cond = OpSLessThan %intt %one %two\n");

  head.setBody("OpSelectionMerge %merge None\n");

  string str = header + nameOps("head", "merge", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> merge;
  str += head >> vector<Block>({merge, f});
  str += f >> merge;
  str += merge >> exit;
  str += exit;

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      MatchesRegex("Header block .\\[head\\] doesn't dominate its merge block "
                   ".\\[merge\\]"));
}

TEST_F(ValidateCFG, UnreachableMerge) {
  Block entry("entry");
  Block branch("branch", SpvOpBranchConditional);
  Block t("t", SpvOpReturn);
  Block f("f", SpvOpReturn);
  Block merge("merge");
  Block end("end", SpvOpReturn);

  branch.setBody(
      " %cond    = OpSLessThan %intt %one %two\n"
      "OpSelectionMerge %merge None\n");

  string str = header + nameOps("branch", "merge", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> branch;
  str += branch >> vector<Block>({t, f});
  str += t;
  str += f;
  str += merge >> end;
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, UnreachableMergeDefinedByOpUnreachable) {
  Block entry("entry");
  Block branch("branch", SpvOpBranchConditional);
  Block t("t", SpvOpReturn);
  Block f("f", SpvOpReturn);
  Block merge("merge", SpvOpUnreachable);

  branch.setBody(
      " %cond    = OpSLessThan %intt %one %two\n"
      "OpSelectionMerge %merge None\n");

  string str = header + nameOps("branch", "merge", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> branch;
  str += branch >> vector<Block>({t, f});
  str += t;
  str += f;
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, UnreachableBlock) {
  Block entry("entry");
  Block unreachable("unreachable");
  Block exit("exit", SpvOpReturn);

  string str = header +
               nameOps("unreachable", "exit", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> exit;
  str += unreachable >> exit;
  str += exit;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, UnreachableBranch) {
  Block entry("entry");
  Block unreachable("unreachable", SpvOpBranchConditional);
  Block unreachablechildt("unreachablechildt");
  Block unreachablechildf("unreachablechildf");
  Block merge("merge");
  Block exit("exit", SpvOpReturn);

  unreachable.setBody(
      " %cond    = OpSLessThan %intt %one %two\n"
      "OpSelectionMerge %merge None\n");
  string str = header +
               nameOps("unreachable", "exit", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> exit;
  str += unreachable >> vector<Block>({unreachablechildt, unreachablechildf});
  str += unreachablechildt >> merge;
  str += unreachablechildf >> merge;
  str += merge >> exit;
  str += exit;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, EmptyFunction) {
  string str = header + types_consts +
               "%func    = OpFunction %voidt None %funct\n" + "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, SingleBlockLoop) {
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block exit("exit", SpvOpReturn);

  loop.setBody(
      "%cond    = OpSLessThan %intt %one %two\n"
      "OpLoopMerge %exit %loop None\n");

  string str =
      header + types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> vector<Block>({loop, exit});
  str += exit;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, NestedLoops) {
  Block entry("entry");
  Block loop1("loop1", SpvOpBranchConditional);
  Block loop2("loop2", SpvOpBranchConditional);
  Block loop2_merge("loop2_merge");
  Block exit("exit", SpvOpReturn);

  loop1.setBody(
      "%cond    = OpSLessThan %intt %one %two\n"
      "OpLoopMerge %exit %loop2 None\n");

  loop2.setBody("OpLoopMerge %loop2_merge %loop2 None\n");

  string str =
      header + types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop1;
  str += loop1 >> vector<Block>({loop2, exit});
  str += loop2 >> vector<Block>({loop2, loop2_merge});
  str += loop2_merge >> exit;
  str += exit;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, NestedSelection) {
  Block entry("entry");
  const int N = 256;
  vector<Block> if_blocks;
  vector<Block> merge_blocks;
  Block inner("inner");

  if_blocks.emplace_back("if0", SpvOpBranchConditional);
  if_blocks[0].setBody(
      "%cond    = OpSLessThan %intt %one %two\n"
      "OpSelectionMerge %if_merge0 None\n");
  merge_blocks.emplace_back("if_merge0", SpvOpReturn);

  for (int i = 1; i < N; i++) {
    stringstream ss;
    ss << i;
    if_blocks.emplace_back("if" + ss.str(), SpvOpBranchConditional);
    if_blocks[i].setBody("OpSelectionMerge %if_merge" + ss.str() + " None\n");
    merge_blocks.emplace_back("if_merge" + ss.str(), SpvOpBranch);
  }
  string str =
      header + types_consts + "%func    = OpFunction %voidt None %funct\n";

  for (int i = 0; i < N - 1; i++) {
    str += if_blocks[i] >> vector<Block>({if_blocks[i + 1], merge_blocks[i]});
  }
  str += if_blocks.back() >> vector<Block>({inner, merge_blocks.back()});
  str += inner >> merge_blocks.back();
  for (int i = N - 1; i > 0; i--) {
    str += merge_blocks[i] >> merge_blocks[i - 1];
  }
  str += merge_blocks[0];
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}
/// TODO(umar): Switch instructions
/// TODO(umar): CFG branching outside of CFG construct
/// TODO(umar): Nested CFG constructs
}
