
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
#include <vector>
#include <utility>

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

class Block {
  string label_;
  string body_;
  SpvOp type_;
  vector<Block> successors_;

 public:
  Block(string label, SpvOp type = SpvOpBranch)
      : label_(label), body_(), type_(type), successors_() {}

  Block &setBody(std::string body) {
    body_ = body;
    return *this;
  }

  operator string() {
    stringstream out;
    out << std::setw(8) << "%" + label_ + "  = OpLabel \n";
    if (!body_.empty()) {
      out << body_;
    }

    switch (type_) {
      case SpvOpBranchConditional:
        out << "OpBranchConditional %cond ";
        for (Block &b : successors_) {
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
        out << "OpReturn\n";
        break;
      case SpvOpUnreachable:
        out << "OpUnreachable\n";
        break;
      case SpvOpBranch:
        out << "OpBranch %" + successors_.front().label_;
        break;
      default:
        assert(1 != 1 && "Unhandled");
    }
    out << "\n";

    return out.str();
  }
  friend Block &operator>>(Block &curr, vector<Block> successors);
  friend Block &operator>>(Block &lhs, Block &successor);
};

Block &operator>>(Block &lhs, vector<Block> successors) {
  if (lhs.type_ == SpvOpBranchConditional) {
    assert(successors.size() == 2);
  } else if (lhs.type_ == SpvOpSwitch) {
    assert(successors.size() > 1);
  }
  lhs.successors_ = successors;
  return lhs;
}

Block &operator>>(Block &lhs, Block &successor) {
  assert(lhs.type_ == SpvOpBranch);
  lhs.successors_.push_back(successor);
  return lhs;
}

string header =
    "OpCapability Shader\n"
    "OpMemoryModel Logical GLSL450\n";

string types_consts =
    nameOps("voidt", "boolt", "intt", "one", "two", "ptrt", "funct") +
    "%voidt   = OpTypeVoid\n"
    "%boolt   = OpTypeBool\n"
    "%intt    = OpTypeInt 32 1\n"
    "%one     = OpConstant %intt 1\n"
    "%two     = OpConstant %intt 2\n"
    "%ptrt    = OpTypePointer Function %intt\n"
    "%funct   = OpTypeFunction %voidt\n";

TEST_F(ValidateCFG, Default) {
  Block first("first");
  Block loop("loop", SpvOpBranchConditional);
  Block cont("cont");
  Block merge("merge", SpvOpReturn);

  loop.setBody(
      "%cond    = OpSLessThan %intt %one %two\n"
      "OpLoopMerge %merge %cont None\n");

  string str = header +
               nameOps("loop", "first", "cont", "merge", make_pair("func", "Main")) +
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
  EXPECT_THAT(getDiagnosticString(), HasSubstr("first block"));
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
              HasSubstr("appears in the binary before its dominator"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("cont"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("branch"));
}

TEST_F(ValidateCFG, MergeBlockTargetedByMultipleHeaderBlocksBad) {
  Block entry("entry");
  Block loop("loop");
  Block badhead("badhead", SpvOpBranchConditional);
  Block t("t");
  Block f("f");
  Block cont("cont");
  Block merge("merge");
  Block end("end", SpvOpReturn);

  // cannot share the same merge
  loop.setBody(
      " %cond2   = OpSLessThan %intt %one %two\n"
      " OpLoopMerge %merge %cont None\n");
  badhead.setBody(
      " %cond1   = OpSLessThan %intt %one %two\n"
      "OpSelectionMerge %merge None\n");

  string str = header + nameOps("merge", make_pair("func", "Main")) + types_consts +
               "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> badhead;
  str += badhead >> vector<Block>({t, f});
  str += t >> merge;
  str += f >> cont;
  str += cont >> loop;
  str += merge >> end;
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("merge"));
}

TEST_F(ValidateCFG, MergeBlockTargetedByMultipleHeaderBlocksSelectionBad) {
  Block entry("entry");
  Block loop("loop");
  Block badhead("badhead", SpvOpBranchConditional);
  Block t("t");
  Block f("f");
  Block cont("cont");
  Block merge("merge");
  Block end("end", SpvOpReturn);

  // cannot share the same merge
  loop.setBody(" OpLoopMerge %merge %cont None\n");

  badhead.setBody(
      " %cond   = OpSLessThan %intt %one %two\n"
      " OpSelectionMerge %merge None\n");

  string str = header + nameOps("merge", make_pair("func", "Main")) + types_consts +
               "%func    = OpFunction %voidt None %funct\n";

  str += entry >> badhead;
  str += badhead >> vector<Block>({t, f});
  str += t >> merge;
  str += f >> cont;
  str += cont >> loop;
  str += loop >> merge;
  str += merge >> end;
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("merge"));
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
  EXPECT_THAT(getDiagnosticString(), HasSubstr("First block"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("entry"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Main"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("bad"));
}

TEST_F(ValidateCFG, BranchConditionalTrueTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpBranchConditional);
  Block f("f");
  Block merge("merge");
  Block end("end", SpvOpReturn);

  bad.setBody(
      " %cond    = OpSLessThan %intt %one %two\n"
      "OpLoopMerge %merge %cont None\n");

  string str = header + nameOps("entry", "bad", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> vector<Block>({entry, f});  // cannot target entry block
  str += f >> merge;
  str += merge >> end;
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("First block"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("entry"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Main"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("bad"));
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
  EXPECT_THAT(getDiagnosticString(), HasSubstr("First block"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("entry"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Main"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("bad"));
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
  EXPECT_THAT(getDiagnosticString(), HasSubstr("First block"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("entry"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Main"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("bad"));
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

  string str = header + nameOps("middle2", make_pair("func", "Main")) + types_consts +
               "%func    = OpFunction %voidt None %funct\n";

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
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Main"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("middle2"));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("are referenced but not defined"));
}

TEST_F(ValidateCFG, HeaderDoesntDominatesMergeBad) {
  Block entry("entry");
  Block bad("bad", SpvOpBranchConditional);
  Block f("f");
  Block merge("merge", SpvOpBranchConditional);
  Block cont("cont");
  Block end("end", SpvOpReturn);

  bad.setBody("OpLoopMerge %merge %cont None\n");

  merge.setBody(
      " %cond    = OpSLessThan %intt %one %two\n"
      "OpSelectionMerge %end None\n");

  string str = header + nameOps("bad", "merge", make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> merge;
  str += merge >> vector<Block>({bad, end});
  str += bad >> vector<Block>({cont, f});
  str += cont >> merge;
  str += f >> end;
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Header block"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("doesn't dominate"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("[bad]"));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("[merge]"));
}

TEST_F(ValidateCFG, UnreachableMerge) {
  Block entry("entry");
  Block branch("branch", SpvOpBranchConditional);
  Block t("t", SpvOpReturn);
  Block f("f", SpvOpReturn);
  Block merge("merge", SpvOpUnreachable);
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
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

}
