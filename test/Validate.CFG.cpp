
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

#include "UnitSPIRV.h"
#include "source/validate.h"
#include "ValidateFixtures.h"
#include "source/diagnostic.h"

using std::array;
using std::make_pair;
using std::pair;
using std::string;
using std::stringstream;
using std::vector;

using ::testing::HasSubstr;

using libspirv::spvResultToString;
using libspirv::ValidationState_t;

using ValidateCFG = spvtest::ValidateBase<bool>;

using libspirv::BasicBlock;
namespace libspirv {
vector<const BasicBlock *> PostOrderSort(const BasicBlock &entry,
                                         size_t size = 10);
}

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
        out << "OpSwitch %one %" + successors_.front().label_ ;
        stringstream ss;
        for (size_t i = 1; i < successors_.size(); i++) {
          ss << " " << i << " %" << successors_[i].label_;
        }
        out << ss.str();
      } break;
      case SpvOpReturn:
        out << "OpReturn\n OpFunctionEnd\n";
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

TEST_F(ValidateCFG, PostOrderLinear) {
  vector<BasicBlock> blocks;
  ValidationState_t state(nullptr, context_);

  for (int i = 0; i < 7; i++) {
    blocks.emplace_back(i, state);
  }

  for (int i = 0; i < 6; i++) {
    blocks[i].RegisterSuccessor({&blocks[i + 1]});
  }

  vector<const BasicBlock *> out = PostOrderSort(blocks[0]);
  vector<uint32_t> gold = {6, 5, 4, 3, 2, 1, 0};

  for (size_t i = 0; i < gold.size(); i++) {
    ASSERT_EQ(gold[i], out[i]->get_id());
  }
}

TEST_F(ValidateCFG, PostOrderWithCycle) {
  vector<BasicBlock> blocks;
  ValidationState_t state(nullptr, context_);

  for (int i = 0; i < 7; i++) {
    blocks.emplace_back(i, state);
  }

  blocks[0].RegisterSuccessor({&blocks[1]});
  blocks[1].RegisterSuccessor({&blocks[2]});
  blocks[2].RegisterSuccessor({&blocks[3], &blocks[4]});
  blocks[3].RegisterSuccessor({&blocks[5]});
  blocks[5].RegisterSuccessor({&blocks[6]});
  blocks[4].RegisterSuccessor({&blocks[2]});

  vector<const BasicBlock *> out = PostOrderSort(blocks[0]);
  vector<array<uint32_t, 7>> possible_gold = {{{4, 6, 5, 3, 2, 1, 0}},
                                              {{6, 5, 3, 4, 2, 1, 0}}};

  ASSERT_TRUE(any_of(begin(possible_gold), end(possible_gold),
                     [&](array<uint32_t, 7> gold) {
                       return equal(begin(gold), end(gold), begin(out),
                                    [](uint32_t val, const BasicBlock *block) {
                                      return val == block->get_id();
                                    });
                     }));
}

TEST_F(ValidateCFG, PostOrderWithSwitch) {
  vector<BasicBlock> blocks;
  ValidationState_t state(nullptr, context_);

  for (int i = 0; i < 7; i++) {
    blocks.emplace_back(i, state);
  }

  blocks[0].RegisterSuccessor({&blocks[1]});
  blocks[1].RegisterSuccessor({&blocks[2]});
  blocks[2].RegisterSuccessor({&blocks[3], &blocks[4], &blocks[6]});
  blocks[3].RegisterSuccessor({&blocks[6]});
  blocks[5].RegisterSuccessor({&blocks[6]});
  blocks[4].RegisterSuccessor({&blocks[5]});

  vector<const BasicBlock *> out = PostOrderSort(blocks[0]);
  vector<std::array<uint32_t, 7>> gold = {{{6, 3, 5, 4, 2, 1, 0}},
                                          {{6, 5, 4, 3, 2, 1, 0}}};

  auto dom = libspirv::CalculateDominators(blocks[0]);
  libspirv::UpdateImmediateDominators(dom);

  // for(auto &block : blocks) {
  //  printDominatorList(block);
  //  std::cout << std::endl;
  //}

  ASSERT_TRUE(any_of(
      begin(gold), end(gold), [&out](std::array<uint32_t, 7> &gold_array) {
        return std::equal(begin(gold_array), end(gold_array), begin(out),
                          [](uint32_t val, const BasicBlock *block) {
                            return val == block->get_id();
                          });
      }));
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

  string str = header + nameOps("loop", "first", "cont", "merge",
                                make_pair("func", "Main")) +
               types_consts + "%func    = OpFunction %voidt None %funct\n";

  str += first >> loop;
  str += loop >> vector<Block>({cont, merge});
  str += cont >> loop;
  str += merge;

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

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
}

TEST_F(ValidateCFG, DISABLED_NonInlineBlock) {
  Block entry("entry");
  Block cont("cont");
  Block loop("loop", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  loop.setBody(R"(
 %cond    = OpSLessThan %intt %one %two
            OpLoopMerge %merge %cont None

)");

  string str = header + nameOps(make_pair("func", "Main")) + types_consts +
               "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += cont >> loop;
  str += loop >> vector<Block>({cont, merge});
  str += merge;

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, MergeBlockTargetedByMultipleHeaderBlocksBad) {
  Block entry("entry");
  Block loop("loop", SpvOpBranch);
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

  string str = header
             + nameOps(make_pair("func", "Main"))
             + types_consts
             + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> badhead;
  str += badhead >> vector<Block>({t, f});
  str += t >> merge;
  str += f >> cont;
  str += cont >> loop;
  str += merge >> end;
  str += end;

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
}

TEST_F(ValidateCFG, BranchTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad");
  Block end("end", SpvOpReturn);
  string str = header
             + nameOps(make_pair("func", "Main"))
             + types_consts
    + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> entry; // Cannot target entry block
  str += end;

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
}

TEST_F(ValidateCFG, BranchConditionalTrueTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpBranchConditional);
  Block f("f");
  Block merge("merge");
  Block end("end", SpvOpReturn);

  bad.setBody(" %cond    = OpSLessThan %intt %one %two\n"
              "OpLoopMerge %merge %cont None\n");

  string str = header
             + nameOps(make_pair("func", "Main"))
             + types_consts
             + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> vector<Block>({entry, f}); // cannot target entry block
  str += f >> merge;
  str += merge >> end;
  str += end;

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
}

TEST_F(ValidateCFG, BranchConditionalFalseTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpBranchConditional);
  Block t("t");
  Block merge("merge");
  Block end("end", SpvOpReturn);

  bad.setBody("%cond    = OpSLessThan %intt %one %two\n"
              "OpLoopMerge %merge %cont None\n");

  string str = header
             + nameOps(make_pair("func", "Main"))
             + types_consts
             + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> vector<Block>({t, entry});
  str += merge >> end;
  str += end;

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
}

TEST_F(ValidateCFG, SwitchTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpSwitch);
  Block block1("block1");
  Block block2("block2");
  Block block3("block3");
  Block def("def"); // default block
  Block merge("merge");
  Block end("end", SpvOpReturn);

  bad.setBody("OpSelectionMerge %merge None\n");

  string str = header
             + nameOps(make_pair("func", "Main"))
             + types_consts
             + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> vector<Block>({def, block1, block2, block3, entry});
  str += def >> merge;
  str += block1 >> merge;
  str += block2 >> merge;
  str += block3 >> merge;
  str += merge >> end;
  str += end;

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
}

}
