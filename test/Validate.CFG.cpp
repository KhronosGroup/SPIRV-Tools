
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
using std::string;
using std::vector;

using ::testing::HasSubstr;

using libspirv::spvResultToString;
using libspirv::ValidationState_t;

using ValidateCFG = spvtest::ValidateBase<bool>;

using libspirv::BasicBlock;
namespace libspirv{
  vector<const BasicBlock*> PostOrderSort(const BasicBlock &entry, size_t size = 10);
}

namespace {

TEST_F(ValidateCFG, PostOrderLinear) {
  vector<BasicBlock> blocks;
  ValidationState_t state(nullptr, context_);

  for(int i = 0; i < 7; i++) {
    blocks.emplace_back(i, state);
  }

  for(int i = 0; i < 6; i++) {
    blocks[i].RegisterSuccessor({&blocks[i+1]});
  }

  vector<const BasicBlock*> out = PostOrderSort(blocks[0]);
  vector<uint32_t> gold = {6, 5, 4, 3, 2, 1, 0};

  for(size_t i = 0; i < gold.size(); i++) {
    ASSERT_EQ(gold[i], out[i]->get_id());
  }
}

TEST_F(ValidateCFG, PostOrderWithCycle) {
  vector<BasicBlock> blocks;
  ValidationState_t state(nullptr, context_);

  for(int i = 0; i < 7; i++) {
    blocks.emplace_back(i, state);
  }

  blocks[0].RegisterSuccessor({&blocks[1]});
  blocks[1].RegisterSuccessor({&blocks[2]});
  blocks[2].RegisterSuccessor({&blocks[3], &blocks[4]});
  blocks[3].RegisterSuccessor({&blocks[5]});
  blocks[5].RegisterSuccessor({&blocks[6]});
  blocks[4].RegisterSuccessor({&blocks[2]});

  vector<const BasicBlock*> out = PostOrderSort(blocks[0]);
  vector<array<uint32_t, 7>> possible_gold = {
    {{4, 6, 5, 3, 2, 1, 0}},
    {{6, 5, 3, 4, 2, 1, 0}}
  };

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

  for(int i = 0; i < 7; i++) {
    blocks.emplace_back(i, state);
  }

  blocks[0].RegisterSuccessor({&blocks[1]});
  blocks[1].RegisterSuccessor({&blocks[2]});
  blocks[2].RegisterSuccessor({&blocks[3], &blocks[4], &blocks[6]});
  blocks[3].RegisterSuccessor({&blocks[6]});
  blocks[5].RegisterSuccessor({&blocks[6]});
  blocks[4].RegisterSuccessor({&blocks[5]});

  vector<const BasicBlock*> out = PostOrderSort(blocks[0]);
  vector<std::array<uint32_t, 7>> gold = {
    {{6, 3, 5, 4, 2, 1, 0}},
    {{6, 5, 4, 3, 2, 1, 0}}
  };

  auto dom = libspirv::CalculateDominators(blocks[0]);
  libspirv::UpdateImmediateDominators(dom);

  //for(auto &block : blocks) {
  //  printDominatorList(block);
  //  std::cout << std::endl;
  //}

  ASSERT_TRUE(
      any_of(begin(gold), end(gold), [&out](std::array<uint32_t, 7> &gold_array) {
        return std::equal(begin(gold_array), end(gold_array), begin(out),
                          [](uint32_t val, const BasicBlock *block) {
                            return val == block->get_id();
                          });
      }));
}

TEST_F(ValidateCFG, Default) {
  string str = R"(
           OpCapability Shader
           OpMemoryModel Logical GLSL450
           OpName %loop "loop"
           OpName %first "first"
           OpName %cont "cont"
           OpName %merge "merge"
           OpName %func "Main"
%voidt   = OpTypeVoid
%boolt   = OpTypeBool
%intt    = OpTypeInt 32 1
%one     = OpConstant %intt 1
%two     = OpConstant %intt 2
%funct   = OpTypeFunction %voidt
%func    = OpFunction %voidt None %funct
%first   = OpLabel
           OpBranch %loop
%loop    = OpLabel
%cond    = OpSLessThan %intt %one %two
           OpLoopMerge %merge %cont None
           OpBranchConditional %cond %cont %merge
%cont    = OpLabel
           OpNop
           OpBranch %loop
%merge   = OpLabel
           OpNop
           OpReturn
           OpFunctionEnd
  )";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, Variable) {
    string str = R"(
           OpCapability Shader
           OpMemoryModel Logical GLSL450
           OpName %func "Main"
%voidt   = OpTypeVoid
%boolt   = OpTypeBool
%intt    = OpTypeInt 32 1
%ptrt    = OpTypePointer Function %intt
%one     = OpConstant %intt 1
%two     = OpConstant %intt 2
%funct   = OpTypeFunction %voidt
%func    = OpFunction %voidt None %funct
%first   = OpLabel
%var     = OpVariable %ptrt Function
           OpBranch %loop
%loop    = OpLabel
%cond    = OpSLessThan %intt %one %two
           OpLoopMerge %merge %cont None
           OpBranchConditional %cond %cont %merge
%cont    = OpLabel
           OpNop
           OpBranch %loop
%merge   = OpLabel
           OpNop
           OpReturn
           OpFunctionEnd
  )";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, VariableNotInFirstBlockBad) {
  string str = R"(
           OpCapability Shader
           OpMemoryModel Logical GLSL450
           OpName %func "Main"
%voidt   = OpTypeVoid
%boolt   = OpTypeBool
%intt    = OpTypeInt 32 1
%ptrt    = OpTypePointer Function %intt
%one     = OpConstant %intt 1
%two     = OpConstant %intt 2
%funct   = OpTypeFunction %voidt
%func    = OpFunction %voidt None %funct
%first   = OpLabel
           OpBranch %loop
%loop    = OpLabel
%varbad  = OpVariable %ptrt Function   ;Varaible not in first block
%cond    = OpSLessThan %intt %one %two
           OpLoopMerge %merge %cont None
           OpBranchConditional %cond %cont %merge
%cont    = OpLabel
           OpNop
           OpBranch %loop
%merge   = OpLabel
           OpNop
           OpReturn
           OpFunctionEnd
  )";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
}

TEST_F(ValidateCFG, NonDominantContinueConstruct) {
  string str = R"(
           OpCapability Shader
           OpMemoryModel Logical GLSL450
           OpName %func "Main"
%voidt   = OpTypeVoid
%boolt   = OpTypeBool
%intt    = OpTypeInt 32 1
%one     = OpConstant %intt 1
%two     = OpConstant %intt 2
%funct   = OpTypeFunction %voidt
%func    = OpFunction %voidt None %funct

%first   = OpLabel
           OpBranch %loop

%cont    = OpLabel
           OpNop
           OpBranch %loop

%loop    = OpLabel
%cond    = OpSLessThan %intt %one %two
           OpLoopMerge %merge %cont None
           OpBranchConditional %cond %cont %merge

%merge   = OpLabel
           OpNop
           OpReturn

           OpFunctionEnd
  )";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, MergeBlockTargetedByMultipleHeaderBlocksBad) {
    string str = R"(
           OpCapability Shader
           OpMemoryModel Logical GLSL450
           OpName %func "Main"
%voidt   = OpTypeVoid
%boolt   = OpTypeBool
%intt    = OpTypeInt 32 1
%one     = OpConstant %intt 1
%two     = OpConstant %intt 2
%funct   = OpTypeFunction %voidt
%func    = OpFunction %voidt None %funct

%first   = OpLabel
           OpBranch %loop

%loop    = OpLabel
%cond2   = OpSLessThan %intt %one %two
           OpLoopMerge %merge1 %cont None
           OpBranch %badhead

%badhead = OpLabel
%cond1   = OpSLessThan %intt %one %two
           OpSelectionMerge %merge1 None    ; cannot share the same merge
           OpBranchConditional %cond1 %t %f

%t       = OpLabel
           OpBranch %merge1
%f       = OpLabel
           OpBranch %cont

%cont    = OpLabel
           OpNop
           OpBranch %loop

%merge1  = OpLabel
           OpNop
           OpBranch %end

%end     = OpLabel
           OpReturn

           OpFunctionEnd
  )";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
}


// TODO(umar): Test optional instructions
}
