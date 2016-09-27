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

#include <gtest/gtest.h>
#include <sstream>
#include <utility>
#include <vector>

#include "pass_fixture.h"

namespace {

// Ignores all leading spaces of the given string |str| and appends it to the
// given |stream| with an additional EOL if |str| is non-empty after ignoring
// all leading spaces.
void StripAppend(std::string str, std::ostringstream* stream) {
  uint32_t pos = 0;
  while (pos < str.size() && str[pos] == ' ') ++pos;
  str = str.substr(pos);
  if (!str.empty()) *stream << str << "\n";
}

// A minimal struct for representing an instruction string for testing purpose.
// It contains a pair of strings: one for original instruction, the other for
// optimized instruction.
struct Inst {
  enum class For {
    kOriginal,
    kOptimized,
    kBoth,
  };

  Inst(std::string inst, For c = For::kBoth) {
    switch (c) {
      case For::kOriginal:
        original = std::move(inst);
        break;
      case For::kOptimized:
        optimized = std::move(inst);
        break;
      case For::kBoth:
        original = inst;
        optimized = std::move(inst);
        break;
    }
  }

  std::string original;
  std::string optimized;
};

// A minimal class for representing a SPIR-V module for testing purpose.
// It contains a vector of Inst instances.
class Module {
 public:
  explicit Module(std::vector<Inst> i) : insts_(std::move(i)) {}

  std::string GetOriginalCode() const {
    std::ostringstream oss;
    for (const auto& inst : insts_) StripAppend(inst.original, &oss);
    return oss.str();
  }
  std::string GetOptimizedCode() const {
    std::ostringstream oss;
    for (const auto& inst : insts_) StripAppend(inst.optimized, &oss);
    return oss.str();
  }
  std::vector<Inst>& insts() { return insts_; }

 private:
  std::vector<Inst> insts_;
};

using namespace spvtools;
using EliminateDeadCodeTest = PassTest<::testing::Test>;
#define TBK Inst::For::kOriginal

TEST_F(EliminateDeadCodeTest, EmptyCode) {
  SinglePassRunAndCheck<opt::EliminateDeadCodePass>("", "", true);
}

TEST_F(EliminateDeadCodeTest, DeadExtInstImport) {
  auto source = Module({
      {"%1 = OpExtInstImport \"GLSL.std.450\"", TBK},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, DeadTypes) {
  auto source = Module({
      {"%void = OpTypeVoid", TBK}, {" %int = OpTypeInt 32 1", TBK},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, DeadConstants) {
  auto source = Module({
      {"  %int = OpTypeInt 32 1", TBK},
      {"%float = OpTypeFloat 32", TBK},
      {"    %3 = OpConstant %int 1", TBK},
      {"    %4 = OpConstant %float 3.14", TBK},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, DeadGlobalVariables) {
  auto source = Module({
      {"%int = OpTypeInt 32 1"},
      {"%_ptr_UniformConstant_int = OpTypePointer UniformConstant %int"},
      {"  %3 = OpVariable %_ptr_UniformConstant_int UniformConstant"},  // 0
      {"%_ptr_Input_int = OpTypePointer Input %int"},
      {"  %5 = OpVariable %_ptr_Input_int Input"},  // 1
      {"%_ptr_Uniform_int = OpTypePointer Uniform %int"},
      {"  %7 = OpVariable %_ptr_Uniform_int Uniform"},  // 2
      {"%_ptr_Output_int = OpTypePointer Output %int"},
      {"  %9 = OpVariable %_ptr_Output_int Output"},  // 3
      {"%_ptr_Workgroup_int = OpTypePointer Workgroup %int"},
      {" %11 = OpVariable %_ptr_Workgroup_int Workgroup"},  // 4
      {"%_ptr_CrossWorkgroup_int = OpTypePointer CrossWorkgroup %int"},
      {" %13 = OpVariable %_ptr_CrossWorkgroup_int CrossWorkgroup"},  // 5
      {"%_ptr_PushConstant_int = OpTypePointer PushConstant %int"},
      {" %15 = OpVariable %_ptr_PushConstant_int PushConstant"},  // 9
      {"%_ptr_Private_int = OpTypePointer Private %int", TBK},
      {" %17 = OpVariable %_ptr_Private_int Private", TBK},  // 6
      {"%_ptr_Generic_int = OpTypePointer Generic %int", TBK},
      // Storage class Function (7) is tested in RecursiveDeadInstsInBasicBlock.
      {" %19 = OpVariable %_ptr_Generic_int Generic", TBK},  // 8
      {"%_ptr_AtomicCounter_int = OpTypePointer AtomicCounter %int", TBK},
      {" %21 = OpVariable %_ptr_AtomicCounter_int AtomicCounter", TBK},  // 10
      {"%_ptr_Image_int = OpTypePointer Image %int", TBK},
      {" %23 = OpVariable %_ptr_Image_int Image", TBK},  // 11
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, DeadInstsInBasicBlock) {
  auto source = Module({
      {"%int = OpTypeInt 32 1"},
      {"  %2 = OpTypeFunction %int"},
      {"  %3 = OpConstant %int 1"},
      {"  %4 = OpFunction %int None %2"},
      {"  %5 = OpLabel"},
      {"  %6 = OpUndef %int", TBK},
      {"  %7 = OpUndef %int", TBK},
      {"       OpReturnValue %3"},
      {"       OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, TypeOnlyUsedByDeadInst) {
  auto source = Module({
      {"%void = OpTypeVoid"},
      {" %int = OpTypeInt 32 1", TBK},  // only used by %6
      {"   %3 = OpTypeFunction %void"},
      {"   %4 = OpFunction %void None %3"},
      {"   %5 = OpLabel"},
      {"   %6 = OpUndef %int", TBK},
      {"        OpReturn"},
      {"        OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, TypeUsedByDeadInstAndLiveInst) {
  auto source = Module({
      {"%int = OpTypeInt 32 1"},
      {"  %2 = OpTypeFunction %int"},
      {"  %3 = OpConstant %int 1"},
      {"  %4 = OpFunction %int None %2"},
      {"  %5 = OpLabel"},
      {"  %6 = OpUndef %int", TBK},
      {"  %7 = OpUndef %int", TBK},
      {"       OpReturnValue %3"},
      {"       OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, RecursiveDeadInstsInBasicBlock) {
  auto source = Module({
      {"%int = OpTypeInt 32 1"},
      {" %ip = OpTypePointer Function %int", TBK},
      {"  %3 = OpTypeFunction %int"},
      {"  %4 = OpConstant %int 1"},
      {"  %5 = OpFunction %int None %3"},
      {"  %6 = OpLabel"},
      {" %p1 = OpVariable %ip Function", TBK},
      {" %p2 = OpVariable %ip Function", TBK},
      {" %v1 = OpLoad %int %p1", TBK},
      {" %v2 = OpLoad %int %p2", TBK},
      {"%sum = OpIAdd %int %v1 %v2", TBK},
      {"       OpReturnValue %4"},
      {"       OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, DeadInstsInMultiBasicBlock) {
  auto source = Module({
      {"%int = OpTypeInt 32 1"},
      {"%_ptr_Function_int = OpTypePointer Function %int"},  // used by %15
      {"  %3 = OpTypeFunction %int"},                        // used by %5
      {"  %4 = OpConstant %int 1"},                          // used by %15

      {"  %5 = OpFunction %int None %3"},

      {"  %6 = OpLabel"},
      {" %p1 = OpVariable %_ptr_Function_int Function", TBK},
      {" %p2 = OpVariable %_ptr_Function_int Function", TBK},
      {" %v1 = OpLoad %int %p1", TBK},
      {" %v2 = OpLoad %int %p2", TBK},
      {"%sum = OpIAdd %int %v1 %v2", TBK},
      {"       OpBranch %12"},

      {" %12 = OpLabel"},
      {" %ud = OpUndef %int", TBK},
      {"       OpBranch %14"},

      {" %14 = OpLabel"},
      {" %15 = OpVariable %_ptr_Function_int Function %4"},
      {" %16 = OpLoad %int %15"},
      {"       OpReturnValue %16"},

      {"       OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, DeadInstsInMultiFunctions) {
  auto source = Module({
      {"%int = OpTypeInt 32 1"},
      {"%_ptr_Function_int = OpTypePointer Function %int"},  // used by %17
      {"  %3 = OpTypeFunction %int"},                        // used by %15
      {"  %4 = OpConstant %int 1"},                          // used by %17

      {"  %5 = OpFunction %int None %3"},
      {"  %6 = OpLabel"},
      {" %p1 = OpVariable %_ptr_Function_int Function", TBK},
      {" %p2 = OpVariable %_ptr_Function_int Function", TBK},
      {" %v1 = OpLoad %int %p1", TBK},
      {" %v2 = OpLoad %int %p2", TBK},
      {"%sum = OpIAdd %int %v1 %v2", TBK},
      {"       OpReturnValue %4"},
      {"       OpFunctionEnd"},

      {" %12 = OpFunction %int None %3"},
      {" %13 = OpLabel"},
      {" %ud = OpUndef %int", TBK},
      {"       OpReturnValue %4"},
      {"       OpFunctionEnd"},

      {" %15 = OpFunction %int None %3"},
      {" %16 = OpLabel"},
      {" %17 = OpVariable %_ptr_Function_int Function %4"},
      {" %18 = OpLoad %int %17"},
      {"       OpReturnValue %18"},
      {"       OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, Phi) {
  auto source = Module({
      {" %void = OpTypeVoid"},
      {" %bool = OpTypeBool"},
      {"  %int = OpTypeInt 32 1"},
      {"%float = OpTypeFloat 32"},
      {"    %5 = OpTypeFunction %int"},
      {"    %6 = OpConstant %int 0"},
      {"    %7 = OpConstant %int 1"},
      {"    %8 = OpConstant %float 0"},
      {"    %9 = OpConstant %float 1"},

      {"   %10 = OpFunction %void None %5"},

      {"   %11 = OpLabel"},
      {"         OpBranch %12"},

      {"   %12 = OpLabel"},
      {"   %13 = OpPhi %int %6 %11 %14 %12"},    // %11 => 0,   %12 => %14
      {"   %15 = OpPhi %float %8 %11 %16 %12"},  // %11 => 0.0, %12 => %16
      {"   %14 = OpIAdd %int %13 %7"},           // %13 + 1
      {"   %16 = OpFAdd %float %15 %9"},         // %15 + 1.0
      {"   %17 = OpSLessThan %bool %13 %7"},     // %13 < 1 ?
      {"         OpLoopMerge %18 %12 None"},
      {"         OpBranchConditional %17 %12 %18"},

      {"   %18 = OpLabel"},
      {"         OpReturn"},

      {"         OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, VariableDecoration) {
  auto source = Module({
      {"        OpDecorate %1 SpecId 10"},
      {"        OpDecorate %2 RelaxedPrecision"},
      {"   %2 = OpDecorationGroup"},
      {"        OpGroupDecorate %2 %3 %4"},
      {"%void = OpTypeVoid"},
      {" %int = OpTypeInt 32 1"},
      {"   %1 = OpSpecConstant %int 0"},  // cannot kill this
      {"   %7 = OpTypeFunction %void"},
      {"   %8 = OpFunction %void None %7"},
      {"   %9 = OpLabel"},
      {"   %3 = OpUndef %int"},  // TODO(antiagainst): this can be killed.
      {"   %4 = OpUndef %int"},  // TODO(antiagainst): this can be killed.
      {"        OpReturn"},
      {"        OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, TypeDecoration) {
  // TODO(antiagainst): this can be killed
  auto source = Module({
      {"              OpMemberDecorate %_struct_1 0 Offset 0"},
      {"       %int = OpTypeInt 32 1"},
      {" %_struct_1 = OpTypeStruct %int %int"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, Debug) {
  auto source = Module({
      {"        OpName %i32 \"i32\"", TBK},
      {"        OpName %ud \"ud\"", TBK},
      {"%void = OpTypeVoid"},
      {" %i32 = OpTypeInt 32 1", TBK},
      {"   %4 = OpTypeFunction %void"},
      {"   %5 = OpFunction %void None %4"},
      {"   %6 = OpLabel"},
      {"  %ud = OpUndef %i32", TBK},
      {"        OpReturn"},
      {"        OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

TEST_F(EliminateDeadCodeTest, DebugAndDecoration) {
  auto source = Module({
      {"        OpName %ud \"ud\""},
      {"        OpDecorate %ud RelaxedPrecision"},
      {"%void = OpTypeVoid"},
      {" %int = OpTypeInt 32 1"},
      {"   %4 = OpTypeFunction %void"},
      {"   %5 = OpFunction %void None %4"},
      {"   %6 = OpLabel"},
      {"  %ud = OpUndef %int"},  // TODO(antiagainst): this can be killed.
      {"        OpReturn"},
      {"        OpFunctionEnd"},
  });

  SinglePassRunAndCheck<opt::EliminateDeadCodePass>(
      source.GetOriginalCode(), source.GetOptimizedCode(), true);
}

}  // anonymous namespace
