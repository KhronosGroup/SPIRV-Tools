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

#include <gmock/gmock.h>

#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"

#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using MergeReturnPassTest = PassTest<::testing::Test>;

TEST_F(MergeReturnPassTest, OneReturn) {
  const std::string before =
R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %1 "simple_kernel"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%1 = OpFunction %2 None %3
%4 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after = before;

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::MergeReturnPass>(before, after, false, false);
}

TEST_F(MergeReturnPassTest, TwoReturnsNoValue) {
  const std::string before =
R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %1 "simple_kernel"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %5
%7 = OpLabel
OpBranchConditional %4 %8 %9
%8 = OpLabel
OpReturn
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %1 "simple_kernel"
%2 = OpTypeVoid
%3 = OpTypeBool
%4 = OpConstantFalse %3
%1 = OpTypeFunction %2
%6 = OpFunction %2 None %5
%7 = OpLabel
OpBranchConditional %4 %8 %9
%8 = OpLabel
OpBranch %10
%9 = OpLabel
OpBranch %10
%10 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::MergeReturnPass>(before, after, false, false);
}

TEST_F(MergeReturnPassTest, TwoReturnsWithValues) {
  const std::string before =
R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeBool
%3 = OpConstantFalse %2
%4 = OpConstant %1 0
%5 = OpConstant %1 1
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
OpBranchConditional %3 %9 %10
%9 = OpLabel
OpReturnValue %4
%10 = OpLabel
OpReturnValue %5
OpFunctionEnd
)";

  const std::string after =
R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeBool
%3 = OpConstantFalse %2
%4 = OpConstant %1 0
%5 = OpConstant %1 1
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
OpBranchConditional %3 %9 %10
%9 = OpLabel
OpBranch %11
%10 = OpLabel
OpBranch %11
%11 = OpLabel
%12 = OpPhi %1 %4 %9 %5 %10
OpReturnValue %12
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::MergeReturnPass>(before, after, false, false);
}

} // anonymous namespace
