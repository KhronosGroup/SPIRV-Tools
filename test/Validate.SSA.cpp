// Copyright (c) 2015 The Khronos Group Inc.
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

// Validation tests for SSA

#include "UnitSPIRV.h"
#include "ValidateFixtures.h"

#include <regex>

using std::string;
using std::regex;
using std::pair;
using std::regex_replace;
namespace {

using Validate = spvtest::ValidateBase<pair<string, bool>>;

TEST_F(Validate, Default) {
  char str[] = R"(
     OpMemoryModel Logical GLSL450
     OpEntryPoint GLCompute %3 ""
     OpExecutionMode %3 LocalSize 1 1 1
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, IdUndefinedBad) {
  char str[] = R"(
     OpMemoryModel Logical GLSL450
     OpEntryPoint GLCompute %4 ""
     OpExecutionMode %4 LocalSize 1 1 1
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%4 = OpFunction %2 None %6
%5 = OpLabel
     OpReturn
     OpFunctionEnd
    )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, IdRedefinedBad) {
  char str[] = R"(
     OpMemoryModel Logical GLSL450
     OpEntryPoint GLCompute %3 ""
     OpExecutionMode %3 LocalSize 1 1 1
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%2 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, DominateUsageBad) {
  char str[] = R"(
     OpMemoryModel Logical GLSL450
     OpEntryPoint GLCompute %3 ""
     OpExecutionMode %3 LocalSize 1 1 1
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%1 = OpTypeVoid
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardNameGood) {
  char str[] = R"(
     OpMemoryModel Logical GLSL450
     OpEntryPoint GLCompute %3 ""
     OpExecutionMode %3 LocalSize 1 1 1
     OpName %3 "main"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, ForwardNameMissingLabelBad) {
  char str[] = R"(
     OpMemoryModel Logical GLSL450
     OpEntryPoint GLCompute %3 ""
     OpExecutionMode %3 LocalSize 1 1 1
     OpName %5 "main"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardMemberNameGood) {
  char str[] = R"(
          OpMemberName %struct 0 "value"
          OpMemberName %struct 1 "size"
%intt   = OpTypeInt 32 1
%uintt  = OpTypeInt 32 0
%struct = OpTypeStruct %intt %uintt
)";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, ForwardMemberNameMissingLabelBad) {
  char str[] = R"(
          OpMemberName %struct 0 "value"
          OpMemberName %bad 1 "size"
%intt   = OpTypeInt 32 1
%uintt  = OpTypeInt 32 0
%struct = OpTypeStruct %intt %uintt
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardDecorateGood) {
  char str[] = R"(
          OpDecorate %var Restrict
%intt   = OpTypeInt 32 1
%ptrt   = OpTypePointer UniformConstant %intt
%var    = OpVariable %ptrt UniformConstant
)";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, ForwardDecorateInvalidIDBad) {
  char str[] = R"(
          OpMemoryModel Logical GLSL450
          OpEntryPoint GLCompute %3 ""
          OpExecutionMode %3 LocalSize 1 1 1
          OpDecorate %missing Restrict
%voidt  = OpTypeVoid
%intt   = OpTypeInt 32 1
%ptrt   = OpTypePointer UniformConstant %intt
%var    = OpVariable %ptrt UniformConstant
%2      = OpTypeFunction %voidt
%3      = OpFunction %voidt None %2
%4      = OpLabel
          OpReturn
          OpFunctionEnd
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardMemberDecorateGood) {
  char str[] = R"(
          OpCapability Matrix
          OpMemberDecorate %struct 1 RowMajor
%intt   = OpTypeInt 32 1
%vec3   = OpTypeVector %intt 3
%mat33  = OpTypeMatrix %vec3 3
%struct = OpTypeStruct %intt %mat33
)";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, ForwardMemberDecorateInvalidIdBad) {
  char str[] = R"(
          OpCapability Matrix
          OpMemberDecorate %missing 1 RowMajor
%intt   = OpTypeInt 32 1
%vec3   = OpTypeVector %intt 3
%mat33  = OpTypeMatrix %vec3 3
%struct = OpTypeStruct %intt %mat33
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardGroupDecorateGood) {
  char str[] = R"(
          OpCapability Matrix
          OpDecorate %dgrp RowMajor
%dgrp   = OpDecorationGroup
          OpGroupDecorate %dgrp %mat33 %mat44
%intt   = OpTypeInt 32 1
%vec3   = OpTypeVector %intt 3
%vec4   = OpTypeVector %intt 4
%mat33  = OpTypeMatrix %vec3 3
%mat44  = OpTypeMatrix %vec4 4
)";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, ForwardGroupDecorateMissingIdBad) {
  char str[] = R"(
          OpCapability Matrix
          OpDecorate %dgrp RowMajor
%dgrp   = OpDecorationGroup
          OpGroupDecorate %dgrp %missing %mat44
%intt   = OpTypeInt 32 1
%vec3   = OpTypeVector %intt 3
%vec4   = OpTypeVector %intt 4
%mat33  = OpTypeMatrix %vec3 3
%mat44  = OpTypeMatrix %vec4 4
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardGroupDecorateDecorationGroupDominateBad) {
  char str[] = R"(
          OpCapability Matrix
          OpDecorate %dgrp RowMajor
          OpGroupDecorate %dgrp %mat33 %mat44
%dgrp   = OpDecorationGroup
%intt   = OpTypeInt 32 1
%vec3   = OpTypeVector %intt 3
%vec4   = OpTypeVector %intt 4
%mat33  = OpTypeMatrix %vec3 3
%mat44  = OpTypeMatrix %vec4 4
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardDecorateInvalidIdBad) {
  char str[] = R"(
          OpMemoryModel Logical GLSL450
          OpEntryPoint GLCompute %3 ""
          OpExecutionMode %3 LocalSize 1 1 1
          OpDecorate %missing Restrict
%voidt  = OpTypeVoid
%intt   = OpTypeInt 32 1
%ptrt   = OpTypePointer UniformConstant %intt
%var    = OpVariable %ptrt UniformConstant
%2      = OpTypeFunction %voidt
%3      = OpFunction %voidt None %2
%4      = OpLabel
          OpReturn
          OpFunctionEnd
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardFunctionCallGood) {
  char str[] = R"(
        OpMemoryModel Logical GLSL450
        OpEntryPoint GLCompute %5 ""
        OpExecutionMode %5 LocalSize 1 1 1
%1    = OpTypeVoid
%2    = OpTypeInt 32 1
%3    = OpTypeInt 32 0
%4    = OpTypeFunction %1
%5    = OpFunction %1 None %4
%6    = OpLabel
%four = OpConstant %2 4
%five = OpConstant %3 5
%7    = OpFunctionCall %1 %9 %four %five
        OpFunctionEnd
%8    = OpTypeFunction %1 %2 %3
%9    = OpFunction %1 None %8
%10   = OpFunctionParameter %2
%11   = OpFunctionParameter %3
%12   = OpLabel
        OpReturn
        OpFunctionEnd
)";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, DISABLED_ForwardBranchConditionalGood) {
  char str[] = R"(
%voidt  = OpTypeVoid
%boolt  = OpTypeBool
%vfunct = OpTypeFunction %voidt
%main   = OpFunction %voidt None %vfunct
%true   = OpConstantTrue %boolt
          OpSelectionMerge %endl None
          OpBranchConditional %true %truel %falsel
%truel  = OpLabel
          OpNop
          OpBranch %endl
%falsel = OpLabel
          OpNop
%endl    = OpLabel
          OpReturn
          OpFunctionEnd
)";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, DISABLED_ForwardBranchConditionalWithWeightsGood) {
  char str[] = R"(
%voidt  = OpTypeVoid
%boolt  = OpTypeBool
%vfunct = OpTypeFunction %voidt
%main   = OpFunction %voidt None %vfunct
%true   = OpConstantTrue %boolt
          OpSelectionMerge %endl None
          OpBranchConditional %true %truel %falsel !1 !9
%truel  = OpLabel
          OpNop
          OpBranch %endl
%falsel = OpLabel
          OpNop
%endl   = OpLabel
          OpReturn
          OpFunctionEnd
)";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, ForwardBranchConditionalNonDominantConditionBad) {
  char str[] = R"(
%voidt  = OpTypeVoid
%boolt  = OpTypeBool
%vfunct = OpTypeFunction %voidt
%main   = OpFunction %voidt None %vfunct
          OpSelectionMerge %endl None
          OpBranchConditional %true %missing %falsel
%truel  = OpLabel
          OpNop
          OpBranch %endl
%falsel = OpLabel
          OpNop
%endl   = OpLabel
%true   = OpConstantTrue %boolt
          OpReturn
          OpFunctionEnd
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardBranchConditionalMissingLabelBad) {
  char str[] = R"(
%voidt  = OpTypeVoid
%boolt  = OpTypeBool
%vfunct = OpTypeFunction %voidt
%main   = OpFunction %voidt None %vfunct
%true   = OpConstantTrue %boolt
          OpSelectionMerge %endl None
          OpBranchConditional %true %missing %falsel
%truel  = OpLabel
          OpNop
          OpBranch %endl
%falsel = OpLabel
          OpNop
%endl   = OpLabel
          OpReturn
          OpFunctionEnd
)";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

const string kBasicTypes = R"(
%voidt  = OpTypeVoid
%boolt  = OpTypeBool
%int8t  = OpTypeInt 8 0
%intt   = OpTypeInt 32 1
%uintt  = OpTypeInt 32 0
%vfunct = OpTypeFunction %voidt
)";

const string kKernelTypes = R"(
          OpCapability DeviceEnqueue
%queuet = OpTypeQueue

%three  = OpConstant %uintt 3
%arr3t  = OpTypeArray %intt %three
%ndt    = OpTypeStruct %intt %arr3t %arr3t %arr3t

%eventt = OpTypeEvent
%intptrt = OpTypePointer UniformConstant %int8t

)";

const string kKernelSetup = R"(
%dqueue = OpGetDefaultQueue %queuet

%offset = OpConstant %intt 0
%local  = OpConstant %intt 1
%gl     = OpConstant %intt 1
%ndval  = OpBuildNDRange %ndt %gl %local %offset

%nevent = OpConstant %intt 0
%event  = OpConstantNull %eventt
%revent = OpUndef %eventt

%firstp = OpConstant %int8t 0
%psize  = OpConstant %intt 0
%palign = OpConstant %intt 32
%lsize  = OpConstant %intt 1
)";

const string kKernelDefinition = R"(
%kfunct = OpTypeFunction %voidt %intptrt
%kfunc  = OpFunction %voidt None %kfunct
%iparam = OpFunctionParameter %intptrt
        OpNop
        OpReturn
        OpFunctionEnd
)";

TEST_F(Validate, DISABLED_EnqueueKernelGood) {
  string str = kBasicTypes + kKernelTypes + kKernelDefinition + R"(
                    %main   = OpFunction %voidt None %vfunct
                    )" +
               kKernelSetup + R"(
                    %err    = OpEnqueueKernel %uintt %dqueue %ndval %nevent %event %revent %kfunc %firstp %psize %palign %lsize
                             OpReturn
                             OpFunctionEnd
                     )";
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, DISABLED_ForwardEnqueueKernelGood) {
  string str = kBasicTypes + kKernelTypes + R"(
                    %main   = OpFunction %voidt None %vfunct
                    )" +
               kKernelSetup + R"(
                    %err    = OpEnqueueKernel %uintt %dqueue %ndval %nevent %event %revent %kfunc %firstp %psize %palign %lsize
                             OpReturn
                             OpFunctionEnd
                     )" +
               kKernelDefinition;
  ValidateInstructions(str, SPV_SUCCESS);
}

TEST_F(Validate, EnqueueMissingFunctionBad) {
  string str = kBasicTypes + kKernelTypes + R"(
                    %main   = OpFunction %voidt None %vfunct
                    )" +
               kKernelSetup + R"(
                    %err    = OpEnqueueKernel %uintt %dqueue %ndval %nevent %event %revent %kfunc %firstp %psize %palign %lsize
                             OpReturn
                             OpFunctionEnd
                     )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

string forwardKernelNonDominantParameterBaseCode = kBasicTypes + kKernelTypes +
                                                   kKernelDefinition +
                                                   R"(
                    %main   = OpFunction %voidt None %vfunct
                    )" + kKernelSetup;

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter1Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                    %err    = OpEnqueueKernel %uintt2 %dqueue %ndval %nevent %event %revent %kfunc %firstp %psize %palign %lsize
                              OpReturn
                              OpFunctionEnd
                    %uintt2 = OpTypeInt 32 0
                    )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter2Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                    %err     = OpEnqueueKernel %uintt %dqueue2 %ndval %nevent %event %revent %kfunc %firstp %psize %palign %lsize
                    %dqueue2 = OpGetDefaultQueue %queuet
                               OpReturn
                               OpFunctionEnd
                    )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter3Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                    %err    = OpEnqueueKernel %uintt %dqueue %ndval2 %nevent %event %revent %kfunc %firstp %psize %palign %lsize
                    %ndval2  = OpBuildNDRange %ndt %gl %local %offset
                              OpReturn
                              OpFunctionEnd
                    )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter4Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                  %err    = OpEnqueueKernel %uintt %dqueue %ndval %nevent2 %event %revent %kfunc %firstp %psize %palign %lsize
                  %nevent2 = OpConstant %intt 0
                            OpReturn
                            OpFunctionEnd
                  )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter5Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                  %err     = OpEnqueueKernel %uintt %dqueue %ndval %nevent %event2 %revent %kfunc %firstp %psize %palign %lsize
                  %event2  = OpConstantNull %eventt
                             OpReturn
                             OpFunctionEnd
                  )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter6Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                  %err     = OpEnqueueKernel %uintt %dqueue %ndval %nevent %event %revent2 %kfunc %firstp %psize %palign %lsize
                  %revent2 = OpUndef %eventt
                             OpReturn
                             OpFunctionEnd
                  )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter8Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                  %err     = OpEnqueueKernel %uintt %dqueue %ndval %nevent %event %revent %kfunc %firstp2 %psize %palign %lsize
                  %firstp2 = OpConstant %int8t 0
                             OpReturn
                             OpFunctionEnd
                  )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter9Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                  %err    = OpEnqueueKernel %uintt %dqueue %ndval %nevent %event %revent %kfunc %firstp %psize2 %palign %lsize
                  %psize2 = OpConstant %intt 0
                            OpReturn
                            OpFunctionEnd
                  )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter10Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                  %err     = OpEnqueueKernel %uintt %dqueue %ndval %nevent %event %revent %kfunc %firstp %psize %palign2 %lsize
                  %palign2 = OpConstant %intt 32
                            OpReturn
                            OpFunctionEnd
                  )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter11Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                  %err     = OpEnqueueKernel %uintt %dqueue %ndval %nevent %event %revent %kfunc %firstp %psize %palign %lsize2
                  %lsize2  = OpConstant %intt 1
                             OpReturn
                             OpFunctionEnd
                  )";
  ValidateInstructions(str, SPV_ERROR_INVALID_ID);
}

static const bool kWithNDrange = true;
static const bool kNoNDrange = false;
pair<string, bool> cases[] = {
    {"OpGetKernelNDrangeSubGroupCount", kWithNDrange},
    {"OpGetKernelNDrangeMaxSubGroupSize", kWithNDrange},
    {"OpGetKernelWorkGroupSize", kNoNDrange},
    {"OpGetKernelPreferredWorkGroupSizeMultiple", kNoNDrange}};

INSTANTIATE_TEST_CASE_P(KernelArgs, Validate, ::testing::ValuesIn(cases));

string SetOps(string &str, pair<string, bool> param) {
  regex placeholder("GET_KERNEL_OP");
  string out = regex_replace(str, placeholder, param.first);
  string out2;

  if (!param.second) {
    regex ndrange_regex("[^\n]%ndval");
    out2 = regex_replace(out, ndrange_regex, "");
  } else {
    out2 = out;
  }
  return out2;
}

TEST_P(Validate, DISABLED_GetKernelGood) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                  %numsg   = GET_KERNEL_OP %uintt %ndval %kfunc %firstp %psize %palign
                             OpReturn
                             OpFunctionEnd
                  )";

  string out = SetOps(str, GetParam());
  ValidateInstructions(out, SPV_SUCCESS);
}

TEST_P(Validate, DISABLED_ForwardGetKernelGood) {
  string str = kBasicTypes + kKernelTypes +
               R"(
                %main    = OpFunction %voidt None %vfunct
                    )" +
               kKernelSetup + R"(
                %numsg   = GET_KERNEL_OP %uintt %ndval %kfunc %firstp %psize %palign
                           OpReturn
                           OpFunctionEnd
                )" +
               kKernelDefinition;

  string out = SetOps(str, GetParam());
  ValidateInstructions(out, SPV_SUCCESS);
}

TEST_P(Validate, ForwardGetKernelMissingDefinitionBad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %numsg   = GET_KERNEL_OP %uintt %ndval %missing %firstp %psize %palign
                         OpReturn
                         OpFunctionEnd
              )";
  string out = SetOps(str, GetParam());
  ValidateInstructions(out, SPV_ERROR_INVALID_ID);
}

TEST_P(Validate, ForwardGetKernelNDrangeSubGroupCountNonDominantParameter1Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                  %numsg   = GET_KERNEL_OP %uintt2 %ndval %kfunc %firstp %psize %palign
                             OpReturn
                             OpFunctionEnd
                  %uintt2  = OpTypeInt 32 0
                  )";
  string out = SetOps(str, GetParam());
  ValidateInstructions(out, SPV_ERROR_INVALID_ID);
}

TEST_P(Validate, ForwardGetKernelNDrangeSubGroupCountNonDominantParameter2Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                %numsg   = GET_KERNEL_OP %uintt %ndval2 %kfunc %firstp %psize %palign
                %ndval2  = OpBuildNDRange %ndt %gl %local %offset
                            OpReturn
                            OpFunctionEnd
                )";
  if (GetParam().second) {
    regex placeholder("GET_KERNEL_OP");
    string out = regex_replace(str, placeholder, GetParam().first);
    ValidateInstructions(out, SPV_ERROR_INVALID_ID);
  }
}

TEST_P(Validate, ForwardGetKernelNDrangeSubGroupCountNonDominantParameter4Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                %numsg   = GET_KERNEL_OP %uintt %ndval %kfunc %firstp2 %psize %palign
                %firstp2 = OpConstant %int8t 0
                            OpReturn
                            OpFunctionEnd
                )";
  string out = SetOps(str, GetParam());
  ValidateInstructions(out, SPV_ERROR_INVALID_ID);
}

TEST_P(Validate, ForwardGetKernelNDrangeSubGroupCountNonDominantParameter5Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                %numsg   = GET_KERNEL_OP %uintt %ndval %kfunc %firstp %psize2 %palign
                %psize2  = OpConstant %intt 0
                            OpReturn
                            OpFunctionEnd
                )";
  string out = SetOps(str, GetParam());
  ValidateInstructions(out, SPV_ERROR_INVALID_ID);
}

TEST_P(Validate, ForwardGetKernelNDrangeSubGroupCountNonDominantParameter6Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                %numsg   = GET_KERNEL_OP %uintt %ndval %kfunc %firstp %psize %palign2
                %palign2 = OpConstant %intt 32
                           OpReturn
                           OpFunctionEnd
                )";
  if (GetParam().second) {
    string out = SetOps(str, GetParam());
    ValidateInstructions(out, SPV_ERROR_INVALID_ID);
  }
}

// TODO(umar): OpPhi
// TODO(umar): OpGroupMemberDecorate
// TODO(umar): OpBranch
}
