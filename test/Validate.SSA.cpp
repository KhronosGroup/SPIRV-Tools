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

using Validate =
    spvtest::ValidateBase<pair<string, bool>, SPV_VALIDATE_SSA_BIT>;

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
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, IdUndefinedBad) {
  char str[] = R"(
          OpMemoryModel Logical GLSL450
%voidt  = OpTypeVoid
%vfunct = OpTypeFunction %voidt
%func   = OpFunction %2 None %missing
%flabel = OpLabel
          OpReturn
          OpFunctionEnd
    )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, IdRedefinedBad) {
  char str[] = R"(
     OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%2 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, DominateUsageBad) {
  char str[] = R"(
     OpMemoryModel Logical GLSL450
%2 = OpTypeFunction %1              ; uses %1 before it's definition
%1 = OpTypeVoid
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardNameGood) {
  char str[] = R"(
     OpMemoryModel Logical GLSL450
     OpName %3 "main"
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, ForwardNameMissingTargetBad) {
  char str[] = R"(
      OpMemoryModel Logical GLSL450
      OpName %5 "main"              ; Target never defined
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardMemberNameGood) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpMemberName %struct 0 "value"
           OpMemberName %struct 1 "size"
%intt   =  OpTypeInt 32 1
%uintt  =  OpTypeInt 32 0
%struct =  OpTypeStruct %intt %uintt
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, ForwardMemberNameMissingTargetBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpMemberName %struct 0 "value"
           OpMemberName %bad 1 "size"     ; Target is not defined
%intt   =  OpTypeInt 32 1
%uintt  =  OpTypeInt 32 0
%struct =  OpTypeStruct %intt %uintt
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardDecorateGood) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpDecorate %var Restrict
%intt   =  OpTypeInt 32 1
%ptrt   =  OpTypePointer UniformConstant %intt
%var    =  OpVariable %ptrt UniformConstant
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, ForwardDecorateInvalidIDBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpDecorate %missing Restrict        ;Missing ID
%voidt  =  OpTypeVoid
%intt   =  OpTypeInt 32 1
%ptrt   =  OpTypePointer UniformConstant %intt
%var    =  OpVariable %ptrt UniformConstant
%2      =  OpTypeFunction %voidt
%3      =  OpFunction %voidt None %2
%4      =  OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardMemberDecorateGood) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpCapability Matrix
           OpMemberDecorate %struct 1 RowMajor
%intt   =  OpTypeInt 32 1
%vec3   =  OpTypeVector %intt 3
%mat33  =  OpTypeMatrix %vec3 3
%struct =  OpTypeStruct %intt %mat33
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, ForwardMemberDecorateInvalidIdBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpCapability Matrix
           OpMemberDecorate %missing 1 RowMajor ; Target not defined
%intt   =  OpTypeInt 32 1
%vec3   =  OpTypeVector %intt 3
%mat33  =  OpTypeMatrix %vec3 3
%struct =  OpTypeStruct %intt %mat33
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardGroupDecorateGood) {
  char str[] = R"(
          OpMemoryModel Logical GLSL450
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
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

  TEST_F(Validate, ForwardGroupDecorateMissingGroupBad) {
    char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpCapability Matrix
           OpDecorate %dgrp RowMajor
%dgrp   =  OpDecorationGroup
           OpGroupDecorate %missing %mat33 %mat44 ; Target not defined
%intt   =  OpTypeInt 32 1
%vec3   =  OpTypeVector %intt 3
%vec4   =  OpTypeVector %intt 4
%mat33  =  OpTypeMatrix %vec3 3
%mat44  =  OpTypeMatrix %vec4 4
)";
    CompileSuccessfully(str);
    ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardGroupDecorateMissingTargetBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpCapability Matrix
           OpDecorate %dgrp RowMajor
%dgrp   =  OpDecorationGroup
           OpGroupDecorate %dgrp %missing %mat44 ; Target not defined
%intt   =  OpTypeInt 32 1
%vec3   =  OpTypeVector %intt 3
%vec4   =  OpTypeVector %intt 4
%mat33  =  OpTypeMatrix %vec3 3
%mat44  =  OpTypeMatrix %vec4 4
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardGroupDecorateDecorationGroupDominateBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpCapability Matrix
           OpDecorate %dgrp RowMajor
           OpGroupDecorate %dgrp %mat33 %mat44 ; Decoration group does not dominate usage
%dgrp   =  OpDecorationGroup
%intt   =  OpTypeInt 32 1
%vec3   =  OpTypeVector %intt 3
%vec4   =  OpTypeVector %intt 4
%mat33  =  OpTypeMatrix %vec3 3
%mat44  =  OpTypeMatrix %vec4 4
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardDecorateInvalidIdBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpDecorate %missing Restrict        ; Missing target
%voidt  =  OpTypeVoid
%intt   =  OpTypeInt 32 1
%ptrt   =  OpTypePointer UniformConstant %intt
%var    =  OpVariable %ptrt UniformConstant
%2      =  OpTypeFunction %voidt
%3      =  OpFunction %voidt None %2
%4      =  OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, FunctionCallGood) {
    char str[] = R"(
         OpMemoryModel Logical GLSL450
%1    =  OpTypeVoid
%2    =  OpTypeInt 32 1
%3    =  OpTypeInt 32 0
%4    =  OpTypeFunction %1
%8    =  OpTypeFunction %1 %2 %3
%four =  OpConstant %2 4
%five =  OpConstant %3 5
%9    =  OpFunction %1 None %8
%10   =  OpFunctionParameter %2
%11   =  OpFunctionParameter %3
%12   =  OpLabel
         OpReturn
         OpFunctionEnd
%5    =  OpFunction %1 None %4
%6    =  OpLabel
%7    =  OpFunctionCall %1 %9 %four %five
         OpFunctionEnd
)";
    CompileSuccessfully(str);
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }

TEST_F(Validate, ForwardFunctionCallGood) {
  char str[] = R"(
         OpMemoryModel Logical GLSL450
%1    =  OpTypeVoid
%2    =  OpTypeInt 32 1
%3    =  OpTypeInt 32 0
%four =  OpConstant %2 4
%five =  OpConstant %3 5
%4    =  OpTypeFunction %1
%5    =  OpFunction %1 None %4
%6    =  OpLabel
%7    =  OpFunctionCall %1 %9 %four %five
         OpFunctionEnd
%8    =  OpTypeFunction %1 %2 %3
%9    =  OpFunction %1 None %8
%10   =  OpFunctionParameter %2
%11   =  OpFunctionParameter %3
%12   =  OpLabel
         OpReturn
         OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, ForwardBranchConditionalGood) {
  char str[] = R"(
            OpMemoryModel Logical GLSL450
%voidt  =   OpTypeVoid
%boolt  =   OpTypeBool
%vfunct =   OpTypeFunction %voidt
%true   =   OpConstantTrue %boolt
%main   =   OpFunction %voidt None %vfunct
            OpSelectionMerge %endl None
            OpBranchConditional %true %truel %falsel
%truel  =   OpLabel
            OpNop
            OpBranch %endl
%falsel =   OpLabel
            OpNop
%endl    =  OpLabel
            OpReturn
            OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, ForwardBranchConditionalWithWeightsGood) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
%voidt  =  OpTypeVoid
%boolt  =  OpTypeBool
%vfunct =  OpTypeFunction %voidt
%true   =  OpConstantTrue %boolt
%main   =  OpFunction %voidt None %vfunct
           OpSelectionMerge %endl None
           OpBranchConditional %true %truel %falsel 1 9
%truel  =  OpLabel
           OpNop
           OpBranch %endl
%falsel =  OpLabel
           OpNop
%endl   =  OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, ForwardBranchConditionalNonDominantConditionBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
%voidt  =  OpTypeVoid
%boolt  =  OpTypeBool
%vfunct =  OpTypeFunction %voidt
%true   =  OpConstantTrue %boolt
%main   =  OpFunction %voidt None %vfunct
           OpSelectionMerge %endl None
           OpBranchConditional %tcpy %truel %falsel ;
%truel  =  OpLabel
           OpNop
           OpBranch %endl
%falsel =  OpLabel
           OpNop
%endl   =  OpLabel
%tcpy   =  OpCopyObject %boolt %true
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardBranchConditionalMissingTargetBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
%voidt  =  OpTypeVoid
%boolt  =  OpTypeBool
%vfunct =  OpTypeFunction %voidt
%true   =  OpConstantTrue %boolt
%main   =  OpFunction %voidt None %vfunct
           OpSelectionMerge %endl None
           OpBranchConditional %true %missing %falsel
%truel  =  OpLabel
           OpNop
           OpBranch %endl
%falsel =  OpLabel
           OpNop
%endl   =  OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

const string kBasicTypes = R"(
           OpMemoryModel Logical GLSL450
           OpCapability Int8
%voidt  =  OpTypeVoid
%boolt  =  OpTypeBool
%int8t  =  OpTypeInt 8 0
%intt   =  OpTypeInt 32 1
%uintt  =  OpTypeInt 32 0
%vfunct =  OpTypeFunction %voidt
)";

const string kKernelTypesAndConstants = R"(
           OpCapability DeviceEnqueue
%queuet  = OpTypeQueue

%three   = OpConstant %uintt 3
%arr3t   = OpTypeArray %intt %three
%ndt     = OpTypeStruct %intt %arr3t %arr3t %arr3t

%eventt  = OpTypeEvent
%intptrt = OpTypePointer UniformConstant %int8t

%offset = OpConstant %intt 0
%local  = OpConstant %intt 1
%gl     = OpConstant %intt 1

%nevent = OpConstant %intt 0
%event  = OpConstantNull %eventt

%firstp = OpConstant %int8t 0
%psize  = OpConstant %intt 0
%palign = OpConstant %intt 32
%lsize  = OpConstant %intt 1
%flags  = OpConstant %intt 0 ; NoWait

%kfunct = OpTypeFunction %voidt %intptrt
)";

const string kKernelSetup = R"(
%dqueue = OpGetDefaultQueue %queuet
%ndval  = OpBuildNDRange %ndt %gl %local %offset
%revent = OpUndef %eventt

)";

const string kKernelDefinition = R"(
%kfunc  = OpFunction %voidt None %kfunct
%iparam = OpFunctionParameter %intptrt
          OpNop
          OpReturn
          OpFunctionEnd
)";

TEST_F(Validate, EnqueueKernelGood) {
  string str = kBasicTypes + kKernelTypesAndConstants + kKernelDefinition + R"(
                %main   = OpFunction %voidt None %vfunct
                )" +
               kKernelSetup + R"(
                %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize
                          OpReturn
                          OpFunctionEnd
                 )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelGood) {
  string str = kBasicTypes + kKernelTypesAndConstants + R"(
                %main   = OpFunction %voidt None %vfunct
                )" +
               kKernelSetup + R"(
                %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize
                         OpReturn
                         OpFunctionEnd
                 )" +
               kKernelDefinition;
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(Validate, EnqueueMissingFunctionBad) {
  string str = kBasicTypes + kKernelTypesAndConstants + R"(
                %main   = OpFunction %voidt None %vfunct
                )" +
               kKernelSetup + R"(
                %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize
                         OpReturn
                         OpFunctionEnd
                 )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

string forwardKernelNonDominantParameterBaseCode = kBasicTypes + kKernelTypesAndConstants +
                                                   kKernelDefinition +
                                                   R"(
                %main   = OpFunction %voidt None %vfunct
                )" + kKernelSetup;

TEST_F(Validate, ForwardEnqueueKernelMissingParameter1Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                %err    = OpEnqueueKernel %missing %dqueue %flags %ndval
                                        %nevent %event %revent %kfunc %firstp
                                        %psize %palign %lsize
                          OpReturn
                          OpFunctionEnd
                )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter2Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                %err     = OpEnqueueKernel %uintt %dqueue2 %flags %ndval
                                            %nevent %event %revent %kfunc
                                            %firstp %psize %palign %lsize
                %dqueue2 = OpGetDefaultQueue %queuet
                           OpReturn
                           OpFunctionEnd
                )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter3Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
                %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval2
                                        %nevent %event %revent %kfunc %firstp
                                        %psize %palign %lsize
                %ndval2  = OpBuildNDRange %ndt %gl %local %offset
                          OpReturn
                          OpFunctionEnd
                )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter4Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent2
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize
              %nevent2 = OpCopyObject %intt %nevent
                        OpReturn
                        OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter5Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event2 %revent %kfunc %firstp %psize
                                        %palign %lsize
              %event2  = OpCopyObject %eventt %event
                         OpReturn
                         OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter6Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent2 %kfunc %firstp %psize
                                        %palign %lsize
              %revent2 = OpCopyObject %eventt %revent
                         OpReturn
                         OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter8Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp2 %psize
                                        %palign %lsize
              %firstp2 = OpCopyObject %int8t %firstp
                         OpReturn
                         OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter9Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %err    = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize2
                                        %palign %lsize
              %psize2 = OpCopyObject %intt %psize
                        OpReturn
                        OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter10Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign2 %lsize
              %palign2 = OpCopyObject %intt %palign
                        OpReturn
                        OpFunctionEnd
              )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_F(Validate, ForwardEnqueueKernelNonDominantParameter11Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %err     = OpEnqueueKernel %uintt %dqueue %flags %ndval %nevent
                                        %event %revent %kfunc %firstp %psize
                                        %palign %lsize2
              %lsize2  = OpCopyObject %intt %lsize
                         OpReturn
                         OpFunctionEnd
              )";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

static const bool kWithNDrange = true;
static const bool kNoNDrange = false;
pair<string, bool> cases[] = {
    {"OpGetKernelNDrangeSubGroupCount", kWithNDrange},
    {"OpGetKernelNDrangeMaxSubGroupSize", kWithNDrange},
    {"OpGetKernelWorkGroupSize", kNoNDrange},
    {"OpGetKernelPreferredWorkGroupSizeMultiple", kNoNDrange}};

INSTANTIATE_TEST_CASE_P(KernelArgs, Validate, ::testing::ValuesIn(cases));

string SetOps(const string &str, pair<string, bool> param) {
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

TEST_P(Validate, GetKernelGood) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %numsg   = GET_KERNEL_OP %uintt %ndval %kfunc %firstp %psize %palign
                         OpReturn
                         OpFunctionEnd
              )";

  string out = SetOps(str, GetParam());
  CompileSuccessfully(out);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(Validate, ForwardGetKernelGood) {
  string str = kBasicTypes + kKernelTypesAndConstants +
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
  CompileSuccessfully(out);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(Validate, ForwardGetKernelMissingDefinitionBad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
          %numsg   = GET_KERNEL_OP %uintt %ndval %missing %firstp %psize %palign
                     OpReturn
                     OpFunctionEnd
          )";
  string out = SetOps(str, GetParam());
  CompileSuccessfully(out);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_P(Validate, ForwardGetKernelNDrangeSubGroupCountMissingParameter1Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
              %numsg   = GET_KERNEL_OP %missing %ndval %kfunc %firstp %psize %palign
                         OpReturn
                         OpFunctionEnd
              )";
  string out = SetOps(str, GetParam());
  CompileSuccessfully(out);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
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
    CompileSuccessfully(out);
    ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  }
}

TEST_P(Validate, ForwardGetKernelNDrangeSubGroupCountNonDominantParameter4Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
            %numsg   = GET_KERNEL_OP %uintt %ndval %kfunc %firstp2 %psize %palign
            %firstp2 = OpCopyObject %int8t %firstp
                        OpReturn
                        OpFunctionEnd
            )";
  string out = SetOps(str, GetParam());
  CompileSuccessfully(out);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_P(Validate, ForwardGetKernelNDrangeSubGroupCountNonDominantParameter5Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
            %numsg   = GET_KERNEL_OP %uintt %ndval %kfunc %firstp %psize2 %palign
            %psize2  = OpCopyObject %intt %psize
                        OpReturn
                        OpFunctionEnd
            )";
  string out = SetOps(str, GetParam());
  CompileSuccessfully(out);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
}

TEST_P(Validate, ForwardGetKernelNDrangeSubGroupCountNonDominantParameter6Bad) {
  string str = forwardKernelNonDominantParameterBaseCode + R"(
            %numsg   = GET_KERNEL_OP %uintt %ndval %kfunc %firstp %psize %palign2
            %palign2 = OpCopyObject %intt %palign
                       OpReturn
                       OpFunctionEnd
            )";
  if (GetParam().second) {
    string out = SetOps(str, GetParam());
    CompileSuccessfully(out);
    ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  }
}

// TODO(umar): OpPhi
// TODO(umar): OpGroupMemberDecorate
// TODO(umar): OpBranch
}
