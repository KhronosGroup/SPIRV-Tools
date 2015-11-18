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

#include "UnitSPIRV.h"

namespace {

class Validate : public ::testing::Test {
 public:
  Validate() : context(spvContextCreate()), binary() {}
  ~Validate() { spvContextDestroy(context); }

  virtual void TearDown() { spvBinaryDestroy(binary); }
  spv_const_binary get_const_binary() { return spv_const_binary(binary); }

  spv_context context;
  spv_binary binary;
};

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
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS,
            spvTextToBinary(context, str, strlen(str), &binary, &diagnostic));
  ASSERT_EQ(SPV_SUCCESS, spvValidate(context, get_const_binary(),
                                     SPV_VALIDATE_ALL, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
}

TEST_F(Validate, InvalidIdUndefined) {
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
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS,
            spvTextToBinary(context, str, strlen(str), &binary, &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, spvValidate(context, get_const_binary(),
                                              SPV_VALIDATE_ALL, &diagnostic));
  ASSERT_NE(nullptr, diagnostic);
  spvDiagnosticPrint(diagnostic);
  spvDiagnosticDestroy(diagnostic);
}

TEST_F(Validate, InvalidIdRedefined) {
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
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS,
            spvTextToBinary(context, str, strlen(str), &binary, &diagnostic));
  // TODO: Fix setting of bound in spvTextTo, then remove this!
  ASSERT_EQ(SPV_ERROR_INVALID_ID, spvValidate(context, get_const_binary(),
                                              SPV_VALIDATE_ALL, &diagnostic));
  ASSERT_NE(nullptr, diagnostic);
  spvDiagnosticPrint(diagnostic);
  spvDiagnosticDestroy(diagnostic);
}

}  // anonymous namespace
