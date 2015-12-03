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

// Common validation fixtures for unit tests

#include "UnitSPIRV.h"
#include "ValidateFixtures.h"

namespace spvtest {

template <typename T>
ValidateBase<T>::ValidateBase()
    : context_(spvContextCreate()), binary_() {}

template <typename T>
ValidateBase<T>::~ValidateBase() {
  spvContextDestroy(context_);
}

template <typename T>
spv_const_binary ValidateBase<T>::get_const_binary() {
  return spv_const_binary(binary_);
}

template <typename T>
void ValidateBase<T>::TearDown() {
  spvBinaryDestroy(binary_);
}

template <typename T>
void ValidateBase<T>::ValidateInstructions(std::string code,
                                           spv_result_t result) {
  spv_diagnostic diagnostic = nullptr;
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(context_, code.c_str(), code.size(),
                                         &binary_, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
  if (result == SPV_SUCCESS) {
    EXPECT_EQ(result, spvValidate(context_, get_const_binary(),
                                  SPV_VALIDATE_ALL, &diagnostic));
    if (diagnostic) {
      spvDiagnosticPrint(diagnostic);
      spvDiagnosticDestroy(diagnostic);
    }
  } else {
    EXPECT_EQ(result, spvValidate(context_, get_const_binary(),
                                  SPV_VALIDATE_ALL, &diagnostic));
    ASSERT_NE(nullptr, diagnostic);
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
}

template class spvtest::ValidateBase<std::pair<std::string, bool>>;
}
