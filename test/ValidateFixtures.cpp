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

// Common validation fixtures for unit tests

#include "ValidateFixtures.h"
#include "UnitSPIRV.h"

#include <functional>
#include <tuple>
#include <utility>

namespace spvtest {

template <typename T, uint32_t OPTIONS>
ValidateBase<T, OPTIONS>::ValidateBase()
    : context_(spvContextCreate()), binary_(), diagnostic_() {}

template <typename T, uint32_t OPTIONS>
ValidateBase<T, OPTIONS>::~ValidateBase() {
  spvContextDestroy(context_);
}

template <typename T, uint32_t OPTIONS>
spv_const_binary ValidateBase<T, OPTIONS>::get_const_binary() {
  return spv_const_binary(binary_);
}

template <typename T, uint32_t OPTIONS>
void ValidateBase<T, OPTIONS>::TearDown() {
  if (diagnostic_) {
    spvDiagnosticPrint(diagnostic_);
  }
  spvDiagnosticDestroy(diagnostic_);
  spvBinaryDestroy(binary_);
}

template <typename T, uint32_t OPTIONS>
void ValidateBase<T, OPTIONS>::CompileSuccessfully(std::string code) {
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(context_, code.c_str(), code.size(),
                                         &binary_, &diagnostic))
      << "ERROR: " << diagnostic->error
      << "\nSPIR-V could not be compiled into binary:\n"
      << code;
}

template <typename T, uint32_t OPTIONS>
spv_result_t ValidateBase<T, OPTIONS>::ValidateInstructions() {
  return spvValidate(context_, get_const_binary(), validation_options_,
                     &diagnostic_);
}

template <typename T, uint32_t OPTIONS>
std::string ValidateBase<T, OPTIONS>::getDiagnosticString() {
  return std::string(diagnostic_->error);
}

template <typename T, uint32_t OPTIONS>
spv_position_t ValidateBase<T, OPTIONS>::getErrorPosition() {
  return diagnostic_->position;
}

template class spvtest::ValidateBase<std::pair<std::string, bool>,
                                     SPV_VALIDATE_SSA_BIT |
                                         SPV_VALIDATE_LAYOUT_BIT>;
template class spvtest::ValidateBase<bool, SPV_VALIDATE_SSA_BIT>;
template class spvtest::ValidateBase<
    std::tuple<int, std::tuple<std::string, std::function<spv_result_t(int)>,
                               std::function<spv_result_t(int)>>>,
    SPV_VALIDATE_LAYOUT_BIT>;
template class spvtest::ValidateBase<int, SPV_VALIDATE_LAYOUT_BIT |
                                              SPV_VALIDATE_ID_BIT>;

template class spvtest::ValidateBase<
    std::tuple<std::string, std::pair<std::string, std::vector<std::string>>>,
    SPV_VALIDATE_INSTRUCTION_BIT>;

template class spvtest::ValidateBase<
    std::string, SPV_VALIDATE_LAYOUT_BIT | SPV_VALIDATE_INSTRUCTION_BIT>;
}
