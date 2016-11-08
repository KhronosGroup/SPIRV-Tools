// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Common validation fixtures for unit tests

#ifndef LIBSPIRV_TEST_VALIDATE_FIXTURES_H_
#define LIBSPIRV_TEST_VALIDATE_FIXTURES_H_

#include "TestFixture.h"
#include "UnitSPIRV.h"

#include <algorithm>
#include <string>
#include <vector>

namespace spvtest {

template <typename T>
class ValidateBase : public ::testing::Test,
                     public ::testing::WithParamInterface<T> {
 public:
  ValidateBase() : diagnostic_() {}

  virtual void TearDown() {
    if (diagnostic_) {
      spvDiagnosticPrint(diagnostic_);
    }
    spvDiagnosticDestroy(diagnostic_);
  }

  // Returns the a spv_const_binary struct

  void CompileSuccessfully(std::string code,
                           spv_target_env env = SPV_ENV_UNIVERSAL_1_0) {
    spv_diagnostic diagnostic = nullptr;
    spv_binary binary;
    ASSERT_EQ(SPV_SUCCESS,
              spvTextToBinary(ScopedContext(env).context, code.c_str(),
                              code.size(), &binary, &diagnostic))
        << "ERROR: " << diagnostic->error
        << "\nSPIR-V could not be compiled into binary:\n"
        << code;

    words_.reserve(binary->wordCount);
    std::copy(binary->code, binary->code + binary->wordCount,
         std::back_inserter(words_));
  }

  void SetBinary(std::vector<uint32_t> &&words) {
    words_ = move(words);
  }

  // Performs validation on the SPIR-V code and compares the result of the
  // spvValidate function
  spv_result_t ValidateInstructions(
      spv_target_env env = SPV_ENV_UNIVERSAL_1_0) {
    return spvValidateBinary(ScopedContext(env).context,
                       words_.data(), words_.size(),
                       &diagnostic_);
  }

  std::string getDiagnosticString() {
    //std::cout << diagnostic_->error << std::endl;
    return std::string(diagnostic_->error); }
  spv_position_t getErrorPosition() { return diagnostic_->position; }

  spv_diagnostic diagnostic_;
  std::vector<uint32_t> words_;
};
}
#endif
