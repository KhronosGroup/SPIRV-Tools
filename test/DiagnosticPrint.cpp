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

#include "UnitSPIRV.h"

namespace {

TEST(DiagnosticPrint, Default) {
  char message[] = "Test Diagnostic!";
  spv_diagnostic_t diagnostic = {{2, 3, 5}, message};
  // TODO: Redirect stderr
  ASSERT_EQ(SPV_SUCCESS, spvDiagnosticPrint(&diagnostic));
  // TODO: Validate the output of spvDiagnosticPrint()
  // TODO: Remove the redirection of stderr
}

TEST(DiagnosticPrint, InvalidDiagnostic) {
  ASSERT_EQ(SPV_ERROR_INVALID_DIAGNOSTIC, spvDiagnosticPrint(nullptr));
}

// TODO(dneto): We should be able to redirect the diagnostic printing.
// Once we do that, we can test diagnostic corner cases.

}  // anonymous namespace
