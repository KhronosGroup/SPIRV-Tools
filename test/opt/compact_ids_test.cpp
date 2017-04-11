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

#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using CompactIdsTest = PassTest<::testing::Test>;

TEST_F(CompactIdsTest, PassOff) {
  const std::string before =
R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%99 = OpTypeInt 32 0
%10 = OpTypeVector %99 2
%20 = OpConstant %99 2
%30 = OpTypeArray %99 %20
)";

  const std::string after = before;

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::NullPass>(before, after, false, false);
}

TEST_F(CompactIdsTest, PassOn) {
  const std::string before =
R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%99 = OpTypeInt 32 0
%10 = OpTypeVector %99 2
%20 = OpConstant %99 2
%30 = OpTypeArray %99 %20
)";

  const std::string after =
R"(OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%1 = OpTypeInt 32 0
%2 = OpTypeVector %1 2
%3 = OpConstant %1 2
%4 = OpTypeArray %1 %3
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  SinglePassRunAndCheck<opt::CompactIdsPass>(before, after, false, false);
}

}  // anonymous namespace
