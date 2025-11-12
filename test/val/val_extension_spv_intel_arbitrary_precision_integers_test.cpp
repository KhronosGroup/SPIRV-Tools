// Copyright (c) 2025 The Khronos Group Inc.
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

// Tests for SPV_INTEL_arbitrary_precision_integers extension

#include <string>

#include "gmock/gmock.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;

using ValidateIntelArbitraryPrecisionIntegers = spvtest::ValidateBase<bool>;

TEST_F(ValidateIntelArbitraryPrecisionIntegers, ArbitraryPrecisionIntegerWithoutExtension) {
  const std::string spirv = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %int19 = OpTypeInt 19 1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Invalid number of bits (19) used for OpTypeInt"));
}

TEST_F(ValidateIntelArbitraryPrecisionIntegers, ArbitraryPrecisionIntegerWithExtension) {
  const std::string spirv = R"(
               OpCapability Shader
               OpCapability ArbitraryPrecisionIntegersINTEL
               OpExtension "SPV_INTEL_arbitrary_precision_integers"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %int19 = OpTypeInt 19 1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIntelArbitraryPrecisionIntegers, ArbitraryPrecisionIntegerVariousBitWidths) {
  const std::string spirv = R"(
               OpCapability Shader
               OpCapability ArbitraryPrecisionIntegersINTEL
               OpExtension "SPV_INTEL_arbitrary_precision_integers"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int1 = OpTypeInt 1 1
        %int3 = OpTypeInt 3 0
        %int7 = OpTypeInt 7 1
       %int13 = OpTypeInt 13 0
       %int19 = OpTypeInt 19 1
       %int33 = OpTypeInt 33 0
       %int65 = OpTypeInt 65 1
      %int128 = OpTypeInt 128 0
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateIntelArbitraryPrecisionIntegers, ArbitraryPrecisionIntegerWithCapabilityOnly) {
  const std::string spirv = R"(
               OpCapability Shader
               OpCapability ArbitraryPrecisionIntegersINTEL
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %int19 = OpTypeInt 19 1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_MISSING_EXTENSION, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("ArbitraryPrecisionIntegersINTEL(5844) requires one of these extensions: SPV_INTEL_arbitrary_precision_integers"));
}

TEST_F(ValidateIntelArbitraryPrecisionIntegers, StandardIntegerTypesStillWork) {
  const std::string spirv = R"(
               OpCapability Shader
               OpCapability ArbitraryPrecisionIntegersINTEL
               OpExtension "SPV_INTEL_arbitrary_precision_integers"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %int32 = OpTypeInt 32 1
       %uint32 = OpTypeInt 32 0
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

}  // namespace
}  // namespace val
}  // namespace spvtools 