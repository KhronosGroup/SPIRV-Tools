// Copyright (c) 2017 The Khronos Group Inc.
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

// Validation tests for illegal instructions

#include <sstream>
#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "val_fixtures.h"

using ::testing::HasSubstr;

using ValidateIns = spvtest::ValidateBase<std::string>;

namespace {

TEST_F(ValidateIns, OpImageSparseSampleProjImplicitLod) {
  char str[] = R"(
             OpCapability Shader
             OpCapability Linkage
             OpCapability Sampled1D
             OpCapability SparseResidency
             OpMemoryModel Logical GLSL450
%void      = OpTypeVoid
%int       = OpTypeInt 32 0
%float     = OpTypeFloat 32
%floatp    = OpTypePointer Input %float
%fnt       = OpTypeFunction %void
%imgt      = OpTypeImage %float 1D 0 0 0 0 Unknown
%sampledt  = OpTypeSampledImage %imgt
%sampledp  = OpTypePointer Uniform %sampledt
%img       = OpVariable %sampledp Uniform
%coord     = OpVariable %floatp Input
%func      = OpFunction %void None %fnt
%label     = OpLabel
%sample    = OpImageSparseSampleProjImplicitLod %float %img %coord
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_ERROR_RESERVED_OPCODE, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Opcode ImageSparseSampleProjImplicitLod is reserved "
                        "for future use."));
}

TEST_F(ValidateIns, OpImageSparseSampleProjExplicitLod) {
  char str[] = R"(
             OpCapability Shader
             OpCapability Linkage
             OpCapability Sampled1D
             OpCapability SparseResidency
             OpMemoryModel Logical GLSL450
%void      = OpTypeVoid
%int       = OpTypeInt 32 0
%float     = OpTypeFloat 32
%floatp    = OpTypePointer Input %float
%fnt       = OpTypeFunction %void
%imgt      = OpTypeImage %float 1D 0 0 0 0 Unknown
%sampledt  = OpTypeSampledImage %imgt
%sampledp  = OpTypePointer Uniform %sampledt
%img       = OpVariable %sampledp Uniform
%coord     = OpVariable %floatp Input
%lod       = OpVariable %floatp Input
%func      = OpFunction %void None %fnt
%label     = OpLabel
%sample    = OpImageSparseSampleProjExplicitLod %float %img %coord Lod %lod
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_ERROR_RESERVED_OPCODE, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Opcode ImageSparseSampleProjExplicitLod is reserved "
                        "for future use."));
}

TEST_F(ValidateIns, OpImageSparseSampleProjDrefImplicitLod) {
  char str[] = R"(
             OpCapability Shader
             OpCapability Linkage
             OpCapability Sampled1D
             OpCapability SparseResidency
             OpMemoryModel Logical GLSL450
%void      = OpTypeVoid
%int       = OpTypeInt 32 0
%float     = OpTypeFloat 32
%floatp    = OpTypePointer Input %float
%fnt       = OpTypeFunction %void
%imgt      = OpTypeImage %float 1D 0 0 0 0 Unknown
%sampledt  = OpTypeSampledImage %imgt
%sampledp  = OpTypePointer Uniform %sampledt
%img       = OpVariable %sampledp Uniform
%coord     = OpVariable %floatp Input
%dref      = OpVariable %floatp Input
%func      = OpFunction %void None %fnt
%label     = OpLabel
%sample    = OpImageSparseSampleProjDrefImplicitLod %float %img %coord %dref
             OpReturn
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_ERROR_RESERVED_OPCODE, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Opcode ImageSparseSampleProjDrefImplicitLod is "
                        "reserved for future use."));
}

TEST_F(ValidateIns, OpImageSparseSampleProjDrefExplicitLod) {
  char str[] = R"(
             OpCapability Shader
             OpCapability Linkage
             OpCapability Sampled1D
             OpCapability SparseResidency
             OpMemoryModel Logical GLSL450
%void      = OpTypeVoid
%int       = OpTypeInt 32 0
%float     = OpTypeFloat 32
%floatp    = OpTypePointer Input %float
%fnt       = OpTypeFunction %void
%imgt      = OpTypeImage %float 1D 0 0 0 0 Unknown
%sampledt  = OpTypeSampledImage %imgt
%sampledp  = OpTypePointer Uniform %sampledt
%img       = OpVariable %sampledp Uniform
%coord     = OpVariable %floatp Input
%lod       = OpVariable %floatp Input
%dref      = OpVariable %floatp Input
%func      = OpFunction %void None %fnt
%label     = OpLabel
%sample    = OpImageSparseSampleProjDrefExplicitLod %float %img %coord %dref Lod %lod
             OpReturn
             OpFunctionEnd
             OpFunctionEnd
)";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_ERROR_RESERVED_OPCODE, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Opcode ImageSparseSampleProjDrefExplicitLod is "
                        "reserved for future use."));
}
}
