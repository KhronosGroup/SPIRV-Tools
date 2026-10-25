// Copyright (c) 2026 Lee Gao
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

#include <string>
#include <tuple>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using MaliPerVertexBlockTest = PassTest<::testing::Test>;

size_t CountOccurrences(const std::string& text, const std::string& needle) {
  size_t count = 0;
  for (size_t pos = text.find(needle); pos != std::string::npos;
       pos = text.find(needle, pos + 1)) {
    ++count;
  }
  return count;
}

TEST_F(MaliPerVertexBlockTest, FoldsLooseGeometryOutput) {
  const std::string text = R"(
               OpCapability Geometry
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %pos
               OpExecutionMode %main InputPoints
               OpExecutionMode %main OutputPoints
               OpExecutionMode %main OutputVertices 1
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%ptr_out_v4float = OpTypePointer Output %v4float
        %pos = OpVariable %ptr_out_v4float Output
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliPerVertexBlockPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeStruct"));
  EXPECT_NE(std::string::npos, disassembly.find("OpMemberDecorate"));
  EXPECT_NE(std::string::npos, disassembly.find("BuiltIn Position"));
  EXPECT_NE(std::string::npos, disassembly.find("Block"));
}

TEST_F(MaliPerVertexBlockTest, LeavesVertexShader) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %pos
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%ptr_out_v4float = OpTypePointer Output %v4float
        %pos = OpVariable %ptr_out_v4float Output
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliPerVertexBlockPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(MaliPerVertexBlockTest, FoldsGeometryInputAndOutput) {
  const std::string text = R"(
               OpCapability Geometry
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %in_pos %out_pos
               OpExecutionMode %main InputPoints
               OpExecutionMode %main OutputPoints
               OpExecutionMode %main OutputVertices 1
               OpDecorate %in_pos BuiltIn Position
               OpDecorate %out_pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
   %v4_array = OpTypeArray %v4float %uint_1
%ptr_in_v4_array = OpTypePointer Input %v4_array
%ptr_out_v4float = OpTypePointer Output %v4float
     %in_pos = OpVariable %ptr_in_v4_array Input
    %out_pos = OpVariable %ptr_out_v4float Output
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliPerVertexBlockPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(2u, CountOccurrences(disassembly, "OpTypeStruct"));
  EXPECT_EQ(2u, CountOccurrences(disassembly, "BuiltIn Position"));
  EXPECT_EQ(2u, CountOccurrences(disassembly, " Block"));
  // The input keeps its array shape, with the block as the element type.
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray"));
}

TEST_F(MaliPerVertexBlockTest, IgnoresOrdinaryInterfaceVariables) {
  const std::string text = R"(
               OpCapability Geometry
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %pos %color
               OpExecutionMode %main InputPoints
               OpExecutionMode %main OutputPoints
               OpExecutionMode %main OutputVertices 1
               OpDecorate %pos BuiltIn Position
               OpDecorate %color Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%ptr_out_v4float = OpTypePointer Output %v4float
        %pos = OpVariable %ptr_out_v4float Output
      %color = OpVariable %ptr_out_v4float Output
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliPerVertexBlockPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(1u, CountOccurrences(disassembly, "OpTypeStruct"));
  // The unrelated Location variable stays in the interface.
  EXPECT_NE(std::string::npos, disassembly.find("Location 0"));
  EXPECT_EQ(2u, CountOccurrences(disassembly, "OpVariable"));
}

TEST_F(MaliPerVertexBlockTest, FoldsTessControlArrayedOutput) {
  const std::string text = R"(
               OpCapability Tessellation
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint TessellationControl %main "main" %pos
               OpExecutionMode %main OutputVertices 3
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_3 = OpConstant %uint 3
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
   %v4_array = OpTypeArray %v4float %uint_3
%ptr_out_v4_array = OpTypePointer Output %v4_array
        %pos = OpVariable %ptr_out_v4_array Output
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliPerVertexBlockPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(1u, CountOccurrences(disassembly, "OpTypeStruct"));
  EXPECT_EQ(1u, CountOccurrences(disassembly, " Block"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray"));
}

TEST_F(MaliPerVertexBlockTest, KeepsInvariantDecoration) {
  const std::string text = R"(
               OpCapability Geometry
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %pos
               OpExecutionMode %main InputPoints
               OpExecutionMode %main OutputPoints
               OpExecutionMode %main OutputVertices 1
               OpDecorate %pos BuiltIn Position
               OpDecorate %pos Invariant
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%ptr_out_v4float = OpTypePointer Output %v4float
        %pos = OpVariable %ptr_out_v4float Output
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliPerVertexBlockPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::string::npos, std::get<0>(result).find("Invariant"));
}

TEST_F(MaliPerVertexBlockTest, LeavesUnsupportedUse) {
  const std::string text = R"(
               OpCapability Geometry
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %pos
               OpExecutionMode %main InputPoints
               OpExecutionMode %main OutputPoints
               OpExecutionMode %main OutputVertices 1
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%ptr_out_v4float = OpTypePointer Output %v4float
        %pos = OpVariable %ptr_out_v4float Output
       %main = OpFunction %void None %fn
      %label = OpLabel
       %copy = OpCopyObject %ptr_out_v4float %pos
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliPerVertexBlockPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
