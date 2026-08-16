// Copyright (c) 2026 The Khronos Group Inc.
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

#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using NonWritablePropagationTest = opt::PassTest<::testing::Test>;

// A storage buffer whose every member is NonWritable gets the decoration on
// the variable as well.  This is the case DXC produces.
TEST_F(NonWritablePropagationTest, PropagatesToStorageBuffer) {
  const std::string text = R"(
; CHECK: OpDecorate %var NonWritable
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %var "var"
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
OpMemberDecorate %buf 0 NonWritable
OpMemberDecorate %buf 1 Offset 4
OpMemberDecorate %buf 1 NonWritable
OpDecorate %var DescriptorSet 0
OpDecorate %var Binding 0
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%float = OpTypeFloat 32
%buf = OpTypeStruct %float %float
%ptr = OpTypePointer StorageBuffer %buf
%var = OpVariable %ptr StorageBuffer
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::NonWritablePropagationPass>(text, true);
}

// The same, for a uniform block rather than a storage buffer.
TEST_F(NonWritablePropagationTest, PropagatesToUniformBuffer) {
  const std::string text = R"(
; CHECK: OpDecorate %var NonWritable
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %var "var"
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
OpMemberDecorate %buf 0 NonWritable
OpDecorate %var DescriptorSet 0
OpDecorate %var Binding 0
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%float = OpTypeFloat 32
%buf = OpTypeStruct %float
%ptr = OpTypePointer Uniform %buf
%var = OpVariable %ptr Uniform
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::NonWritablePropagationPass>(text, true);
}

// The pre-1.3 spelling of a storage buffer: Uniform storage class with
// BufferBlock rather than Block.
TEST_F(NonWritablePropagationTest, PropagatesToBufferBlock) {
  const std::string text = R"(
; CHECK: OpDecorate %var NonWritable
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %var "var"
OpDecorate %buf BufferBlock
OpMemberDecorate %buf 0 Offset 0
OpMemberDecorate %buf 0 NonWritable
OpDecorate %var DescriptorSet 0
OpDecorate %var Binding 0
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%float = OpTypeFloat 32
%buf = OpTypeStruct %float
%ptr = OpTypePointer Uniform %buf
%var = OpVariable %ptr Uniform
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::NonWritablePropagationPass>(text, true);
}

// One writable member means the buffer as a whole is writable.  This is the
// case an over-eager implementation would get wrong.
TEST_F(NonWritablePropagationTest, DoesNotPropagateWhenOneMemberIsWritable) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate %var NonWritable
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %var "var"
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
OpMemberDecorate %buf 0 NonWritable
OpMemberDecorate %buf 1 Offset 4
OpDecorate %var DescriptorSet 0
OpDecorate %var Binding 0
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%float = OpTypeFloat 32
%buf = OpTypeStruct %float %float
%ptr = OpTypePointer StorageBuffer %buf
%var = OpVariable %ptr StorageBuffer
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::NonWritablePropagationPass>(text, true);
}

// A variable that is already decorated must be left alone.  This is checked
// on the reported status rather than the output: duplicate decorations are
// filtered during serialization, so re-adding one is invisible in the binary
// but would still report a change and needlessly invalidate analyses.
TEST_F(NonWritablePropagationTest, DoesNotReportChangeWhenAlreadyDecorated) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %var "var"
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
OpMemberDecorate %buf 0 NonWritable
OpDecorate %var NonWritable
OpDecorate %var DescriptorSet 0
OpDecorate %var Binding 0
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%float = OpTypeFloat 32
%buf = OpTypeStruct %float
%ptr = OpTypePointer StorageBuffer %buf
%var = OpVariable %ptr StorageBuffer
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunToBinary<opt::NonWritablePropagationPass>(text, true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

// An empty struct is vacuously "all members NonWritable".  Decorating on that
// basis would assert read-only with no evidence, so it is left alone.
TEST_F(NonWritablePropagationTest, DoesNotPropagateForEmptyStruct) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate %var NonWritable
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %var "var"
OpDecorate %buf Block
OpDecorate %var DescriptorSet 0
OpDecorate %var Binding 0
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%buf = OpTypeStruct
%ptr = OpTypePointer StorageBuffer %buf
%var = OpVariable %ptr StorageBuffer
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::NonWritablePropagationPass>(text, true);
}

// NonWritable is only legal on a variable pointing at a buffer or storage
// image, so a Private variable must be left alone even though its struct is
// fully decorated.
TEST_F(NonWritablePropagationTest, DoesNotPropagateForNonBufferVariable) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate %var NonWritable
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %var "var"
OpMemberDecorate %buf 0 NonWritable
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%float = OpTypeFloat 32
%buf = OpTypeStruct %float
%ptr = OpTypePointer Private %buf
%var = OpVariable %ptr Private
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::NonWritablePropagationPass>(text, true);
}

// A descriptor array of buffers carries Block on the struct, one array layer
// down.  The buffer helpers unpack that layer, so this propagates too.
TEST_F(NonWritablePropagationTest, PropagatesThroughDescriptorArray) {
  const std::string text = R"(
; CHECK: OpDecorate %var NonWritable
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %var "var"
OpDecorate %buf Block
OpMemberDecorate %buf 0 Offset 0
OpMemberDecorate %buf 0 NonWritable
OpDecorate %var DescriptorSet 0
OpDecorate %var Binding 0
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%buf = OpTypeStruct %float
%arr = OpTypeArray %buf %uint_2
%ptr = OpTypePointer StorageBuffer %arr
%var = OpVariable %ptr StorageBuffer
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::NonWritablePropagationPass>(text, true);
}

}  // namespace
